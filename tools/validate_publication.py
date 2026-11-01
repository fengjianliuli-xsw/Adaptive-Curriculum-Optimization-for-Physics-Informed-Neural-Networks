"""Check publication integrity, readable animations, README targets and source preservation."""
from pathlib import Path
import ast,gzip,hashlib,json,re
from urllib.parse import unquote
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'provenance/source_manifest.json').read_text(encoding='utf-8'))
    archives=0;unchanged=0
    for row in manifest['files']:
        path=ROOT/row['path'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['published_sha256'],path
        if row['mode']=='lossless_gzip':
            assert hashlib.sha256(gzip.decompress(raw)).hexdigest()==row['original_sha256'],path
            archives+=1
        if path.suffix=='.py':
            assert row['mode']=='verbatim'
            assert hashlib.sha256(raw).hexdigest()==row['original_sha256'];unchanged+=1
    readme=(ROOT/'README.md').read_text(encoding='utf-8');targets=re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)',readme)
    local=[];gifs=[]
    for url in targets:
        if url.startswith(('https:','http:','#','mailto:')):continue
        path=ROOT/unquote(url.split('#')[0]);assert path.exists(),url;local.append(url)
        if path.suffix=='.gif':
            with Image.open(path) as im:
                assert im.n_frames>1,(url,im.n_frames);im.seek(im.n_frames-1);im.load()
                gifs.append(dict(path=url,frames=im.n_frames,bytes=path.stat().st_size))
    assert len(gifs)>=8
    # Parse every Python source without creating __pycache__ or starting any experiments.
    python_files=[];large=[];secret_hits=[]
    patterns=[re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(r'github_pat_[A-Za-z0-9_]{40,}'),re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p in {'.git','.venv','__pycache__','tmp','test_tmp'} for p in path.relative_to(ROOT).parts):continue
        if path.stat().st_size>50*2**20:large.append(str(path.relative_to(ROOT)))
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path));python_files.append(str(path.relative_to(ROOT)))
        if path.suffix in ['.py','.md','.txt','.json','.yml','.yaml','.toml','.ps1']:
            text=path.read_text(encoding='utf-8-sig')
            if any(p.search(text) for p in patterns):secret_hits.append(str(path.relative_to(ROOT)))
    assert not large,large;assert not secret_hits,secret_hits
    result=dict(source_manifest_files=len(manifest['files']),original_python_sources_preserved=unchanged,gzip_aggregates_verified=archives,
                readme_local_targets_checked=len(local),readme_animations=gifs,python_sources_parsed=len(python_files),files_over_50MiB=large,credential_pattern_hits=secret_hits,
                training_started=False,notes=['All numerical data are historical results; no new optimizer/scheduler training for publication.','No credentials were queried or printed by this validation.'])
    (ROOT/'provenance/publication_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Validated snapshot files:',len(manifest['files']),'original Python sources:',unchanged,'gzip aggregates:',archives,'README links:',len(local),'GIFs:',len(gifs),'parsed Python files:',len(python_files))


if __name__=='__main__':main()
