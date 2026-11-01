"""Maintainer helper: copy original experiment workspaces into this publication checkout.

Never writes to the source workspaces. Aggregated JSON files are losslessly gzipped;
per-method records, checkpoints and existing figures/animations remain unchanged.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil

REPO=Path(__file__).resolve().parents[1]
EXCLUDE_DIRS={'__pycache__','.pytest_cache','.venv','.cache','.git','tmp','test_tmp','node_modules'}


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,default=Path('F:/RA'))
    parser.add_argument('--legacy',type=Path,default=Path.home()/'Documents/ChatGPT/RA/pinn_optimizer_lab')
    args=parser.parse_args()
    roots={args.legacy.name:args.legacy,**{name:args.workspace/name for name in ['pinn_2d_optimizer_lab','hertz_cylinder_lab','pinn_scheduler_lab','reports']}}
    marker=REPO/'provenance/source_manifest.json'
    if marker.exists():raise SystemExit('Snapshot already exists. Review updates explicitly rather than overwrite the published snapshot.')
    records=[];omitted=[]
    for name,root in roots.items():
        assert root.is_dir(),root
        for source in sorted(root.rglob('*')):
            if not source.is_file():continue
            relative=source.relative_to(root)
            if any(p in EXCLUDE_DIRS for p in relative.parts) or source.suffix.lower() in ['.pyc','.pyo','.zip']:continue
            target=REPO/name/relative
            assert target.resolve().is_relative_to(REPO.resolve())
            if 'runs' in relative.parts and source.name in ['dashboard.html','data.js']:
                omitted.append({'path':f'{name}/{relative.as_posix()}','reason':'generated browser page; rebuild with tools/rebuild_dashboards.py'})
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            data=source.read_bytes();sha=digest(data)
            if 'runs' in relative.parts and source.name in ['results.json','data.json']:
                target=target.with_suffix(target.suffix+'.gz')
                target.write_bytes(gzip.compress(data,compresslevel=6,mtime=0))
                mode='lossless_gzip';assert digest(gzip.decompress(target.read_bytes()))==sha
            else:
                shutil.copy2(source,target);mode='verbatim'
            if name=='pinn_scheduler_lab' and source.name=='.gitignore':
                original=REPO/'provenance/original_sources'/name/'.gitignore';original.parent.mkdir(parents=True,exist_ok=True);original.write_bytes(data)
                text=data.decode('utf-8');text='\n'.join(line for line in text.splitlines() if line.strip()!='runs/')+'\n# Existing experiment records are intentionally versioned in this publication.\n'
                target.write_text(text,encoding='utf-8');mode='publication_ignore_rules'
            records.append(dict(path=target.relative_to(REPO).as_posix(),original_relative=f'{name}/{relative.as_posix()}',original_sha256=sha,published_sha256=digest(target.read_bytes()),original_bytes=len(data),published_bytes=target.stat().st_size,mode=mode))
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps(dict(format_version=1,original_workspace_locations={n:str(p) for n,p in roots.items()},files=records,omitted_generated_pages=omitted,notes=['No new training was run for publication.','Training/PDE/optimizer source and numerical records are copied verbatim.','Only the scheduler .gitignore is adjusted to version the existing runs; its original is archived.','Aggregated JSON is gzip-compressed without changing its bytes; per-method JSON/CSV/checkpoints are directly readable.','Existing report metadata and absolute provenance paths describe the original workspaces.']),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Snapshot files:',len(records),'MiB:',round(sum(r['published_bytes'] for r in records)/2**20,2),'generated pages omitted:',len(omitted))


if __name__=='__main__':main()
