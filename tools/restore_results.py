"""Restore the exact original aggregated JSON. Does not train or overwrite different data."""
from pathlib import Path
import argparse,gzip,hashlib,json

ROOT=Path(__file__).resolve().parents[1]


def restore():
    manifest=json.loads((ROOT/'provenance/source_manifest.json').read_text(encoding='utf-8'))
    count=0
    for entry in manifest['files']:
        if entry['mode']!='lossless_gzip':continue
        source=ROOT/entry['path'];target=ROOT/entry['original_relative']
        assert target.resolve().is_relative_to(ROOT)
        raw=gzip.decompress(source.read_bytes())
        assert hashlib.sha256(raw).hexdigest()==entry['original_sha256'],source
        if target.exists():
            if target.read_bytes()!=raw:raise RuntimeError(f'Refusing to overwrite changed local results: {target}')
        else:target.write_bytes(raw)
        count+=1
    return count


if __name__=='__main__':print('Exact aggregated JSON restored/verified:',restore())
