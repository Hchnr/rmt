"""Repack existing, intact uv archives as wheels for an offline reproducible env.

This avoids registry downloads; it does not alter shared caches or installed packages.
Run using Python 3.13 (the ABI of the cached torch wheel). All selected packages and
source archive paths are recorded for review. Only active, non-extra deps are used.
"""
import argparse
import email
import json
from pathlib import Path
import zipfile
from packaging.requirements import Requirement
from packaging.tags import sys_tags, parse_tag
from packaging.utils import canonicalize_name
from packaging.version import Version

parser=argparse.ArgumentParser()
parser.add_argument('--cache',default='/opt/uv_cache/archive-v0')
parser.add_argument('--output',default='/tmp/rmt-wheelhouse')
parser.add_argument('--resolve-only',action='store_true')
args=parser.parse_args()
available={}
compatible=set(sys_tags())
for archive in Path(args.cache).iterdir():
    for meta in archive.glob('*.dist-info/METADATA'):
        wheel=meta.parent/'WHEEL'
        if not wheel.exists(): continue
        w=email.message_from_string(wheel.read_text())
        tags=w.get_all('Tag',[])
        if not any(parse_tag(t)&compatible for t in tags): continue
        m=email.message_from_string(meta.read_text())
        name=canonicalize_name(m['Name'])
        available.setdefault(name,[]).append((Version(m['Version']),m,w,archive))
requirements={}
roots=['torch==2.10.0+cu130','transformers==4.57.6','accelerate==1.13.0','pytest==8.4.2','PyYAML==6.0.3','safetensors==0.8.0']
queue=[Requirement(r) for r in roots]
chosen={}
while queue:
    req=queue.pop(0)
    if req.marker and not req.marker.evaluate({'extra':''}): continue
    name=canonicalize_name(req.name)
    requirements.setdefault(name,[]).append(req)
    candidates=[x for x in available.get(name,[]) if all(r.specifier.contains(x[0],prereleases=True) for r in requirements[name])]
    if not candidates:
        raise RuntimeError(f'No compatible cached wheel for {name}: {requirements[name]}')
    pick=max(candidates,key=lambda x:x[0])
    if name in chosen and chosen[name][0]==pick[0]: continue
    chosen[name]=pick
    queue.extend(Requirement(r) for r in pick[1].get_all('Requires-Dist',[]))
manifest={n:{'version':str(v[0]),'archive':str(v[3])} for n,v in sorted(chosen.items())}
print(json.dumps(manifest,indent=2),flush=True)
if args.resolve_only: raise SystemExit()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
for name,(version,meta,wheel,archive) in sorted(chosen.items()):
    tag=next(t for t in wheel.get_all('Tag') if parse_tag(t)&compatible)
    filename=f'{name.replace("-","_")}-{version}-{tag}.whl'
    dest=out/filename
    if dest.exists(): continue
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_STORED) as z:
        for f in archive.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:
                z.write(f,f.relative_to(archive))
    print('packed',filename,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
