"""Install only missing active EvalScope dependency requirements, preserving system torch."""
from importlib.metadata import requires,version,PackageNotFoundError
from packaging.requirements import Requirement
import subprocess
from pathlib import Path

for attempt in range(8):
    queue=['evalscope'];seen=set();missing=[]
    while queue:
        name=queue.pop()
        if name.lower() in seen or name.lower()=='torch':continue
        seen.add(name.lower())
        try: dependencies=requires(name) or []
        except PackageNotFoundError:continue
        for raw in dependencies:
            req=Requirement(raw)
            if req.marker and not req.marker.evaluate({'extra':''}):continue
            if req.name.lower()=='torch':continue
            try:
                current=version(req.name)
                if not req.specifier.contains(current,prereleases=True):missing.append(str(req))
            except PackageNotFoundError:missing.append(str(req))
            queue.append(req.name)
    missing=sorted(set(missing))
    if not missing:
        print('Active EvalScope dependency closure satisfied (system torch preserved).',flush=True);break
    file=Path('/tmp/rmt-eval-transitive.txt');file.write_text('\n'.join(missing)+'\n')
    print('Missing dependency requirements:',missing,flush=True)
    subprocess.run(['uv','pip','install','--python',str(Path('.venv-eval/bin/python')),'--no-deps','-r',str(file)],check=True)
else:raise RuntimeError('Dependency closure did not converge')
