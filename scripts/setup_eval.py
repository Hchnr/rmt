"""Create isolated evaluation environment without changing the model environment."""
from pathlib import Path
import os
import subprocess

subprocess.run(['uv','venv','--python','/usr/bin/python3.12','--system-site-packages','.venv-eval'],check=True)
subprocess.run(['uv','pip','install','--python','.venv-eval/bin/python','--no-deps','evalscope==1.0.0'],check=True)
subprocess.run(['.venv-eval/bin/python','scripts/complete_eval_environment.py'],check=True)
# The user supplied the trusted proxy for dataset downloads. NLTK requires an
# explicit opt-in; no proxy URL or credential is embedded in the repository.
env={**os.environ,'NLTK_ALLOW_PROXIED_URLOPEN':'1'}
subprocess.run(['.venv-eval/bin/python','-c',
    "import nltk; [nltk.download(n,download_dir='artifacts/v0.0.3/nltk_data',quiet=True,raise_on_error=True) for n in ['punkt','punkt_tab']]"],env=env,check=True)
Path('reports/v0.0.3').mkdir(parents=True,exist_ok=True)
with open('reports/v0.0.3/eval_environment.txt','w') as out:
    subprocess.run(['uv','pip','freeze','--python','.venv-eval/bin/python'],stdout=out,check=True)

import hashlib,json
lock=json.loads(Path('configs/eval/nltk_resources.json').read_text())
for name,expected in lock['sha256'].items():
    actual=hashlib.sha256((Path('artifacts/v0.0.3/nltk_data/tokenizers')/name).read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError(f'NLTK resource differs from evaluated version: {name}')
