"""Resume verified byte ranges when a single HF stream is bandwidth limited."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import requests


def main():
    root=Path('artifacts/v0.0.4_dynamic_recurr/teacher/Qwen3-32B')
    repo='Qwen/Qwen3-32B';rev='9216db5781bf21249d130ec9da846c4624c16137'
    meta=requests.get(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true',timeout=30)
    meta.raise_for_status();files=meta.json()['siblings']
    chunk=8*1024*1024;lock=threading.Lock();work=[];descriptors={};manifests={}
    for item in files:
        name=item['rfilename']
        if not name.endswith('.safetensors'):continue
        size=item['lfs']['size'];path=root/name
        state_path=Path(str(path)+'.ranges.json');part=Path(str(path)+'.part')
        if path.exists():continue
        if state_path.exists():state=json.loads(state_path.read_text())
        else:
            prefix=part.stat().st_size if part.exists() else 0
            state={'size':size,'chunk':chunk,'done':list(range(prefix//chunk))}
            state_path.write_text(json.dumps(state))
        fd=os.open(part,os.O_RDWR|os.O_CREAT,0o644);os.ftruncate(fd,size)
        descriptors[name]=fd;manifests[name]=(state,state_path,item)
        for index,start in enumerate(range(0,size,chunk)):
            if index not in state['done']:work.append((name,index,start,min(size-1,start+chunk-1)))
    print(json.dumps({'pending_ranges':len(work),'workers':48,'chunk_bytes':chunk}),flush=True)
    def fetch(task):
        name,index,start,end=task
        for attempt in range(5):
            try:
                with requests.get(f'https://huggingface.co/{repo}/resolve/{rev}/{name}?rmt_range={start}',
                    headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,90)) as response:
                    expected=f'bytes {start}-{end}/{manifests[name][0]["size"]}'
                    if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError('Unexpected range response')
                    data=response.content
                    if len(data)!=end-start+1:raise ValueError('Incomplete range')
                    written=os.pwrite(descriptors[name],data,start)
                    if written!=len(data):raise IOError('Short write')
                with lock:
                    state,state_path,_=manifests[name];state['done'].append(index)
                    temp=Path(str(state_path)+'.tmp');temp.write_text(json.dumps(state));temp.replace(state_path)
                return True
            except Exception as error:
                if attempt==4:print(json.dumps({'file':name,'range':index,'error_type':type(error).__name__}),flush=True)
                else:time.sleep(min(2**attempt,8))
        return False
    done=0;failed=0;start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(48) as pool:
        for success in pool.map(fetch,work):
            done+=1;failed+=not success
            if done%32==0:print(json.dumps({'completed_ranges':done,'failed':failed,'seconds':time.monotonic()-start}),flush=True)
    for fd in descriptors.values():os.close(fd)
    if failed:raise RuntimeError(f'{failed} ranges failed; retained verified progress for explicit retry')
    for name,(state,state_path,item) in manifests.items():
        part=root/(name+'.part');h=hashlib.sha256()
        with part.open('rb') as f:
            for block in iter(lambda:f.read(chunk),b''):h.update(block)
        if h.hexdigest()!=item['lfs']['sha256']:raise ValueError(f'Final checksum mismatch: {name}')
        part.replace(root/name)
        print(json.dumps({'file':name,'sha256':h.hexdigest(),'status':'verified'}),flush=True)


if __name__=='__main__':main()
