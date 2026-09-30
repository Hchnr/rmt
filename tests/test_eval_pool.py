"""The evaluation proxy must preserve requests and reject mismatched replicas."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import requests


def test_proxy_preserves_payload_and_checks_metadata(tmp_path):
    seen=[]
    class Backend(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,value):
            payload=json.dumps(value).encode();self.send_response(200);self.end_headers();self.wfile.write(payload)
        def do_GET(self):self.send({'checkpoint':self.server.checkpoint})
        def do_POST(self):
            value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.server.server_port,value));time.sleep(0.05);self.send(value)
    servers=[ThreadingHTTPServer(('127.0.0.1',0),Backend) for _ in range(2)]
    for server in servers:
        server.checkpoint='same';threading.Thread(target=server.serve_forever,daemon=True).start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    script=str(Path(__file__).parents[1]/'scripts/serve_eval_pool.py')
    cmd=[sys.executable,script,'--ports',','.join(str(s.server_port) for s in servers),'--port',str(port)]
    session=requests.Session();session.trust_env=False
    proc=None
    try:
        proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        for _ in range(100):
            try:session.get(f'http://127.0.0.1:{port}/health',timeout=0.1).raise_for_status();break
            except requests.RequestException:time.sleep(0.02)
        else:raise AssertionError('Pool not ready')
        def send(i):
            client=requests.Session();client.trust_env=False
            body={'seed':i,'messages':[{'role':'user','content':'exact text'}]}
            result=client.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=body,timeout=5)
            result.raise_for_status();assert result.json()==body
        with ThreadPoolExecutor(4) as pool:list(pool.map(send,range(4)))
        assert len(seen)==4 and len({p for p,_ in seen})==2
        assert session.get(f'http://127.0.0.1:{port}/metadata').json()['replica_pool']['count']==2
        proc.terminate();proc.wait(timeout=5);proc=None
        servers[1].checkpoint='different'
        mismatch=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5)
        assert mismatch.returncode!=0 and b'Replica metadata mismatch' in mismatch.stderr
    finally:
        if proc:proc.terminate();proc.wait(timeout=5)
        for server in servers:server.shutdown();server.server_close()
