"""Loopback-only least-busy proxy for identical single-GPU evaluation replicas."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import requests
p=argparse.ArgumentParser();p.add_argument('--ports',required=True);p.add_argument('--port',type=int,default=9010);a=p.parse_args()
ports=list(map(int,a.ports.split(',')));active={x:0 for x in ports};lock=threading.Lock()
session=requests.Session();session.trust_env=False
metadata=[session.get(f'http://127.0.0.1:{x}/metadata',timeout=10).json() for x in ports]
assert all(x==metadata[0] for x in metadata),'Replica metadata mismatch'
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def send(self,status,value):
  payload=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
 def do_GET(self):
  if self.path=='/metadata':self.send(200,{**metadata[0],'replica_pool':{'count':len(ports),'policy':'least_busy'}})
  elif self.path=='/health':self.send(200,{'ready':True})
  else:self.send(404,{'error':'Unknown endpoint'})
 def do_POST(self):
  if self.path!='/v1/chat/completions':self.send(404,{'error':'Unsupported endpoint'});return
  body=self.rfile.read(int(self.headers['Content-Length']))
  with lock:port=min(ports,key=lambda x:active[x]);active[port]+=1
  try:
   client=requests.Session();client.trust_env=False
   response=client.post(f'http://127.0.0.1:{port}{self.path}',data=body,headers={'Content-Type':'application/json'},timeout=7200)
   self.send(response.status_code,response.json())
  except requests.RequestException as error:self.send(502,{'error':type(error).__name__})
  finally:
   with lock:active[port]-=1
print(json.dumps({'ready':True,'port':a.port,'replicas':ports}),flush=True)
ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
