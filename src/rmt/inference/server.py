"""Loopback OpenAI-compatible non-streaming service with actual microbatching."""
import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import queue
import threading
import time
import uuid
from .runner import Runner, Generation


class Batcher:
    def __init__(self,runner,size=4,wait=.015):
        self.runner,self.size,self.wait=runner,size,wait
        self.queue=queue.Queue(maxsize=256)
        threading.Thread(target=self.work,daemon=True).start()

    def submit(self,prompt,cfg):
        response=queue.Queue(maxsize=1)
        self.queue.put((prompt,cfg,response),timeout=5)
        result=response.get(timeout=3600)
        if isinstance(result,Exception): raise result
        return result

    def execute(self,jobs,retries=0):
        try:
            outputs=self.runner.generate([j[0] for j in jobs],[j[1] for j in jobs])
            for job,out in zip(jobs,outputs):
                out['batch_split_retries']=retries
                job[2].put(out)
        except Exception as error:
            import torch
            split_needed=isinstance(error,torch.cuda.OutOfMemoryError) or (isinstance(error,ValueError) and 'exceeds configured context' in str(error))
            if split_needed and len(jobs)>1:
                # The traceback holds failed generation tensors/cache alive.
                # Release it before attempting a smaller batch.
                error.__traceback__=None
                torch.cuda.empty_cache()
                mid=len(jobs)//2;self.execute(jobs[:mid],retries+1);self.execute(jobs[mid:],retries+1)
            else:
                logging.exception("Inference batch failed (size=%s)",len(jobs))
                for job in jobs: job[2].put(error)

    def work(self):
        while True:
            jobs=[self.queue.get()];deadline=time.monotonic()+self.wait
            while len(jobs)<self.size:
                try: jobs.append(self.queue.get(timeout=max(0,deadline-time.monotonic())))
                except queue.Empty: break
            # Bucket queued requests by rendered prompt length, keeping each
            # response queue attached so sorting cannot change request identity.
            jobs.sort(key=lambda j:len(j[0]))
            self.execute(jobs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--backend',choices=['rmt','qwen'],default='rmt')
    p.add_argument('--port',type=int,default=8801);p.add_argument('--name',default='rmt')
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--max-context',type=int,default=4096)
    p.add_argument('--eager',action='store_true');p.add_argument('--attention',default='eager')
    args=p.parse_args()
    runner=Runner(args.model,args.backend,not args.eager,max_context=args.max_context,attention=args.attention)
    batcher=Batcher(runner,args.batch_size)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,code,body):
            payload=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path=='/health': self.send(200,{'status':'ok'});return
            if self.path=='/v1/models': self.send(200,{'object':'list','data':[{'id':args.name,'object':'model'}]});return
            if self.path=='/metadata':
                self.send(200,{'model':args.model,'backend':args.backend,'compiled':not args.eager,
                    'generation_termination':{'eos_token_ids':sorted(runner.eos),'pad_token_id':runner.pad},
                    'cache':'recurrence_capacity' if args.backend=='rmt' else 'hf_dynamic',
                    'max_context':runner.max_context,'batch_size':args.batch_size,'source_sha256':runner.source_hashes,
                    'routing_mode':getattr(runner.model.config,'routing_mode',None),'attention':args.attention,
                    'recurrence_config':{key:getattr(runner.model.config,key,None) for key in (
                        'num_experts','num_recurrences','max_recurrences','min_recurrences','halting_policy',
                        'halt_patience','halt_threshold','halt_relative_threshold','halt_probability_threshold',
                        'recurrence_schedule','tail_experts','learned_routing_start','defer_probability_checks')},
                    'engine_environment':runner.engine_environment});return
            self.send(404,{'error':{'message':'Unknown endpoint'}})
        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4*1024*1024: raise ValueError('Invalid body length')
                body=json.loads(self.rfile.read(length))
                if self.path not in ['/v1/chat/completions','/v1/completions']: raise ValueError('Unknown endpoint')
                for key in ['stream','tools','logprobs']:
                    if body.get(key): raise ValueError(f'{key} is not supported')
                if body.get('n',1)!=1: raise ValueError('n must be 1; issue separate seeded requests')
                if body.get('model',args.name)!=args.name: raise ValueError('Unknown model')
                stop=body.get('stop') or []
                if isinstance(stop,str): stop=[stop]
                thinking=body.get('chat_template_kwargs',{}).get('enable_thinking',False)
                cfg=Generation(max_new_tokens=body.get('max_completion_tokens',body.get('max_tokens',512)),
                    temperature=body.get('temperature',.7),top_p=body.get('top_p',.8),top_k=body.get('top_k',20),
                    presence_penalty=body.get('presence_penalty',1.5),seed=body.get('seed',17),
                    stop=tuple(stop),enable_thinking=thinking)
                cfg.validate()
                chat=self.path=='/v1/chat/completions'
                prompt=runner.render(body['messages'],thinking) if chat else body['prompt']
                if not isinstance(prompt,str): raise ValueError('Only one text prompt per request is supported')
                out=batcher.submit(prompt,cfg)
                choice={'index':0,'finish_reason':out['finish_reason']}
                choice.update({'message':{'role':'assistant','content':out['text']}} if chat else {'text':out['text']})
                self.send(200,{'id':'cmpl-'+uuid.uuid4().hex,'object':'chat.completion' if chat else 'text_completion',
                    'created':int(time.time()),'model':args.name,'choices':[choice],
                    'usage':{'prompt_tokens':out['prompt_tokens'],'completion_tokens':out['completion_tokens'],
                             'total_tokens':out['prompt_tokens']+out['completion_tokens']},
                    'rmt_metadata':{'prompt_sha256':out['prompt_sha256'],'seed':cfg.seed,'batch_split_retries':out['batch_split_retries'],'recurrence':out.get('recurrence')}})
            except (ValueError,KeyError,TypeError) as error: self.send(400,{'error':{'message':str(error)}})
            except Exception as error: self.send(500,{'error':{'message':type(error).__name__}})
    print(json.dumps({'ready':True,'port':args.port,'model':args.name}),flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__': main()
