"""HF-compatible cached batch runner, shared by service and offline evaluation."""
import hashlib
import math
from pathlib import Path
import time
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..configuration_rmt import RmtConfig
from ..modeling_rmt import RmtForCausalLM
from ..cache import RmtCapacityCache
from ..runtime import enforce_gpu_scope,versions


@dataclass(frozen=True)
class Generation:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.5
    seed: int = 17
    stop: tuple = ()
    enable_thinking: bool = False

    def validate(self):
        if isinstance(self.max_new_tokens,bool) or not isinstance(self.max_new_tokens,int) or not 1 <= self.max_new_tokens <= 38912:
            raise ValueError('max_new_tokens must be in [1,38912]')
        if not isinstance(self.top_k,int) or isinstance(self.top_k,bool) or not all(math.isfinite(x) for x in (self.temperature,self.top_p,self.presence_penalty)) or self.temperature < 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError('Invalid sampling configuration')
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError('presence_penalty must be in [-2,2]')
        if any(not isinstance(s,str) or not s for s in self.stop):
            raise ValueError('stop strings must be nonempty strings')


def sample(logits, generated, cfg, generator):
    scores = logits.float().clone()
    if generated and cfg.presence_penalty:
        scores[torch.tensor(list(set(generated)),device=scores.device)] -= cfg.presence_penalty
    if cfg.temperature == 0:
        return scores.argmax().item()
    scores /= cfg.temperature
    if cfg.top_k:
        threshold = scores.topk(min(cfg.top_k,scores.numel())).values[-1]
        scores.masked_fill_(scores < threshold, -float('inf'))
    sorted_scores, order = scores.sort(descending=True)
    probabilities = sorted_scores.softmax(-1)
    remove = probabilities.cumsum(-1)-probabilities >= cfg.top_p
    sorted_scores.masked_fill_(remove, -float('inf'))
    return order[torch.multinomial(sorted_scores.softmax(-1),1,generator=generator)].item()


class Runner:
    def __init__(self, path, backend='rmt', compiled=True, device='cuda:0',
                 attention='eager', max_context=4096, routing=None):
        enforce_gpu_scope()
        torch.set_num_threads(4)
        self.path,self.backend,self.compiled = str(path),backend,compiled
        self.source_hashes={str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parents[1].rglob('*.py') if 'evaluation' not in p.parts}
        self.device=torch.device(device)
        self.engine_environment=versions()
        self.engine_environment.pop('cuda_visible_devices',None)
        self.engine_environment.update(gpu_name=torch.cuda.get_device_name(self.device),
            compute_capability=list(torch.cuda.get_device_capability(self.device)))
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.tokenizer.padding_side='left'
        if backend=='rmt':
            config=RmtConfig.from_pretrained(path,local_files_only=True)
            config._attn_implementation=attention
            if routing: config.routing_mode=routing
            self.model=RmtForCausalLM.from_pretrained(path,config=config,local_files_only=True,
                dtype=torch.bfloat16,attn_implementation=attention).to(device).eval()
            if compiled: self.model.model.cell.bank.compile_projections()
        elif backend=='qwen':
            self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
                dtype=torch.bfloat16,attn_implementation=attention).to(device).eval()
            if compiled: raise ValueError('Qwen reference runner uses eager; use vLLM for optimized Qwen')
        else: raise ValueError('Unknown backend')
        self.max_context=min(max_context,self.model.config.max_position_embeddings)
        eos=self.model.generation_config.eos_token_id
        self.eos=set(eos if isinstance(eos,list) else [eos])
        self.pad=self.tokenizer.pad_token_id

    def render(self, messages, thinking=False):
        return self.tokenizer.apply_chat_template(messages,tokenize=False,
            add_generation_prompt=True,enable_thinking=thinking)

    @torch.inference_mode()
    def generate(self, prompts, configs=None, use_cache=True):
        if not prompts: return []
        configs=configs or [Generation() for _ in prompts]
        if len(configs)!=len(prompts): raise ValueError('One configuration per prompt required')
        for cfg in configs: cfg.validate()
        encoded=[self.tokenizer.encode(p,add_special_tokens=False) for p in prompts]
        if any(not ids for ids in encoded): raise ValueError('Empty prompt is unsupported')
        width=max(map(len,encoded)); budget=max(c.max_new_tokens for c in configs)
        if width+budget>self.max_context: raise ValueError('Prompt plus output exceeds configured context')
        ids=torch.tensor([[self.pad]*(width-len(x))+x for x in encoded],device=self.device)
        mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in encoded],device=self.device)
        cache=RmtCapacityCache(width+budget) if use_cache and self.backend=='rmt' else None
        histories=[[] for _ in prompts]; done=[False]*len(prompts); texts=['']*len(prompts)
        reasons=['length']*len(prompts)
        generators=[torch.Generator(device=self.device).manual_seed(c.seed) for c in configs]
        depth_stats=[{'prefill_depth_sum':0,'prefill_positions':0,'decode_depth_sum':0,'decode_positions':0,'cap_positions':0} for _ in prompts]
        start=time.perf_counter(); first=None; token_times=[]
        for step in range(budget):
            positions=(mask.cumsum(-1)-1).clamp_min(0)
            current=ids[:,-1:] if use_cache and step else ids
            output=self.model(input_ids=current,attention_mask=mask,position_ids=positions[:,-current.shape[1]:],
                past_key_values=cache,use_cache=use_cache,logits_to_keep=1)
            if getattr(output,'exit_depths',None) is not None:
                depths=output.exit_depths
                measured=depths if step==0 else depths[:,-1:]
                exits=output.exit_reasons if step==0 else output.exit_reasons[:,-1:]
                phase='prefill' if step==0 else 'decode'
                statistics=torch.stack((measured.sum(-1),(measured>0).sum(-1),(exits==2).sum(-1)),-1).tolist()
                for row,(depth_sum,positions,cap_count) in enumerate(statistics):
                    if not done[row]:
                        depth_stats[row][phase+'_depth_sum']+=depth_sum
                        depth_stats[row][phase+'_positions']+=positions
                        depth_stats[row]['cap_positions']+=cap_count
            if use_cache: cache=output.past_key_values
            greedy_batch = output.logits[:,-1].argmax(-1).tolist() if all(
                c.temperature == 0 and c.presence_penalty == 0 for c in configs) else None
            active=[not x for x in done]
            next_ids=[]
            for row,cfg in enumerate(configs):
                if done[row]: next_ids.append(self.pad); continue
                token=greedy_batch[row] if greedy_batch is not None else sample(output.logits[row,-1],histories[row],cfg,generators[row])
                histories[row].append(token);next_ids.append(token)
                texts[row]=self.tokenizer.decode(histories[row],skip_special_tokens=True)
                stops=[texts[row].find(s) for s in cfg.stop if s in texts[row]]
                if token in self.eos or stops:
                    done[row]=True;reasons[row]='stop'
                    if stops: texts[row]=texts[row][:min(stops)]
                elif len(histories[row])>=cfg.max_new_tokens: done[row]=True
            torch.cuda.synchronize(self.device)
            now=time.perf_counter()
            if first is None: first=now-start
            token_times.append(now-start)
            if all(done): break
            ids=torch.cat((ids,torch.tensor(next_ids,device=self.device)[:,None]),1)
            mask=torch.cat((mask,torch.tensor(active,device=self.device,dtype=mask.dtype)[:,None]),1)
        elapsed=time.perf_counter()-start
        return [{'text':texts[i],'token_ids':histories[i],'finish_reason':reasons[i],
                 'prompt_tokens':len(encoded[i]),'completion_tokens':len(histories[i]),
                 'prompt_sha256':hashlib.sha256(prompts[i].encode()).hexdigest(),
                 'batch_seconds':elapsed,'ttft_seconds':first,'step_end_seconds':token_times,'recurrence':depth_stats[i]}
                for i in range(len(prompts))]
