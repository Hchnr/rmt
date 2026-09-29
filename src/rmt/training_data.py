"""Immutable token records and deterministic assistant-target packed batches."""
import hashlib
import json
from pathlib import Path
import random
import torch
from .data import pack_sequences


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenize_response(tokenizer, messages):
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('Expected a final assistant response')
    prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not text.startswith(prefix):
        raise ValueError('Final assistant is not a non-thinking continuation')
    ids=tokenizer.encode(text,add_special_tokens=False)
    pre=tokenizer.encode(prefix,add_special_tokens=False)
    if ids[:len(pre)]!=pre:raise ValueError('Token boundary crosses response prefix')
    labels=[-100]*len(pre)+ids[len(pre):]
    return ids,labels


def load_packs(path,length,pad_id,seed=17,shuffle=True):
    records=[json.loads(x) for x in Path(path).read_text().split('\n') if x.strip()]
    if shuffle:random.Random(seed).shuffle(records)
    chunks=[]
    for row in records:
        if len(row['input_ids'])!=len(row['labels']):raise ValueError('Token/label length mismatch')
        for start in range(0,len(row['input_ids']),length):
            ids=row['input_ids'][start:start+length];labels=row['labels'][start:start+length]
            labels=labels.copy();labels[0]=-100
            if any(x!=-100 for x in labels[1:]):chunks.append((ids,labels))
    packs=[];current=[];size=0
    def flush():
        batch=pack_sequences([x[0] for x in current],pad_id,length)
        labels=[token for _,ys in current for token in ys]
        batch['labels']=torch.tensor([labels+[-100]*(length-len(labels))],dtype=torch.long)
        packs.append(batch)
    for item in chunks:
        if size+len(item[0])>length:flush();current=[];size=0
        current.append(item);size+=len(item[0])
    if current:flush()
    if not packs:raise ValueError('No supervised packed batches')
    return packs


def batch_at(packs,step,rank,world,device):
    return {k:v.to(device) for k,v in packs[(step*world+rank)%len(packs)].items()}
