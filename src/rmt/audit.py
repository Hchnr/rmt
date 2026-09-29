"""Read-only asset audit and small CUDA/compile/NCCL probes."""
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from .runtime import enforce_gpu_scope, write_report, versions


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assets(path, full_hash=False):
    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensors, shards = {}, []
    for filename in sorted(set(index['weight_map'].values())):
        file = root / filename
        with file.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(size))
        header.pop('__metadata__', None)
        assert max(v['data_offsets'][1] for v in header.values()) + 8 + size == file.stat().st_size
        for key, value in header.items():
            assert index['weight_map'][key] == filename and key not in tensors
            tensors[key] = value
        entry = {'filename': filename, 'bytes': file.stat().st_size, 'tensor_count': len(header)}
        if full_hash:
            entry['sha256'] = sha256(file)
        shards.append(entry)
    assert tensors.keys() == index['weight_map'].keys()
    return {'source': str(root.resolve()), 'tensor_count': len(tensors),
            'stored_parameters': sum(math.prod(v['shape']) for v in tensors.values()),
            'dtype_counts': {d:sum(v['dtype']==d for v in tensors.values()) for d in {v['dtype'] for v in tensors.values()}},
            'shards':shards, 'full_weight_hashes':full_hash,
            'metadata_sha256':{f:sha256(root/f) for f in ['config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors.index.json']}}


def probe_cuda(distributed=False):
    enforce_gpu_scope()
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    local_rank = int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda',local_rank)
    result={'status':'passed', **versions(), 'device':torch.cuda.get_device_name(device)}
    x=torch.randn(16,32,device=device,dtype=torch.bfloat16,requires_grad=True)
    (x@x.T).float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    q=torch.randn(1,2,16,8,device=device,dtype=torch.bfloat16,requires_grad=True)
    F.scaled_dot_product_attention(q,q,q,is_causal=True).float().square().mean().backward()
    assert torch.isfinite(q.grad).all()
    compiled=torch.compile(lambda t: (t.sin()*t).sum(),fullgraph=True)
    torch.testing.assert_close(compiled(x.float()),(x.float().sin()*x.float()).sum())
    result['compile_backend']='inductor'
    if distributed:
        dist.init_process_group('nccl')
        value=torch.tensor(float(dist.get_rank()+1),device=device)
        dist.all_reduce(value)
        expected=dist.get_world_size()*(dist.get_world_size()+1)/2
        assert value.item()==expected
        result['nccl_world_size']=dist.get_world_size()
        dist.destroy_process_group()
    result['peak_memory_bytes']=torch.cuda.max_memory_allocated(device)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-model')
    parser.add_argument('--full-hash',action='store_true')
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--distributed',action='store_true')
    parser.add_argument('--output',default='reports/bootstrap/audit.json')
    args=parser.parse_args()
    enforce_gpu_scope()
    result={'status':'passed','environment':versions()}
    if args.base_model: result['assets']=assets(args.base_model,args.full_hash)
    if args.cuda or args.distributed: result['gpu_probe']=probe_cuda(args.distributed)
    if int(os.environ.get('RANK',0))==0:
        write_report(args.output,result)
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
