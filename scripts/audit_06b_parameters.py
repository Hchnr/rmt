"""Count unique serialized weights without loading a model onto any GPU."""
from collections import defaultdict
import json
import math
from pathlib import Path
from safetensors import safe_open
import torch

rows=[]
for name, root in [('native',Path('/share/project/eai_pwm/models/Qwen/Qwen3-0.6B')),('rmt',Path('artifacts/v0.0.5/reference_hf'))]:
    groups=defaultdict(int)
    serialized_parameters=0
    duplicate_tied_parameters=0
    for path in root.glob('*.safetensors'):
        with safe_open(path,framework='pt',device='cpu') as file:
            for key in file.keys():
                count=math.prod(file.get_slice(key).get_shape())
                serialized_parameters+=count
                if key=='lm_head.weight' and 'model.embed_tokens.weight' in file.keys():
                    assert json.loads((root/'config.json').read_text())['tie_word_embeddings']
                    assert torch.equal(file.get_tensor(key),file.get_tensor('model.embed_tokens.weight'))
                    duplicate_tied_parameters+=count
                    continue
                if 'embed_tokens' in key or key=='lm_head.weight':group='tied_embedding_and_lm_head'
                elif '.router.' in key:group='router'
                elif '.mlp.' in key:group='ffn_projections'
                elif '.self_attn.' in key and '_proj.' in key:group='attention_projections'
                elif 'norm' in key:group='normalizations'
                else:raise ValueError(f'Unclassified weight {key}')
                groups[group]+=count
    rows.append({'name':name,'path':str(root),'groups':dict(groups),'unique_parameters':sum(groups.values()),
                 'serialized_parameters':serialized_parameters,'duplicate_tied_parameters':duplicate_tied_parameters})
assert rows[0]['unique_parameters']==596049920
assert rows[1]['unique_parameters']==596079600
assert rows[1]['groups']['router']==29680
Path('reports/v0.0.5/parameter_audit.json').write_text(json.dumps({'status':'passed','rows':rows,'scope':'Logical unique weights. Native file stores identical embedding/head twice; equality verified and shared weights counted once. Runtime tying separately verified by reference.json. Fixed layer-order training does not use router gradients.'},indent=2)+'\n')
print(rows)
