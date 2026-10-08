"""Freeze comparable CE/KL pilot configs; target budget counts actual labels."""
from pathlib import Path
import yaml

root=Path('configs/v0.0.5');root.mkdir(parents=True,exist_ok=True)
base=dict(base_model='/share/project/eai_pwm/models/Qwen/Qwen3-0.6B',
    train_data='artifacts/v0.0.5/corpus/teacher32_train.jsonl',
    dev_data='artifacts/v0.0.5/corpus/dev.jsonl',seed=17,steps=10000,
    target_budget=1000000,attention='sdpa',checkpoint=True,compile=True,
    loss_chunk_size=64,learning_rate=1e-5,router_learning_rate=1e-4,
    lr_warmup_steps=8,validation_batches=32,eval_every=200,save_every=200,
    save=True,export=True,verify_replay=True,sequence_length=2048,
    routing_mode='layer_order',prior_start=4.0,
    model_overrides=dict(num_recurrences=28,max_recurrences=36,min_recurrences=28,
                        halting_policy='fixed',recurrence_schedule='tail',tail_experts=8))
for name,kd in [('ce',0.0),('kl01',0.1),('kl05',0.5)]:
    cfg=dict(base,kd_weight=kd,artifact_dir=f'artifacts/v0.0.5/pilot_{name}',
             report=f'reports/v0.0.5/pilot_{name}.json')
    (root/f'pilot_{name}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    smoke=dict(cfg,steps=8,target_budget=1000000,validation_batches=2,
               eval_every=0,save_every=0,verify_replay=False,
               artifact_dir=f'artifacts/v0.0.5/smoke_{name}',
               report=f'reports/v0.0.5/smoke_{name}.json')
    (root/f'smoke_{name}.yaml').write_text(yaml.safe_dump(smoke,sort_keys=False))
