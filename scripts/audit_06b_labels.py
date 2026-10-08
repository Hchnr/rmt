"""Audit final-assistant supervision on every retokenized source record."""
import json
from pathlib import Path
from transformers import AutoTokenizer

base = '/share/project/eai_pwm/models/Qwen/Qwen3-0.6B'
tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
reports = []
for path in sorted(Path('artifacts/v0.0.5/corpus').glob('*.jsonl')):
    if path.name not in {'train.jsonl', 'teacher32_train.jsonl', 'mixed_train.jsonl', 'dev.jsonl', 'halt_calibration.jsonl', 'test.jsonl'}:
        continue
    examples = {}
    count = targets = 0
    for line in path.open():
        row = json.loads(line)
        ids, labels = row['input_ids'], row['labels']
        prefix = tok.apply_chat_template(row['messages'][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        assert ids[:len(prefix_ids)] == prefix_ids
        assert labels[:len(prefix_ids)] == [-100] * len(prefix_ids)
        assert labels[len(prefix_ids):] == ids[len(prefix_ids):]
        assert tok.convert_tokens_to_ids('<|im_end|>') in labels[len(prefix_ids):]
        assert len(ids) <= 2048 and len(ids) == len(labels)
        count += 1
        targets += sum(x != -100 for x in labels[1:])
        if row['domain'] not in examples:
            examples[row['domain']] = {'id': row['id'], 'masked_prefix_tokens': len(prefix_ids),
                'supervised_tokens': len(ids) - len(prefix_ids),
                'boundary': [{'position': i, 'id': ids[i], 'label': labels[i], 'token': tok.convert_ids_to_tokens(ids[i])}
                             for i in range(max(0, len(prefix_ids)-4), min(len(ids), len(prefix_ids)+4))],
                'last_tokens': [{'id': i, 'token': tok.convert_ids_to_tokens(i)} for i in ids[-4:]]}
    reports.append({'file': str(path), 'records': count, 'shifted_targets': targets, 'domain_boundary_examples': examples})
Path('reports/v0.0.5/label_audit.json').write_text(json.dumps({'status': 'passed', 'rows': reports,
    'policy': 'Only final assistant continuation is supervised, including the non-thinking template continuation and EOS. Prior turns and final assistant prefix are masked. Complete docs fit2048; pack boundary/gradient isolation separately covered by unit tests.'}, indent=2, ensure_ascii=False)+'\n')
print([(r['file'], r['records'], r['shifted_targets']) for r in reports])
