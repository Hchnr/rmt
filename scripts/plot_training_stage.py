"""Export reviewable diagnostic curves for the frozen deterministic controls."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('reports/v0.0.4')
fig,axes=plt.subplots(1,3,figsize=(13,3.6))
for name,label in [('fixed_128','Fixed layer order'),('opened_128','Learned, prior 4 to 3')]:
 d=json.loads((root/(name+'.json')).read_text());steps=d['steps'];checks=d['validation']
 axes[0].plot([x['step'] for x in checks],[x['ce'] for x in checks],marker='o',label=label)
 axes[1].plot([x['step'] for x in steps],[x['off_layer_fraction']*100 for x in steps],label=label)
 axes[2].plot([x['step'] for x in steps[1:]],[x['seconds'] for x in steps[1:]],label=label,alpha=.8)
axes[0].set(title='Development pilot CE',ylabel='Assistant-target CE')
axes[1].set(title='Actual routing changes',ylabel='Off-layer decisions (%)')
axes[2].set(title='Training step time',ylabel='Seconds (first step excluded)')
for ax in axes:ax.set_xlabel('Optimizer step');ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Deterministic 4 x H100, same token budget; not a benchmark quality claim',fontsize=11)
fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=170);fig.savefig(root/'training_curves.svg');plt.close(fig)

svg=root/'training_curves.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
