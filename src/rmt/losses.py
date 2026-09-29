"""Exact CE + teacher-to-student KL, checkpointed over vocabulary projections."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def shifted_targets(labels, attention_mask=None, segment_ids=None):
    targets=labels[:,1:].clone()
    if attention_mask is not None:
        targets.masked_fill_(~(attention_mask[:,1:].bool() & attention_mask[:,:-1].bool()),-100)
    if segment_ids is not None:
        targets.masked_fill_((segment_ids[:,1:]!=segment_ids[:,:-1]) | (segment_ids[:,1:]<0),-100)
    return targets


def distillation_loss(hidden, head_weight, labels, attention_mask=None, segment_ids=None,
                      teacher_hidden=None, teacher_weight=None, chunk_size=32,
                      temperature=1.0, ce_weight=1.0, kd_weight=1.0, recompute=True):
    if temperature<=0 or chunk_size<1:
        raise ValueError('Positive temperature and chunk_size required')
    if kd_weight and (teacher_hidden is None or teacher_weight is None):
        raise ValueError('KL requires frozen teacher hidden states and head')
    targets=shifted_targets(labels,attention_mask,segment_ids).reshape(-1)
    h=hidden[:,:-1].reshape(-1,hidden.shape[-1])
    th=None if teacher_hidden is None else teacher_hidden[:,:-1].detach().reshape(-1,teacher_hidden.shape[-1])
    denominator=(targets!=-100).sum().clamp_min(1)
    ce_total=hidden.sum()*0
    kd_total=hidden.sum()*0
    for start in range(0,h.shape[0],chunk_size):
        stop=start+chunk_size
        target=targets[start:stop]
        teacher_chunk=None if th is None else th[start:stop]
        def part(student_h, student_w, target=target, teacher_chunk=teacher_chunk):
            logits=F.linear(student_h,student_w).float()
            ce=F.cross_entropy(logits,target,ignore_index=-100,reduction='sum')
            if kd_weight:
                with torch.no_grad():
                    tl=F.linear(teacher_chunk,teacher_weight.detach()).float()/temperature
                    log_t=F.log_softmax(tl,dim=-1)
                log_s=F.log_softmax(logits/temperature,dim=-1)
                per_token=F.kl_div(log_s,log_t,log_target=True,reduction='none').sum(-1)
                kd=(per_token*(target!=-100)).sum()*temperature**2
            else:
                kd=logits.sum()*0
            return ce,kd
        if recompute and torch.is_grad_enabled():
            ce,kd=checkpoint(part,h[start:stop],head_weight,use_reentrant=False)
        else:
            ce,kd=part(h[start:stop],head_weight)
        ce_total=ce_total+ce
        kd_total=kd_total+kd
    ce_total=ce_total/denominator
    kd_total=kd_total/denominator
    return ce_weight*ce_total+kd_weight*kd_total,ce_total,kd_total
