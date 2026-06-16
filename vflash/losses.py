import torch
import torch.nn.functional as F


def fused_loss(draft_logits, target_ids, target_logits, weight,
               ce_weight=1.0, kl_weight=1.0, kl_temp=1.0, kl_topk=0):
    """Weighted CE (vs sampled/argmax target tokens) + KL distillation (vs target dist).

    draft_logits  [P,V]   drafter logits at draft positions
    target_ids    [P]     target's next-token ids (self-distillation labels)
    target_logits [P,V]   target distribution (None disables KL)
    weight        [P]     per-position weights (validity * decay * loss_mask)
    """
    denom = weight.sum().clamp_min(1e-6)

    ce = F.cross_entropy(draft_logits, target_ids, reduction="none")
    ce = (ce * weight).sum() / denom

    kl = draft_logits.new_zeros(())
    if target_logits is not None and kl_weight > 0:
        t_logp = F.log_softmax(target_logits / kl_temp, dim=-1)
        d_logp = F.log_softmax(draft_logits / kl_temp, dim=-1)
        if kl_topk and kl_topk < t_logp.shape[-1]:
            t_top, idx = t_logp.topk(kl_topk, dim=-1)            # [P,k]
            t_p = t_top.exp()
            t_p = t_p / t_p.sum(-1, keepdim=True)
            d_top = d_logp.gather(-1, idx)
            kl_pos = (t_p * (t_top - t_top.logsumexp(-1, keepdim=True) - d_top)).sum(-1)
        else:
            kl_pos = (t_logp.exp() * (t_logp - d_logp)).sum(-1)
        kl = (kl_pos * weight).sum() / denom * (kl_temp ** 2)

    total = ce_weight * ce + kl_weight * kl
    return total, ce.detach(), kl.detach()
