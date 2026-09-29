"""Small deterministic fixtures and independent-document packing."""
import torch

FIXTURES = [
    "Explain why the sky is blue in one short sentence.",
    "计算 17 加 25，并简要解释计算过程。",
    "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5",
    "A train travels 60 km in one hour. How far does it travel in two hours?",
]


def pack_sequences(sequences, pad_token_id=0, length=None, device="cpu"):
    """Pack one row; reset positions and prevent cross-document target shifts."""
    total = sum(len(s) for s in sequences)
    length = total if length is None else length
    if total > length or any(len(s) == 0 for s in sequences):
        raise ValueError("Packing requires nonempty documents fitting within length")
    ids, segments, positions, labels = [], [], [], []
    for segment, seq in enumerate(sequences):
        seq = list(seq)
        ids.extend(seq)
        segments.extend([segment] * len(seq))
        positions.extend(range(len(seq)))
        labels.extend([-100] + seq[1:])
    padding = length - total
    return {
        "input_ids": torch.tensor([ids + [pad_token_id] * padding], dtype=torch.long, device=device),
        "segment_ids": torch.tensor([segments + [-1] * padding], dtype=torch.long, device=device),
        "position_ids": torch.tensor([positions + [0] * padding], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[1] * total + [0] * padding], dtype=torch.long, device=device),
        "labels": torch.tensor([labels + [-100] * padding], dtype=torch.long, device=device),
    }
