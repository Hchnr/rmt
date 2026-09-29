"""HF DynamicCache indexed by recurrence, never by selected expert."""
from transformers.cache_utils import DynamicCache


class RmtCache(DynamicCache):
    layout_version = 1


class RmtCapacityCache(DynamicCache):
    """Preallocated per-recurrence KV storage with append-only logical lengths.

    Unlike HF StaticCache, returned tensors expose the used prefix only. This
    keeps attention/mask semantics identical to DynamicCache while avoiding cat.
    Projection graphs do not include cache mutation or changing attention length.
    """
    def __init__(self, capacity):
        super().__init__()
        if capacity < 1:
            raise ValueError('Positive capacity required')
        self.capacity = capacity
        self.storage = {}
        self.lengths = {}

    def get_seq_length(self, layer_idx=0):
        return self.lengths.get(layer_idx, 0)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.requires_grad or value_states.requires_grad:
            raise ValueError('Capacity cache is inference-only')
        start = self.get_seq_length(layer_idx)
        end = start + key_states.shape[-2]
        if end > self.capacity:
            raise ValueError('KV cache capacity exceeded')
        if layer_idx not in self.storage:
            shape = (*key_states.shape[:-2], self.capacity, key_states.shape[-1])
            self.storage[layer_idx] = (key_states.new_empty(shape), value_states.new_empty(shape))
        keys, values = self.storage[layer_idx]
        if key_states.shape != value_states.shape or key_states.shape[:-2] != keys.shape[:-2] or key_states.shape[-1] != keys.shape[-1]:
            raise ValueError('Cache batch/head shape changed; reset requires the same storage layout')
        if key_states.dtype != keys.dtype or key_states.device != keys.device:
            raise ValueError('Cache dtype/device changed')
        keys[..., start:end, :].copy_(key_states)
        values[..., start:end, :].copy_(value_states)
        self.lengths[layer_idx] = end
        return keys[..., :end, :], values[..., :end, :]

    def batch_select_indices(self, indices):
        self.storage = {i:(k.index_select(0,indices),v.index_select(0,indices))
                        for i,(k,v) in self.storage.items()}

    def reorder_cache(self, beam_idx):
        self.batch_select_indices(beam_idx)

    def reset(self):
        self.lengths.clear()

    def get_max_cache_shape(self, layer_idx=0):
        return self.capacity
