"""Capacity cache adapter for the pinned Transformers Qwen3 mask interface."""
from ..cache import RmtCapacityCache


class NativeCapacityCache(RmtCapacityCache):
    def get_mask_sizes(self, cache_position, layer_idx=0):
        # Unlike the inherited DynamicCache, storage lives in our per-layer
        # buffers rather than self.layers. Report the logical used prefix.
        return self.get_seq_length(layer_idx)+cache_position.shape[0], 0
