"""HF DynamicCache indexed by recurrence, never by selected expert."""
from transformers.cache_utils import DynamicCache


class RmtCache(DynamicCache):
    layout_version = 1
