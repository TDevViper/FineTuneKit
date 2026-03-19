import mlx.core as mx
from mlx_lm.tuner.datasets import TextDataset, CacheDataset

def make_dataset(texts, tokenizer, max_length=512):
    """
    CacheDataset wraps a TextDataset.
    TextDataset.process() tokenizes each item.
    CacheDataset caches the processed (tokenized) results.
    """
    raw = [{"text": t} for t in texts]
    inner = TextDataset(raw, tokenizer, text_key="text")
    return CacheDataset(inner)
