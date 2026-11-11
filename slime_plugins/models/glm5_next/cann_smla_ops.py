"""Load operators through the official installed CANN package."""
from functools import lru_cache


@lru_cache(maxsize=1)
def load_sparse_ops():
    names = ("sparse_flash_mla_metadata", "sparse_flash_mla",
             "sparse_flash_mla_grad_metadata", "sparse_flash_mla_grad")
    return _load_ops(names)


@lru_cache(maxsize=1)
def load_mhc_ops():
    names = ("mhc_pre_sinkhorn_backward", "mhc_post_backward", "mhc_pre_sinkhorn", "mhc_post")
    return _load_ops(names)


@lru_cache(maxsize=1)
def load_kpool_ops():
    names = ("key_pool", "pool_key_indexer")
    return _load_ops(names)


def _load_ops(names):
    # Let the official package discover installed custom operator entry points.
    # Reading modules directly from ASCEND_HOME_PATH would bypass wheel updates.
    import cann_ops_transformer

    # Python wrappers retain the official autograd and keyword interfaces.
    namespace = cann_ops_transformer.ops
    missing = [name for name in names if namespace is None or not hasattr(namespace, name)]
    if missing:
        raise RuntimeError(f"cann_ops_transformer is missing required APIs: {missing}")
    return namespace
