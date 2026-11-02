"""Load selected installed CANN modules without eager graph imports."""
import builtins
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace


@lru_cache(maxsize=1)
def load_sparse_ops():
    names = ("sparse_flash_mla_metadata", "sparse_flash_mla",
             "sparse_flash_mla_grad_metadata", "sparse_flash_mla_grad")
    return _load_ops("attention", ("sparse_flash_mla", "sparse_flash_mla_grad"), names)


@lru_cache(maxsize=1)
def load_mhc_ops():
    names = ("mhc_pre_sinkhorn_backward", "mhc_post_backward", "mhc_pre_sinkhorn", "mhc_post")
    return _load_ops("mhc", names, names)


@lru_cache(maxsize=1)
def load_kpool_ops():
    names = ("key_pool", "pool_key_indexer")
    return _load_ops("attention", names, names)


def _load_ops(category, op_names, names):
    # Respect an SDK already initialized by the calling application.
    existing = sys.modules.get("cann_ops_transformer")
    if existing is not None and all(hasattr(existing, n) for n in names):
        return existing
    existing_ops = getattr(existing, "ops", None)
    if existing_ops is not None and all(hasattr(existing_ops, n) for n in names):
        return existing_ops
    home = os.environ.get("ASCEND_HOME_PATH")
    if not home:
        raise RuntimeError("Source the native activate.sh before using CANN operators")
    root = Path(home) / "python/site-packages/cann_ops_transformer"
    def sdk_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cann_ops_transformer.op_builder" and level == 0:
            return builder
        if name == "cann_ops_transformer.utils.arg_check" and level == 0:
            return load_module("_glm53_cann_smla_arg_check", root / "utils/arg_check.py")
        if level == 2 and name in op_names:
            return load_module("_glm53_cann_" + name, root / "ops" / category / name / (name + ".py"))
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load_module(name, path):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # This import mapping belongs only to this module's globals. It does
        # not replace builtins.__import__ or the SDK package globally.
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=sdk_import)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    builder = load_module("_glm53_cann_smla_op_builder", root / "op_builder/builder.py")
    namespace = {}
    for op in op_names:
        module = load_module("_glm53_cann_" + op, root / "ops" / category / op / (op + ".py"))
        for symbol in names:
            if hasattr(module, symbol):
                namespace[symbol] = getattr(module, symbol)
    return SimpleNamespace(**namespace)
