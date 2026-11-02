import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable

import torch

from slime.utils import accelerator

logger = logging.getLogger(__name__)

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


def _tb_log_enabled() -> bool:
    return os.environ.get("GLM53_TB_LOG", "0") == "1"


class TensorBackuper:
    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        t0 = time.perf_counter()
        backup_dict = self._backups[tag]
        with torch.profiler.record_function("glm53::tensor_backup"):
            for name, param in self._source_getter():
                if name not in backup_dict:
                    backup_dict[name] = torch.empty_like(
                        param, device=torch.device("cpu"), pin_memory=False
                    )
                backup_dict[name].copy_(param.detach(), non_blocking=False)
        accelerator.synchronize()
        if _tb_log_enabled():
            tensors = len(backup_dict)
            nbytes = sum(t.numel() * t.element_size() for t in backup_dict.values())
            logger.info(
                "TB backup tag=%s tensors=%d bytes=%d elapsed=%.3fs",
                tag, tensors, nbytes, time.perf_counter() - t0,
            )

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        t0 = time.perf_counter()
        backup_dict = self._backups[tag]
        with torch.profiler.record_function("glm53::tensor_restore"):
            for name, param in self._source_getter():
                assert name in backup_dict
                param.copy_(backup_dict[name], non_blocking=False)
        accelerator.synchronize()
        if _tb_log_enabled():
            logger.info(
                "TB restore tag=%s tensors=%d elapsed=%.3fs",
                tag, len(backup_dict), time.perf_counter() - t0,
            )
