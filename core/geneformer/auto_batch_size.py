"""Pick a batch size by measuring the real model on the current GPU.

Two ceilings matter: GPU memory, and the batch beyond which throughput stops
improving (on unified-memory boxes such as GB10 the second one usually binds
first, so sizing from free memory alone picks a needlessly large batch). The
probe below measures both, then keeps the smallest batch whose throughput is
within `min_gain` of the best it saw.

Results are cached per (GPU, model, sequence length, dtype) because probing
costs about a minute; set GENEFORMER_BATCH_SIZE_CACHE=off to always re-probe.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)
DEFAULT_MEMORY_FRACTION = 0.8
DEFAULT_MIN_GAIN = 0.05


def is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def coerce_batch_size(value: Any, *, default: int) -> int | str:
    """Pass "auto" through untouched; otherwise parse an int."""
    if is_auto(value):
        return "auto"
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid batch size %r; using %d.", value, default)
        return default


def _is_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _cache_path() -> Path | None:
    raw = os.environ.get("GENEFORMER_BATCH_SIZE_CACHE", "").strip()
    if raw.lower() in ("off", "0", "false", "no"):
        return None
    if raw:
        return Path(raw)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "mouse-geneformer-webui" / "batch_size.json"


def _cache_key(fields: Mapping[str, Any]) -> str:
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _read_cache(path: Path | None, key: str) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable batch-size cache %s: %s", path, e)
        return None
    if isinstance(entry, Mapping) and isinstance(entry.get("batch_size"), int):
        return int(entry["batch_size"])
    return None


def _write_cache(path: Path | None, key: str, batch_size: int, fields: Mapping[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                data = {}
        data[key] = {"batch_size": batch_size, "fields": dict(fields)}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("Could not write batch-size cache %s: %s", path, e)


def calibrate_batch_size(
    probe: Callable[[int], None],
    *,
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    min_gain: float = DEFAULT_MIN_GAIN,
    repeats: int = 2,
    label: str = "batch size",
) -> int:
    """Run `probe(batch)` over increasing batches; return the best usable one."""
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    free, total = torch.cuda.mem_get_info()
    # Budget off free memory, not device total: other jobs may share this GPU.
    budget = baseline + int(memory_fraction * free)
    chosen = int(candidates[0])
    best_rate = 0.0

    print(
        f"Calibrating {label} (budget {budget / 2**30:.1f} GiB, "
        f"{free / 2**30:.1f} GiB free)...",
        flush=True,
    )
    if free < 0.6 * (total - baseline):
        print(
            "  Note: another job is using this GPU, so throughput measurements are "
            "pessimistic and the batch size will be conservative.",
            flush=True,
        )
    for candidate in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            probe(candidate)  # warm up kernels / allocator before timing
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(repeats):
                probe(candidate)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        except Exception as e:  # noqa: BLE001 - OOM is the expected stop condition
            if not _is_oom(e):
                raise
            print(f"  batch {candidate}: out of memory, stopping.", flush=True)
            torch.cuda.empty_cache()
            break

        peak = torch.cuda.max_memory_allocated()
        rate = (candidate * repeats) / max(elapsed, 1e-9)
        print(
            f"  batch {candidate}: {rate:,.0f} samples/s, peak {peak / 2**30:.1f} GiB",
            flush=True,
        )

        if peak > budget:
            print(f"  batch {candidate}: over memory budget, keeping {chosen}.", flush=True)
            break
        if best_rate and (rate - best_rate) / best_rate < min_gain:
            print(f"  batch {candidate}: no meaningful speedup, keeping {chosen}.", flush=True)
            break

        chosen, best_rate = int(candidate), rate
        if peak * 2 > budget:
            print(f"  doubling again would exceed the budget, keeping {chosen}.", flush=True)
            break

    torch.cuda.empty_cache()
    print(f"Selected {label}: {chosen}", flush=True)
    return chosen


def resolve_batch_size(
    value: Any,
    *,
    default: int,
    probe: Callable[[int], None],
    cache_fields: Mapping[str, Any],
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    min_gain: float = DEFAULT_MIN_GAIN,
    label: str = "batch size",
) -> int:
    """Return `value` as an int, calibrating on this GPU when it is "auto"."""
    if not is_auto(value):
        return int(value)
    if not torch.cuda.is_available():
        logger.warning("auto %s requires CUDA; using %d.", label, default)
        return default

    fields = {
        **dict(cache_fields),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "memory_fraction": memory_fraction,
        "candidates": list(candidates),
    }
    path = _cache_path()
    key = _cache_key(fields)
    cached = _read_cache(path, key)
    if cached is not None:
        print(f"Using cached {label}: {cached}  (cache: {path})", flush=True)
        return cached

    chosen = calibrate_batch_size(
        probe,
        candidates=candidates,
        memory_fraction=memory_fraction,
        min_gain=min_gain,
        label=label,
    )
    _write_cache(path, key, chosen, fields)
    return chosen
