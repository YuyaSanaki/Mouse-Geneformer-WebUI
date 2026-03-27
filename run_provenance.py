from __future__ import annotations

import os
import shutil
import subprocess
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def write_service_provenance(
    run_root: Path,
    config_path: Path | None = None,
    extra_meta: dict[str, Any] | None = None,
    input_paths: dict[str, str] | None = None,
    fast_fingerprint: bool = True
) -> dict[str, Any]:
    """
    Standardized provenance recorder for any service.
    
    Args:
        run_root: Directory where metadata will be stored.
        config_path: Optional path to the YAML config used for this run.
        extra_meta: Optional dictionary of service-specific metadata.
        input_paths: Optional dict of {label: path} to fingerprint.
        fast_fingerprint: Whether to use fast metadata-based fingerprinting.
    """
    run_root.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy config if provided
    if config_path and config_path.is_file():
        shutil.copy2(config_path, run_root / f"{config_path.stem}_used{config_path.suffix}")

    # 2. Collect base metadata
    meta: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source_path": str(config_path.resolve()) if config_path else None,
    }
    
    if extra_meta:
        meta.update(extra_meta)

    # 3. Add Git SHA
    git_sha = os.environ.get("ISP_GIT_COMMIT", "").strip()
    if not git_sha:
        try:
            rev = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True, text=True, timeout=5, check=False
            )
            if rev.returncode == 0:
                git_sha = rev.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    if git_sha:
        meta["git_commit"] = git_sha

    # 4. Compute Fingerprints
    if input_paths and os.environ.get("ENABLE_INPUT_FINGERPRINT", "").lower() in ("1", "true", "yes"):
        from run_input_fingerprint import fingerprint_local_path
        print(f"Computing input fingerprints ({'fast' if fast_fingerprint else 'full'} mode)...")
        fps = {}
        for label, path_str in input_paths.items():
            if path_str:
                fp = fingerprint_local_path(Path(path_str), fast=fast_fingerprint)
                fps[label] = fp
                digest = fp.get("content_fingerprint_sha256")
                print(f"  {label}: {digest or 'error'}")
        meta["input_content_fingerprints"] = fps

    # 5. Save metadata
    with open(run_root / "run_metadata.yaml", "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    
    return meta

def update_service_provenance(run_root: Path, updates: dict[str, Any]) -> None:
    """Update existing run_metadata.yaml with end-of-run info."""
    path = run_root / "run_metadata.yaml"
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    meta.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
