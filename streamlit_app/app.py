"""
Mouse Geneformer — Streamlit control panel (same image as CLI; jobs run via subprocess).

Layout: data upload | run type | YAML configuration | execute + live log + outputs.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import yaml

def _repo_root() -> Path:
    """Repo root (contains streamlit_upload.py, data_input_layout.py)."""
    env = os.environ.get("WEBUI_ROOT")
    if env:
        return Path(env).resolve()
    # streamlit run .../streamlit_app/app.py often sets cwd to streamlit_app/, not /app
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "streamlit_upload.py").is_file():
        return candidate
    return Path(os.getcwd()).resolve()


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_data_input_layout():
    """Load data_input_layout from repo root (avoids stale sys.modules / wrong path)."""
    import importlib.util

    path = ROOT / "data_input_layout.py"
    if not path.is_file():
        raise ImportError(f"Missing {path}")
    spec = importlib.util.spec_from_file_location("data_input_layout", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["data_input_layout"] = mod
    spec.loader.exec_module(mod)
    required = (
        "count_single_cell_samples",
        "diagnose_input_dir",
        "resolve_single_cell_input_dir",
    )
    missing = [n for n in required if not hasattr(mod, n)]
    if missing:
        raise ImportError(
            f"{path} is outdated (missing {', '.join(missing)}). "
            "Restart the webui container after updating the repo."
        )
    return mod


_dil = _import_data_input_layout()
count_single_cell_samples = _dil.count_single_cell_samples
diagnose_input_dir = _dil.diagnose_input_dir
resolve_single_cell_input_dir = _dil.resolve_single_cell_input_dir
unique_states_from_samples = _dil.unique_states_from_samples

from streamlit_upload import (
    import_study_zip,
    normalize_study_name,
    resolve_study_tokenize_dir,
    study_folder,
    summarize_existing_study,
)

discover_sample_dirs = _dil.discover_sample_dirs
WORKSPACE = Path(os.environ.get("WEBUI_WORKSPACE", ROOT / "data" / "streamlit_workspace")).resolve()

RUN_FILES = {
    "Pipeline (E2E)": "pipeline.yaml",
    "ISP UMAP": "isp_umap.yaml",
}


def _default_config_path(run_label: str) -> Path:
    return ROOT / "config" / RUN_FILES[run_label]


def _load_default_yaml() -> None:
    name = st.session_state.get("run_type_sel", "Pipeline (E2E)")
    path = _default_config_path(name)
    if path.is_file():
        st.session_state["yaml_editor"] = path.read_text(encoding="utf-8")
    else:
        st.session_state["yaml_editor"] = f"# Missing file: {path}\n"
    if name == "Pipeline (E2E)":
        _sync_pipeline_form_from_yaml(_session_upload_dir())


def _ensure_workspace() -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return WORKSPACE


def _session_upload_dir() -> Path:
    sid = st.session_state.setdefault("upload_session_id", uuid.uuid4().hex[:10])
    d = _ensure_workspace() / "uploads" / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tail_log(path: Path, max_bytes: int = 64_000) -> str:
    if not path.is_file():
        return "(Waiting for log file…)"
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = b"... (showing tail)\n" + data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def _build_command_and_env(run_label: str, config_path: Path) -> tuple[list[str], dict[str, str]]:
    cfg = str(config_path)
    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "true")

    if run_label == "ISP UMAP":
        cmd = ["python3", str(ROOT / "run_isp_umap.py"), "--config", cfg]
        env["ISP_UMAP_CONFIG"] = cfg
    elif run_label == "Pipeline (E2E)":
        cmd = ["python3", str(ROOT / "run_pipeline.py"), "--config", cfg]
        env["PIPELINE_CONFIG"] = cfg
        env.setdefault("ISP_NUM_GPUS", os.environ.get("ISP_NUM_GPUS", "1"))
    else:
        raise ValueError(run_label)
    return cmd, env


def _guess_output_roots(run_label: str, cfg: dict) -> list[Path]:
    roots: list[Path] = []
    if run_label == "ISP UMAP":
        roots.append(ROOT / "output")
    elif run_label == "Pipeline (E2E)":
        paths = cfg.get("paths") or {}
        out_root = paths.get("output_root")
        if out_root:
            roots.append(Path(str(out_root)))
    return roots


def _build_pipeline_run_zip(run_dir: Path) -> tuple[bytes, str] | None:
    """Zip one pipeline_* run folder (checkpoints, figures, logs, configs, etc.)."""
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or not run_dir.name.startswith("pipeline_"):
        return None

    entries: list[tuple[Path, str]] = []
    for fp in sorted(run_dir.rglob("*")):
        if fp.is_file():
            rel = fp.relative_to(run_dir)
            entries.append((fp, f"{run_dir.name}/{rel.as_posix()}"))

    if not entries:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp, arcname in entries:
            zf.write(fp, arcname=arcname)
    return buf.getvalue(), f"{run_dir.name}.zip"


def _pipeline_run_summary(run_dir: Path) -> list[str]:
    """Short list of top-level artifacts in a pipeline run folder."""
    lines: list[str] = []
    for name in (
        "figures",
        "finetune",
        "isp_results",
        "ispstats_results",
        "tokenized_dataset",
        "stage_configs",
        "pipeline_run.log",
        "isp_run.log",
    ):
        path = run_dir / name
        if path.exists():
            lines.append(f"`{name}/`" if path.is_dir() else f"`{name}`")
    return lines


_FIGURE_SUFFIXES = frozenset({".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"})


def _list_figure_files(directory: Path) -> list[Path]:
    directory = directory.resolve()
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES
    )


def _discover_figures_dirs(roots: list[str | Path], run_label: str) -> list[Path]:
    """Find figures/ (or ISP UMAP PNGs) under recent run folders."""
    found: list[Path] = []
    seen: set[str] = set()

    def add(fig_dir: Path) -> None:
        key = str(fig_dir.resolve())
        if key in seen:
            return
        if _list_figure_files(fig_dir):
            seen.add(key)
            found.append(fig_dir)

    for raw in roots:
        root = Path(raw)
        if not root.is_dir():
            continue
        if root.name.startswith(("pipeline_", "isp_")):
            add(root / "figures")
            if run_label == "ISP UMAP":
                umaps = [p for p in root.glob("umap_*.png") if p.is_file()]
                if umaps:
                    add(root)
            continue
        for fig_dir in root.rglob("figures"):
            if fig_dir.is_dir() and fig_dir.parent.name.startswith(
                ("pipeline_", "isp_", "finetune_")
            ):
                add(fig_dir)
        if run_label == "ISP UMAP":
            for run_dir in sorted(
                root.glob("**/isp_umap_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:3]:
                if run_dir.is_dir() and list(run_dir.glob("umap_*.png")):
                    add(run_dir)

    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def _build_figures_zip(fig_dirs: list[Path]) -> tuple[bytes, str] | None:
    """Zip all figure files; arcnames include run folder for clarity."""
    entries: list[tuple[Path, str]] = []
    for fig_dir in fig_dirs:
        run_name = fig_dir.parent.name if fig_dir.name == "figures" else fig_dir.name
        prefix = f"{run_name}/"
        for fp in _list_figure_files(fig_dir):
            try:
                rel = fp.relative_to(fig_dir)
            except ValueError:
                rel = Path(fp.name)
            entries.append((fp, f"{prefix}{rel.as_posix()}"))

    if not entries:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp, arcname in entries:
            zf.write(fp, arcname=arcname)

    primary = fig_dirs[0].parent.name if fig_dirs[0].name == "figures" else fig_dirs[0].name
    return buf.getvalue(), f"{primary}_figures.zip"


def _latest_pipeline_run_dirs(output_root: str | Path, limit: int = 5) -> list[Path]:
    """Newest pipeline_* folders under {output_root}/{YYYYMMDD}/."""
    root = Path(str(output_root))
    if not root.is_dir():
        return []
    found: list[Path] = []
    for date_dir in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not date_dir.is_dir() or len(date_dir.name) != 8 or not date_dir.name.isdigit():
            continue
        for run_dir in date_dir.glob("pipeline_*"):
            if run_dir.is_dir():
                found.append(run_dir)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:limit]


def _read_pipeline_yaml() -> dict:
    text = st.session_state.get("yaml_editor", "")
    try:
        cfg = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _read_pipeline_data_fields() -> tuple[str, str | None]:
    """Return (input_dir, output_prefix) from the YAML editor; output_prefix None if unset/null."""
    cfg = _read_pipeline_yaml()
    data = cfg.get("data") or {}
    input_dir = str(data.get("input_dir") or "")
    raw_prefix = data.get("output_prefix")
    if raw_prefix is None or str(raw_prefix).strip().lower() in ("", "null"):
        return input_dir, None
    return input_dir, str(raw_prefix).strip()


def _read_pipeline_perturbation() -> dict:
    return _read_pipeline_yaml().get("perturbation") or {}


def _display_input_dir_line(upload_dir: Path) -> str:
    """Resolved data.input_dir for Run directory panel (YAML / session after Apply)."""
    cfg = _read_pipeline_yaml()
    from_yaml = str((cfg.get("data") or {}).get("input_dir") or "").strip()
    if from_yaml:
        return from_yaml
    cached = st.session_state.get("pipeline_tokenize_dir")
    if cached:
        return str(cached)
    tokenize_dir = _compute_tokenize_input_dir(upload_dir)
    if tokenize_dir:
        return str(tokenize_dir)
    return ""


def _display_output_root_line() -> str:
    """paths.output_root from Config YAML (refreshes after Apply / editor edits)."""
    paths = _read_pipeline_yaml().get("paths") or {}
    raw = paths.get("output_root")
    if raw is None or str(raw).strip().lower() in ("", "null"):
        return "/app/output"
    return str(raw).strip()


def _newest_pipeline_run_since(output_root: str | Path, since_ts: float) -> Path | None:
    """Newest pipeline_* folder under output_root created at or after since_ts (epoch)."""
    for run_dir in _latest_pipeline_run_dirs(output_root, limit=30):
        try:
            if run_dir.stat().st_mtime >= since_ts:
                return run_dir
        except OSError:
            continue
    return None


def _sync_pipeline_output_run_dir() -> None:
    """Set pipeline_output_run_dir after Run job creates DATE/pipeline_<UTC>/."""
    started = st.session_state.get("pipeline_job_started_ts")
    if started is None or st.session_state.get("run_type_sel") != "Pipeline (E2E)":
        return
    found = _newest_pipeline_run_since(_display_output_root_line(), float(started))
    if found is not None:
        st.session_state["pipeline_output_run_dir"] = str(found.resolve())


def _render_run_directory_input(upload_dir: Path, run_label: str) -> None:
    """data.input_dir block (Pipeline only; call after YAML editor)."""
    if run_label != "Pipeline (E2E)":
        return
    input_dir_line = _display_input_dir_line(upload_dir)
    if input_dir_line:
        st.code(f"data.input_dir (auto)\n{input_dir_line}", language="text")
    else:
        st.code("data.input_dir (auto)", language="text")


def _render_run_directory_output(run_label: str) -> None:
    """Pipeline run folder after Run job (call after YAML editor)."""
    if run_label == "Pipeline (E2E)":
        pipeline_run = st.session_state.get("pipeline_output_run_dir")
        if pipeline_run:
            st.code(f"pipeline run folder\n{pipeline_run}", language="text")
        st.caption(
            "**Run job** creates `<DATE>/pipeline_<UTC>/` under `paths.output_root` in Config YAML. "
            "The run folder path appears here after the job starts."
        )
    else:
        st.caption(
            "ISP UMAP writes `<DATE>/isp_umap_<time>/` under `/app/output` when you **Run job**."
        )


def _default_isp_start_end(states: list[str]) -> tuple[str, str]:
    if not states:
        return "Disease", "Ctrl"
    if len(states) == 1:
        return states[0], states[0]
    if "Disease" in states and "Ctrl" in states:
        return "Disease", "Ctrl"
    return states[0], states[1]


def _sync_isp_state_selectors(
    states: list[str] | None = None,
    *,
    force: bool = False,
) -> None:
    """Initialize ISP dropdown options; set values only if unset or force=True (new zip)."""
    pert = _read_pipeline_perturbation()
    detected = states if states is not None else st.session_state.get("pipeline_detected_states") or []
    options = sorted({str(s) for s in detected if s} | {str(pert.get("start_state") or "")} | {str(pert.get("end_state") or "")} - {""})
    if not options:
        options = ["Disease", "Ctrl"]

    default_start, default_end = _default_isp_start_end(detected)
    start_val = str(pert.get("start_state") or default_start)
    end_val = str(pert.get("end_state") or default_end)
    if start_val not in options:
        options = sorted(set(options) | {start_val})
    if end_val not in options:
        options = sorted(set(options) | {end_val})

    st.session_state["pipeline_isp_state_options"] = options
    if force or "pipeline_isp_start_state" not in st.session_state:
        st.session_state["pipeline_isp_start_state"] = start_val
    if force or "pipeline_isp_end_state" not in st.session_state:
        st.session_state["pipeline_isp_end_state"] = end_val


def _zip_upload_fingerprint(uploaded_file) -> str:
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"


def _raw_study_name() -> str:
    return str(st.session_state.get("pipeline_study_name") or "").strip()


def _compute_tokenize_input_dir(upload_dir: Path) -> Path | None:
    """Tokenize study root from fixed upload session + editable study name."""
    return resolve_study_tokenize_dir(upload_dir, _raw_study_name())


def _apply_study_settings_to_yaml(upload_dir: Path) -> bool:
    """Write data.input_dir and output_prefix from study name (no manual path edit)."""
    study_name = normalize_study_name(_raw_study_name())
    if not study_name:
        st.session_state["pipeline_set_input_msg"] = (
            "warning",
            "Enter an **experiment name** in Study name, then upload a .zip or click "
            "**Apply setting to Config YAML**.",
        )
        return False
    tokenize_dir = _compute_tokenize_input_dir(upload_dir)
    if tokenize_dir is None:
        target = study_folder(upload_dir, study_name)
        hint = (
            f"No Single-Cell data under `{target}`. Upload a .zip first, "
            "or set **Study name** to match the folder created on import."
        )
        others = [
            p.name
            for p in sorted(upload_dir.iterdir())
            if p.is_dir() and not p.name.startswith(".")
        ]
        if others:
            hint += f" Folders in this session: `{', '.join(others)}`."
        st.session_state["pipeline_set_input_msg"] = ("warning", hint)
        return False
    study_name = normalize_study_name(_raw_study_name())
    states = unique_states_from_samples(tokenize_dir)
    st.session_state["pipeline_detected_states"] = states
    _patch_pipeline_yaml(
        input_dir=tokenize_dir,
        output_prefix=study_name,
    )
    st.session_state["pipeline_tokenize_dir"] = str(tokenize_dir)
    return True


def _process_study_zip_upload(uploaded_file, upload_dir: Path) -> None:
    """Import zip once; do not re-run on every widget rerun."""
    buf = uploaded_file.getbuffer()
    data = buf.getvalue() if hasattr(buf, "getvalue") else bytes(buf)
    study_name = normalize_study_name(_raw_study_name())
    tokenize_dir, summary = import_study_zip(data, upload_dir, _raw_study_name())
    states = unique_states_from_samples(tokenize_dir)
    st.session_state["pipeline_detected_states"] = states
    st.session_state["pipeline_tokenize_dir"] = str(tokenize_dir)
    isp_start, isp_end = _default_isp_start_end(states)
    _sync_isp_state_selectors(states, force=True)
    _patch_pipeline_yaml(
        input_dir=tokenize_dir,
        output_prefix=study_name,
        isp_start_state=isp_start,
        isp_end_state=isp_end,
    )
    st.session_state["processed_zip_fingerprint"] = _zip_upload_fingerprint(uploaded_file)
    st.success(f"Study **{study_name}** imported. `data.input_dir` → `{tokenize_dir}`")
    st.markdown(summary)


def _infer_study_name_from_yaml(upload_dir: Path) -> str:
    input_dir, prefix = _read_pipeline_data_fields()
    try:
        rel = Path(input_dir).resolve().relative_to(upload_dir.resolve())
        if rel.parts:
            return rel.parts[0]
    except ValueError:
        pass
    if prefix:
        return str(prefix)
    return ""


def _sync_pipeline_form_from_yaml(upload_dir: Path) -> None:
    """Initialize pipeline form fields from the current YAML editor text."""
    if "pipeline_study_name" not in st.session_state:
        st.session_state["pipeline_study_name"] = _infer_study_name_from_yaml(upload_dir)
    tokenize_dir = _compute_tokenize_input_dir(upload_dir)
    if tokenize_dir is not None:
        st.session_state["pipeline_tokenize_dir"] = str(tokenize_dir)


def _detected_states_from_upload(upload_dir: Path) -> list[str]:
    tokenize_dir = resolve_study_tokenize_dir(upload_dir, _raw_study_name())
    if tokenize_dir is None:
        return []
    try:
        return unique_states_from_samples(tokenize_dir)
    except Exception:
        return []


def _patch_pipeline_yaml(
    *,
    input_dir: str | Path | None = None,
    output_prefix: str | None = "__unset__",
    isp_start_state: str | None = None,
    isp_end_state: str | None = None,
) -> bool:
    """Patch pipeline YAML fields in the editor."""
    cfg = _read_pipeline_yaml()
    cfg.setdefault("data", {})
    if input_dir is not None:
        cfg["data"]["input_dir"] = str(input_dir)
    if output_prefix != "__unset__":
        cfg["data"]["output_prefix"] = output_prefix
    if isp_start_state is not None or isp_end_state is not None:
        cfg.setdefault("perturbation", {})
        if isp_start_state is not None:
            cfg["perturbation"]["start_state"] = isp_start_state
            cfg["perturbation"]["organ_data"] = isp_start_state
        if isp_end_state is not None:
            cfg["perturbation"]["end_state"] = isp_end_state
    dumped = yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
    st.session_state["yaml_editor"] = dumped
    if input_dir is not None:
        st.session_state["pipeline_tokenize_dir"] = str(input_dir)
    return True


def _apply_isp_states_to_yaml() -> None:
    _patch_pipeline_yaml(
        isp_start_state=st.session_state.get("pipeline_isp_start_state"),
        isp_end_state=st.session_state.get("pipeline_isp_end_state"),
    )


def _poll_active_job() -> None:
    proc = st.session_state.get("active_proc")
    if proc is None:
        return
    code = proc.poll()
    if code is not None:
        st.session_state["active_proc"] = None
        st.session_state["last_exit_code"] = code
        st.session_state["last_job_finished_utc"] = datetime.now(timezone.utc).isoformat()
        _sync_pipeline_output_run_dir()


WEBUI_REPO_URL = "https://github.com/YuyaSanaki/Mouse-Geneformer-WebUI"


def main() -> None:
    st.set_page_config(page_title="Mouse Geneformer WebUI", layout="wide")
    st.title("Mouse Geneformer WebUI")
    st.markdown(
        f"[{WEBUI_REPO_URL}]({WEBUI_REPO_URL})",
    )

    _poll_active_job()

    upload_dir = _session_upload_dir()
    st.session_state["upload_session_dir"] = str(upload_dir)
    if "pipeline_study_name" not in st.session_state:
        st.session_state["pipeline_study_name"] = _infer_study_name_from_yaml(upload_dir)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Study name")
        st.text_input(
            "Study name",
            key="pipeline_study_name",
            label_visibility="collapsed",
            placeholder="Experiment name (e.g. MyExperiment)",
        )
        st.caption("Enter your **experiment name** before uploading a .zip (folder + `output_prefix`).")
        st.subheader("Upload data.zip")
        st.caption(
            "Zip with compressed `/data/` sample subfolders named `Time-State-Suffix/` "
            "(e.g. `1w-Ctrl-SingleCell/`, `1w-Disease-SingleCell/`). Each sample needs "
            "`barcodes.tsv.gz`, `features.tsv.gz`, and `matrix.tsv.gz`."
        )
        zip_upload = st.file_uploader(
            "data.zip file",
            type=["zip"],
            label_visibility="collapsed",
        )
        if zip_upload is not None:
            zip_fp = _zip_upload_fingerprint(zip_upload)
            if st.session_state.get("processed_zip_fingerprint") != zip_fp:
                try:
                    _process_study_zip_upload(zip_upload, upload_dir)
                except (zipfile.BadZipFile, ValueError) as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Upload failed: {e}")
        else:
            study_name = normalize_study_name(_raw_study_name())
            if study_name and study_folder(upload_dir, study_name).is_dir():
                st.markdown(summarize_existing_study(upload_dir, study_name))

    with c2:
        st.subheader("Run type")
        run_types = list(RUN_FILES.keys())
        default_idx = run_types.index("Pipeline (E2E)") if "Pipeline (E2E)" in run_types else 0
        st.selectbox(
            "Run type",
            run_types,
            index=default_idx,
            key="run_type_sel",
            label_visibility="collapsed",
            on_change=_load_default_yaml,
        )
        if "yaml_editor" not in st.session_state:
            _load_default_yaml()
        if st.session_state.get("run_type_sel") == "Pipeline (E2E)":
            st.info(
                "Runs **Tokenize → Fine-tune → ISP** in one job. Set paths and ISP states below "
                "(or edit full YAML). Outputs under `{paths.output_root}/{DATE}/pipeline_<UTC>/`."
            )
            _sync_pipeline_form_from_yaml(upload_dir)
            upload_states = _detected_states_from_upload(upload_dir)
            if upload_states:
                st.session_state["pipeline_detected_states"] = upload_states
            if "pipeline_isp_start_state" not in st.session_state:
                _sync_isp_state_selectors(
                    st.session_state.get("pipeline_detected_states"),
                    force=False,
                )
            detected = st.session_state.get("pipeline_detected_states") or []
            if detected:
                st.caption(f"Detected states from sample folders: **{', '.join(detected)}**")
            isp_options = st.session_state.get("pipeline_isp_state_options") or ["Disease", "Ctrl"]
            c_start, c_end = st.columns(2)
            with c_start:
                st.selectbox(
                    "ISP start_state",
                    isp_options,
                    key="pipeline_isp_start_state",
                    on_change=_apply_isp_states_to_yaml,
                    help="Perturb cells in this condition (writes `perturbation.start_state` to YAML).",
                )
            with c_end:
                st.selectbox(
                    "ISP end_state",
                    isp_options,
                    key="pipeline_isp_end_state",
                    on_change=_apply_isp_states_to_yaml,
                    help="Goal state for in-silico shift (writes `perturbation.end_state` to YAML).",
                )
            pert = _read_pipeline_perturbation()
            st.caption(
                f"ISP: start=`{pert.get('start_state', '')}` end=`{pert.get('end_state', '')}` "
                "(updates when you change the dropdowns)"
            )
            if st.button("Apply setting to Config YAML", type="secondary"):
                if _apply_study_settings_to_yaml(upload_dir):
                    st.session_state["pipeline_set_input_msg"] = (
                        "success",
                        f"Applied settings for `{st.session_state['pipeline_study_name']}` to Config YAML.",
                    )
                st.rerun()
            msg = st.session_state.pop("pipeline_set_input_msg", None)
            if msg:
                kind, text = msg
                if kind == "success":
                    st.success(text)
                else:
                    st.warning(text)

    with c3:
        st.subheader("Run directory")
        run_input_slot = st.empty()
        execute_slot = st.empty()
        run_output_slot = st.empty()
        if st.button("Reset YAML to template on disk", type="secondary"):
            _load_default_yaml()
            st.rerun()

    st.text_area("YAML", key="yaml_editor", height=420)

    if st.session_state.get("active_proc") is not None:
        _sync_pipeline_output_run_dir()

    run_label = st.session_state.get("run_type_sel", "Pipeline (E2E)")
    with run_input_slot.container():
        _render_run_directory_input(upload_dir, run_label)

    proc = st.session_state.get("active_proc")
    busy = proc is not None and proc.poll() is None
    run_clicked = False
    with execute_slot.container():
        st.subheader("Execute")
        run_clicked = st.button("Run job", type="primary", disabled=busy)

    with run_output_slot.container():
        _render_run_directory_output(run_label)

    if run_clicked:
        yaml_text = st.session_state.get("yaml_editor", "")
        try:
            cfg_obj = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML: {e}")
            cfg_obj = None

        if cfg_obj is not None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
            run_dir = _ensure_workspace() / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = run_dir / "config.yaml"
            cfg_path.write_text(yaml_text, encoding="utf-8")
            log_path = run_dir / "console.log"

            run_label = st.session_state["run_type_sel"]
            try:
                cmd, env = _build_command_and_env(run_label, cfg_path)
            except ValueError as e:
                st.error(str(e))
                cmd, env = [], os.environ.copy()

            if cmd:
                st.session_state["last_run_dir"] = run_dir
                st.session_state["last_output_roots"] = [str(p) for p in _guess_output_roots(run_label, cfg_obj)]
                st.session_state["last_exit_code"] = None
                if run_label == "Pipeline (E2E)":
                    st.session_state["pipeline_job_started_ts"] = time.time()
                    st.session_state.pop("pipeline_output_run_dir", None)
                log_file = open(log_path, "w", encoding="utf-8", buffering=1)
                try:
                    p = subprocess.Popen(
                        cmd,
                        cwd=str(ROOT),
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    log_file.close()
                st.session_state["active_proc"] = p
                st.session_state["active_log_path"] = log_path
                st.success(f"Started. Run folder: `{run_dir}`")
                st.rerun()

    st.subheader("Logs & status")
    if busy:
        st.warning("Job running…")
        log_path = st.session_state.get("active_log_path")
        if isinstance(log_path, Path):
            st.code(_tail_log(log_path), language="text")
        time.sleep(1.0)
        st.rerun()
    else:
        log_path = st.session_state.get("active_log_path")
        if isinstance(log_path, Path) and log_path.is_file():
            st.code(_tail_log(log_path), language="text")
        if st.session_state.get("last_exit_code") is not None:
            code = st.session_state["last_exit_code"]
            if code == 0:
                st.success(f"Last job finished OK (exit {code}).")
            else:
                st.error(f"Last job failed (exit {code}).")

    st.subheader("Outputs")
    run_label = st.session_state.get("run_type_sel", "")
    if run_label == "Pipeline (E2E)":
        pipeline_run = st.session_state.get("pipeline_output_run_dir")
        if pipeline_run:
            run_path = Path(pipeline_run)
            st.markdown(f"**Pipeline run folder:** `{run_path}`")
            for item in _pipeline_run_summary(run_path):
                st.caption(f"  · {item}")
            zip_payload = _build_pipeline_run_zip(run_path)
            if zip_payload:
                zip_bytes, zip_name = zip_payload
                st.download_button(
                    label=f"Download pipeline run (.zip) — {zip_name}",
                    data=zip_bytes,
                    file_name=zip_name,
                    mime="application/zip",
                    key="dl_pipeline_run_zip",
                    help=(
                        "Full contents of this pipeline run only: finetune checkpoints, "
                        "ISP outputs, figures, tokenized dataset, logs, and stage configs."
                    ),
                )
            elif st.session_state.get("last_exit_code") == 0:
                st.caption("Pipeline run folder exists but contains no files to download yet.")
        else:
            st.caption(
                "After **Run job**, the pipeline run folder appears here. "
                "Download includes only that run (not other recent pipeline folders)."
            )
    else:
        roots = st.session_state.get("last_output_roots") or []
        if roots:
            st.markdown("**Output roots:**")
            for r in roots:
                p = Path(r)
                st.write(f"- `{p}` — exists: {p.is_dir()}")
        fig_dirs = _discover_figures_dirs(roots, run_label)
        zip_payload = _build_figures_zip(fig_dirs) if fig_dirs else None
        if zip_payload:
            zip_bytes, zip_name = zip_payload
            st.download_button(
                label=f"Download figures (.zip) — {zip_name}",
                data=zip_bytes,
                file_name=zip_name,
                mime="application/zip",
                key="dl_figures_zip",
                help="PNG/PDF figures from the ISP UMAP run folder.",
            )
            for fd in fig_dirs[:3]:
                n = len(_list_figure_files(fd))
                st.caption(f"Included: `{fd}` ({n} file(s))")
        elif roots and st.session_state.get("last_exit_code") == 0:
            st.caption(
                "No figure files found yet "
                "(expected `umap_*.png` under `isp_umap_*`)."
            )

    last_run = st.session_state.get("last_run_dir")
    if last_run:
        st.markdown(f"**Last run metadata:** `{last_run}`")
        for name in ("config.yaml", "console.log"):
            fp = Path(last_run) / name
            if fp.is_file():
                st.download_button(
                    label=f"Download {name}",
                    data=fp.read_bytes(),
                    file_name=f"{Path(last_run).name}_{name}",
                    key=f"dl_{name}",
                )


main()
