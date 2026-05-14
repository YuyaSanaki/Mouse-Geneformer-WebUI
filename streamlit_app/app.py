"""
Mouse Geneformer — Streamlit control panel (same image as CLI; jobs run via subprocess).

Layout: data upload | run type | YAML configuration | execute + live log + outputs.
"""
from __future__ import annotations

import io
import os
import shlex
import subprocess
import textwrap
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import yaml

# Repository root: Docker WORKDIR is /app; local dev uses current working directory.
ROOT = Path(os.environ.get("WEBUI_ROOT", os.getcwd())).resolve()
WORKSPACE = Path(os.environ.get("WEBUI_WORKSPACE", ROOT / "data" / "streamlit_workspace")).resolve()

RUN_FILES = {
    "ISP": "isp.yaml",
    "Tokenize": "tokenize.yaml",
    "Fine-tune": "finetune.yaml",
    "ISP UMAP": "isp_umap.yaml",
}


def _default_config_path(run_label: str) -> Path:
    return ROOT / "config" / RUN_FILES[run_label]


def _load_default_yaml() -> None:
    name = st.session_state.get("run_type_sel", "ISP")
    path = _default_config_path(name)
    if path.is_file():
        st.session_state["yaml_editor"] = path.read_text(encoding="utf-8")
    else:
        st.session_state["yaml_editor"] = f"# Missing file: {path}\n"


def _ensure_workspace() -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return WORKSPACE


def _session_upload_dir() -> Path:
    sid = st.session_state.setdefault("upload_session_id", uuid.uuid4().hex[:10])
    d = _ensure_workspace() / "uploads" / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_extract_zip(zbytes: bytes, dest: Path) -> Path:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = (dest / member.filename).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dest)
    return dest


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

    if run_label == "ISP":
        nproc = os.environ.get("ISP_NUM_GPUS", "1")
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(nproc),
            str(ROOT / "run_isp.py"),
            "--config",
            cfg,
        ]
    elif run_label == "Tokenize":
        cmd = ["python3", str(ROOT / "execute_tokenizer_pipeline.py")]
        env["TOKENIZE_CONFIG"] = cfg
    elif run_label == "Fine-tune":
        cmd = ["python3", str(ROOT / "run_finetune.py"), "--config", cfg]
        env["FINETUNE_CONFIG"] = cfg
    elif run_label == "ISP UMAP":
        cmd = ["python3", str(ROOT / "run_isp_umap.py"), "--config", cfg]
        env["ISP_UMAP_CONFIG"] = cfg
    else:
        raise ValueError(run_label)
    return cmd, env


def _guess_output_roots(run_label: str, cfg: dict) -> list[Path]:
    roots: list[Path] = []
    if run_label == "Tokenize":
        data = cfg.get("data") or {}
        if data.get("output_dir"):
            roots.append(Path(str(data["output_dir"])))
    elif run_label in ("ISP", "Fine-tune"):
        paths = cfg.get("paths") or {}
        if paths.get("output_root"):
            roots.append(Path(str(paths["output_root"])))
    elif run_label == "ISP UMAP":
        roots.append(ROOT / "output")
    return roots


def _poll_active_job() -> None:
    proc = st.session_state.get("active_proc")
    if proc is None:
        return
    code = proc.poll()
    if code is not None:
        st.session_state["active_proc"] = None
        st.session_state["last_exit_code"] = code
        st.session_state["last_job_finished_utc"] = datetime.now(timezone.utc).isoformat()


def main() -> None:
    st.set_page_config(page_title="Mouse Geneformer", layout="wide")
    st.title("Mouse Geneformer")
    st.caption(
        f"Repository root: `{ROOT}` · Workspace: `{WORKSPACE}` · "
        "Jobs run in this container (subprocess), same code as CLI."
    )

    _poll_active_job()

    upload_dir = _session_upload_dir()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Data upload")
        st.caption(
            "Files are saved under the workspace. Upload limit is raised in `.streamlit/config.toml` "
            "and Compose (`STREAMLIT_SERVER_MAX_UPLOAD_SIZE`, default here ~1 TB). "
            "For very large data, copying into the bind-mounted `data/` tree and pointing YAML at "
            "that path avoids holding the whole file in RAM."
        )
        uploaded = st.file_uploader("Upload", accept_multiple_files=True)
        if uploaded:
            for uf in uploaded:
                out = upload_dir / uf.name
                out.write_bytes(uf.getbuffer())
                st.success(f"Saved `{out}`")
            for uf in uploaded:
                if uf.name.lower().endswith(".zip"):
                    try:
                        dest = upload_dir / uf.name.removesuffix(".zip")
                        extracted = _safe_extract_zip(uf.getbuffer(), dest)
                        st.info(
                            f"Extracted zip to `{extracted}` — set `data.input_dir` "
                            "(or loom paths) in YAML to this folder."
                        )
                    except (zipfile.BadZipFile, ValueError) as e:
                        st.error(f"Zip error: {e}")
        st.text_area("Upload directory (read-only)", value=str(upload_dir), height=68, disabled=True)

    with c2:
        st.subheader("Run type")
        st.selectbox(
            "Pipeline",
            list(RUN_FILES.keys()),
            key="run_type_sel",
            on_change=_load_default_yaml,
        )
        if "yaml_editor" not in st.session_state:
            _load_default_yaml()
        st.markdown(
            textwrap.dedent(
                """
                | Type | Script |
                |------|--------|
                | ISP | `run_isp.py` (via `accelerate launch`) |
                | Tokenize | `execute_tokenizer_pipeline.py` |
                | Fine-tune | `run_finetune.py` |
                | ISP UMAP | `run_isp_umap.py` |
                """
            )
        )

    with c3:
        st.subheader("Run configuration")
        st.caption(
            "Edit YAML below. Paths are container paths (e.g. `/app/...`). "
            "Reload defaults when switching run type."
        )
        if st.button("Reset YAML to template on disk", type="secondary"):
            _load_default_yaml()
            st.rerun()

    st.text_area("YAML", key="yaml_editor", height=420)

    st.subheader("Execute")
    proc = st.session_state.get("active_proc")
    busy = proc is not None and proc.poll() is None

    col_run, col_cmd = st.columns([1, 2])
    with col_run:
        run_clicked = st.button("Run job", type="primary", disabled=busy)
    with col_cmd:
        if st.session_state.get("last_cmd_display"):
            st.code(st.session_state["last_cmd_display"], language="bash")

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
                st.session_state["last_cmd_display"] = " ".join(shlex.quote(c) for c in cmd)
                extra: list[str] = []
                if run_label == "Tokenize":
                    extra.append(f"TOKENIZE_CONFIG={shlex.quote(str(cfg_path))}")
                elif run_label == "Fine-tune":
                    extra.append(f"FINETUNE_CONFIG={shlex.quote(str(cfg_path))}")
                elif run_label == "ISP UMAP":
                    extra.append(f"ISP_UMAP_CONFIG={shlex.quote(str(cfg_path))}")
                if extra:
                    st.session_state["last_cmd_display"] = (
                        "export " + " ".join(extra) + " && \\\n" + st.session_state["last_cmd_display"]
                    )

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
    roots = st.session_state.get("last_output_roots") or []
    if roots:
        st.markdown("**Output roots from YAML:**")
        for r in roots:
            p = Path(r)
            st.write(f"- `{p}` — exists: {p.is_dir()}")
            if p.is_dir():
                try:
                    subs = sorted(p.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:12]
                    for s in subs:
                        st.caption(f"  · `{s.name}`")
                except OSError:
                    pass
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
