"""
deploy_to_hf.py
===============
End-to-end deployment of the recipe recommender Hugging Face Space.

Steps:
  1. Verify all required files exist locally (table with sizes + status).
  2. Stage large artifacts into space_repo/ (copy if not already there).
  3. Create the Space repo if it does not exist.
  4. Upload space_repo/ via api.upload_folder().
  5. Poll until the Space is RUNNING (max 5 minutes).
  6. Smoke test via gradio_client or raw HTTP.
  7. Print final checklist for the project report.

Usage:
    export HF_TOKEN="hf_..."
    export HF_USERNAME="your-username"
    python deploy_to_hf.py

    # Skip the slow LLM smoke test:
    python deploy_to_hf.py --skip-smoke
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# ── third-party (must be available) ──────────────────────────────────────────
try:
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    sys.exit("Run: pip install huggingface_hub")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Run: pip install tqdm")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
from recetas.config import DATA, MODELS, SPACE_REPO_DIR, HF as _HF_CFG

HF_TOKEN      = _HF_CFG.token
HF_USERNAME   = _HF_CFG.username
HF_SPACE_NAME = _HF_CFG.space_name
SPACE_REPO    = _HF_CFG.space_repo

# Directories
SPACE_DIR    = SPACE_REPO_DIR
PROJECT_ROOT = SPACE_REPO_DIR.parent

# Terminal colour helpers (disable on Windows if needed)
_USE_COLOR = sys.platform != "win32" or os.environ.get("FORCE_COLOR")
GREEN  = "\033[32m" if _USE_COLOR else ""
RED    = "\033[31m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
CYAN   = "\033[36m" if _USE_COLOR else ""
BOLD   = "\033[1m"  if _USE_COLOR else ""
RESET  = "\033[0m"  if _USE_COLOR else ""

OK      = f"{GREEN}✓ OK{RESET}"
MISSING = f"{RED}✗ MISSING{RESET}"
WARN    = f"{YELLOW}⚠ WARN{RESET}"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# FILE MANIFEST
# (name, source_directory, description)
# ─────────────────────────────────────────────────────────────────────────────
MANIFEST: list[tuple[str, Path, str]] = [
    # ── code / config (en space_repo/) ──────────────────────────────────────
    ("app_optimized.py", SPACE_DIR, "Gradio application (active)"),
    ("requirements.txt", SPACE_DIR, "Python dependencies"),
    ("README.md",        SPACE_DIR, "HF Space metadata"),
    (".gitattributes",   SPACE_DIR, "Git LFS config"),
    ("packages.txt",     SPACE_DIR, "OS-level packages"),
    # ── artefactos grandes (en data/processed/ y models/) ────────────────────
    (DATA.faiss_index.name,      DATA.faiss_index.parent,      "FAISS vector index"),
    (DATA.final_parquet.name,    DATA.final_parquet.parent,    "Recipe dataframe"),
    (DATA.ingredient_catalog.name, DATA.ingredient_catalog.parent, "Ingredient catalog"),
    (MODELS.class_labels.name,   MODELS.class_labels.parent,   "CNN class labels (opcional)"),
]

# Artefactos que viven fuera de space_repo/ y deben copiarse allí antes del upload
ARTIFACTS_TO_STAGE = [
    (src / name, SPACE_DIR / name)
    for name, src, _ in MANIFEST
    if src != SPACE_DIR
]

# Patterns ignored by api.upload_folder()
IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    "venv",
    ".venv",
    ".env",
    "*.egg-info",
    "node_modules",
    ".DS_Store",
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — credential check
# ─────────────────────────────────────────────────────────────────────────────
def check_credentials() -> None:
    errors = []
    if not HF_TOKEN:
        errors.append("HF_TOKEN is not set  →  export HF_TOKEN='hf_...'")
    if not HF_USERNAME:
        errors.append("HF_USERNAME is not set  →  export HF_USERNAME='your-name'")
    if errors:
        print(f"\n{RED}Credential errors:{RESET}")
        for e in errors:
            print(f"  {RED}•{RESET} {e}")
        sys.exit(1)
    print(f"  Deploying as  : {_c(HF_USERNAME, BOLD)}")
    print(f"  Space repo    : {_c(SPACE_REPO, CYAN)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — verify local files
# ─────────────────────────────────────────────────────────────────────────────
def step1_verify_files() -> bool:
    print(f"\n{BOLD}STEP 1  — Verifying local files{RESET}")
    print(f"  {'File':<35} {'Size':>10}   {'Source':<15} Status")
    print("  " + "─" * 75)

    # Archivos opcionales — no bloquean el deploy
    OPTIONAL = {"class_labels.json", "efficientnet_ingredients.pth"}

    all_ok = True
    for name, src_dir, desc in MANIFEST:
        path = src_dir / name
        if path.exists() and path.stat().st_size > 0:
            size_mb = path.stat().st_size / 1_048_576
            size_str = f"{size_mb:.2f} MB" if size_mb >= 0.01 else f"{path.stat().st_size} B"
            print(f"  {name:<35} {size_str:>10}   {src_dir.name:<15} {OK}")
        elif name in OPTIONAL:
            print(f"  {name:<35} {'—':>10}   {src_dir.name:<15} {WARN}  opcional ({desc})")
        else:
            print(f"  {name:<35} {'—':>10}   {src_dir.name:<15} {MISSING}  ({desc})")
            all_ok = False

    if not all_ok:
        print(f"\n  {RED}Archivos requeridos faltantes.{RESET}")
        if not (PROJECT_ROOT / "recipe_faiss.index").exists():
            print("    • recipe_faiss.index: ejecuta translate_and_rebuild.py primero")
        if not (PROJECT_ROOT / "df_final_embeddings.parquet").exists():
            print("    • df_final_embeddings.parquet: ejecuta translate_and_rebuild.py primero")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — stage large artifacts into space_repo/
# ─────────────────────────────────────────────────────────────────────────────
def step2_stage_artifacts() -> None:
    print(f"\n{BOLD}STEP 2  — Staging artifacts into {SPACE_DIR}/{RESET}")
    SPACE_DIR.mkdir(parents=True, exist_ok=True)

    OPTIONAL = {"class_labels.json", "efficientnet_ingredients.pth"}
    for src, dest in ARTIFACTS_TO_STAGE:
        name = src.name
        if not src.exists():
            if name in OPTIONAL:
                print(f"  {name:<40} {WARN} opcional — omitido")
            else:
                print(f"  {name:<40} {MISSING} — archivo requerido no encontrado")
            continue
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            print(f"  {name:<40} already staged — skip")
            continue
        size_mb = src.stat().st_size / 1_048_576
        print(f"  {name:<40} copying {size_mb:.1f} MB … ", end="", flush=True)
        shutil.copy2(src, dest)
        print(f"{OK}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — create Space repo if needed
# ─────────────────────────────────────────────────────────────────────────────
def step3_create_space(api: HfApi) -> None:
    print(f"\n{BOLD}STEP 3  — Creating Space repo{RESET}")
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
        token=HF_TOKEN,
    )
    try:
        info = api.repo_info(repo_id=SPACE_REPO, repo_type="space")
        print(f"  Space ready  :  {_c(f'https://huggingface.co/spaces/{SPACE_REPO}', CYAN)}")
        print(f"  Last modified:  {info.lastModified}")
    except Exception as exc:
        print(f"  {WARN}  Could not fetch Space info: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — upload folder
# ─────────────────────────────────────────────────────────────────────────────
def step4_upload(api: HfApi) -> None:
    print(f"\n{BOLD}STEP 4  — Uploading {SPACE_DIR} to Hub{RESET}")

    staged_files = list(SPACE_DIR.rglob("*"))
    staged_files = [f for f in staged_files if f.is_file()]
    total_mb = sum(f.stat().st_size for f in staged_files) / 1_048_576
    print(f"  {len(staged_files)} files  ·  {total_mb:.1f} MB total")
    print(f"  Ignore patterns: {IGNORE_PATTERNS}")

    with tqdm(total=1, desc="  Uploading folder", unit="batch", ncols=80) as pbar:
        api.upload_folder(
            folder_path=str(SPACE_DIR),
            repo_id=SPACE_REPO,
            repo_type="space",
            ignore_patterns=IGNORE_PATTERNS,
            token=HF_TOKEN,
            commit_message="deploy: automated upload via deploy_to_hf.py",
        )
        pbar.update(1)

    print(f"  Upload complete {OK}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — poll for RUNNING status
# ─────────────────────────────────────────────────────────────────────────────
# States that mean we should keep waiting
_TRANSIENT = {"BUILDING", "RUNNING_BUILDING", "SLEEP", "PAUSED"}
# States that mean we should abort
_FATAL     = {"CONFIG_ERROR", "BUILD_ERROR", "NO_APP_FILE"}


def step5_wait_for_running(api: HfApi, max_polls: int = 30, interval: int = 10) -> bool:
    print(f"\n{BOLD}STEP 5  — Waiting for Space to become RUNNING{RESET}")
    print(f"  Polling every {interval}s  (max {max_polls * interval}s = {max_polls * interval // 60} min)")
    space_url = f"https://huggingface.co/spaces/{SPACE_REPO}"

    for i in range(max_polls):
        try:
            runtime = api.get_space_runtime(repo_id=SPACE_REPO, token=HF_TOKEN)
            stage   = runtime.stage
        except Exception as exc:
            print(f"  [{i+1:02d}/{max_polls}]  Could not fetch runtime: {exc} — retrying…")
            time.sleep(interval)
            continue

        stage_str = _c(stage, GREEN if stage == "RUNNING" else
                               RED  if stage in _FATAL    else YELLOW)
        print(f"  [{i+1:02d}/{max_polls}]  Stage: {stage_str}", end="")

        if stage == "RUNNING":
            print(f"\n  Space is live  →  {_c(space_url, CYAN)}")
            return True

        if stage in _FATAL:
            print(f"\n  {RED}Fatal build error.{RESET}")
            print(f"  Check build logs:  {space_url}?logs=build")
            return False

        if stage == "RUNTIME_ERROR":
            # App may still be reachable; proceed with smoke test
            print(f"  — {YELLOW}runtime error, attempting smoke test anyway{RESET}")
            return True

        print(f"  — sleeping {interval}s…")
        time.sleep(interval)

    print(f"  {YELLOW}Timeout reached. Space may still be building.{RESET}")
    print(f"  Check manually:  {_c(space_url, CYAN)}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — smoke test
# ─────────────────────────────────────────────────────────────────────────────
# Gradio 4.x queue-based POST payload for fn_index 0 (find_recipes)
_SMOKE_PAYLOAD = {
    "data": [
        None,                     # image (no file upload)
        "tomato, garlic, basil",  # text query
        "any",                    # dietary
        "any",                    # speed
        "any",                    # dish type
    ],
    "fn_index": 0,
    "session_hash": "deploy_smoke",
}

_SPACE_HOST = f"https://{HF_USERNAME}-{HF_SPACE_NAME.replace('_', '-').lower()}.hf.space"


def _smoke_via_gradio_client() -> bool:
    """Preferred path: uses gradio_client which handles the queue protocol."""
    try:
        from gradio_client import Client  # type: ignore[import]
    except ImportError:
        return False

    print(f"  Using gradio_client…")
    try:
        client = Client(_SPACE_HOST, verbose=False)
        result = client.predict(
            None,
            "tomato, garlic, basil",
            "any",
            "any",
            "any",
            api_name="/find_recipes",
        )
        # find_recipes returns 7 values; first is the status markdown string
        if result and isinstance(result, (list, tuple)) and len(result) >= 4:
            print(f"  Response preview: {str(result[0])[:120]}")
            return True
        return False
    except Exception as exc:
        print(f"  gradio_client call failed: {exc}")
        return False


def _smoke_via_requests() -> bool:
    """Fallback: raw HTTP POST to the Gradio /run/predict endpoint."""
    try:
        import requests  # type: ignore[import]
    except ImportError:
        print(f"  {WARN}  requests not installed — skipping HTTP smoke test")
        return False

    url = f"{_SPACE_HOST}/run/predict"
    print(f"  POST {url}")
    try:
        resp = requests.post(url, json=_SMOKE_PAYLOAD, timeout=120)
    except requests.exceptions.Timeout:
        print(f"  {WARN}  Request timed out after 120s (Space may be sleeping — try again)")
        return False
    except Exception as exc:
        print(f"  {RED}Request error:{RESET} {exc}")
        return False

    if resp.status_code != 200:
        print(f"  {RED}HTTP {resp.status_code}{RESET}  body: {resp.text[:300]}")
        return False

    try:
        body = resp.json()
        data = body.get("data", [])
        if not data:
            print(f"  {WARN}  Empty 'data' in response: {body}")
            return False
        print(f"  Response preview: {str(data[0])[:120]}")
        return True
    except Exception as exc:
        print(f"  {RED}JSON parse error:{RESET} {exc}  |  raw: {resp.text[:200]}")
        return False


def step6_smoke_test() -> bool:
    print(f"\n{BOLD}STEP 6  — Smoke test{RESET}")
    print(f"  Endpoint: {_c(_SPACE_HOST, CYAN)}")
    print(f"  Query   : 'tomato, garlic, basil' | vegetarian | fast")

    # Try gradio_client first, fall back to raw HTTP
    ok = _smoke_via_gradio_client() or _smoke_via_requests()

    if ok:
        print(f"  {GREEN}{BOLD}Smoke test PASSED ✓{RESET}")
    else:
        print(f"  {YELLOW}Smoke test could not be verified automatically.{RESET}")
        print(f"  Test manually:  {_c(f'https://huggingface.co/spaces/{SPACE_REPO}', CYAN)}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — final checklist
# ─────────────────────────────────────────────────────────────────────────────
def step7_print_checklist(smoke_ok: bool) -> None:
    smoke_badge = f"{GREEN}PASS{RESET}" if smoke_ok else f"{YELLOW}MANUAL CHECK NEEDED{RESET}"
    print(f"""
{BOLD}{'=' * 62}{RESET}
{BOLD}  CHECKLIST PARA EL REPORTE{RESET}
{BOLD}{'=' * 62}{RESET}

  Space URL :  https://huggingface.co/spaces/{SPACE_REPO}
  CNN repo  :  https://huggingface.co/{_HF_CFG.cnn_repo}
  LLM repo  :  https://huggingface.co/{_HF_CFG.llm_repo}
  Dataset   :  https://huggingface.co/datasets/wafaaelhusseini/extended-recipes-dataset-64k-dishes

  Modos de entrada implementados:
    {GREEN}[x]{RESET} Foto de ingrediente  →  CNN  →  FAISS
    {GREEN}[x]{RESET} Texto libre          →  FAISS
    {GREEN}[x]{RESET} Filtros (dieta, velocidad)
    {GREEN}[x]{RESET} Ejemplos predefinidos (gr.Examples)

  Salidas implementadas:
    {GREEN}[x]{RESET} Panel A — Ingredientes detectados (imagen o badge de color)
    {GREEN}[x]{RESET} Panel B — Top-5 recetas con imagen del platillo
    {GREEN}[x]{RESET} Panel C — Receta narrada por LLM (streaming, lazy load)
    {GREEN}[x]{RESET} Panel D — Chat sobre la receta activa (TinyLlama)
    {GREEN}[x]{RESET} Panel E — Transparencia del pipeline (query + scores + timing)

  Optimizaciones:
    {GREEN}[x]{RESET} lru_cache + threading.Lock en CNN y LLM
    {GREEN}[x]{RESET} Carga lazy — tiempo de arranque del Space < 5 s
    {GREEN}[x]{RESET} Resultados en dos fases: Phase 1 < 3 s, Phase 2 on-demand
    {GREEN}[x]{RESET} Git LFS para .gguf / .index / .parquet / .npy / .pth

  Infraestructura:
    {GREEN}[x]{RESET} CPU-only (tier gratuito de HF Spaces)
    {GREEN}[x]{RESET} queue(max_size=3) para throttling
    {GREEN}[x]{RESET} packages.txt: libgomp1 (OpenMP para FAISS + PyTorch)

  Smoke test automático: {smoke_badge}
{BOLD}{'=' * 62}{RESET}
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy recipe recommender Space to HF Hub")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the smoke test (useful when Space LLM takes >2 min to cold-start)",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip steps 2-4 (verify + checklist only)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 62}{RESET}")
    print(f"{BOLD}  Recipe Demo — HF Space Deployment{RESET}")
    print(f"{BOLD}{'=' * 62}{RESET}")

    # ── credentials ───────────────────────────────────────────────────────────
    check_credentials()
    api = HfApi(token=HF_TOKEN)

    # ── step 1 ─────────────────────────────────────────────────────────────
    files_ok = step1_verify_files()
    if not files_ok:
        print(f"\n{RED}Aborting — fix missing files before deploying.{RESET}")
        sys.exit(1)

    if args.skip_upload:
        print(f"\n{YELLOW}--skip-upload set: skipping steps 2-4.{RESET}")
    else:
        # ── step 2 ─────────────────────────────────────────────────────────
        step2_stage_artifacts()

        # ── step 3 ─────────────────────────────────────────────────────────
        step3_create_space(api)

        # ── step 4 ─────────────────────────────────────────────────────────
        step4_upload(api)

    # ── step 5 ─────────────────────────────────────────────────────────────
    running = step5_wait_for_running(api)

    # ── step 6 ─────────────────────────────────────────────────────────────
    smoke_ok = False
    if args.skip_smoke:
        print(f"\n{BOLD}STEP 6  — Smoke test{RESET}  {YELLOW}(skipped){RESET}")
    elif running:
        smoke_ok = step6_smoke_test()
    else:
        print(f"\n{BOLD}STEP 6  — Smoke test{RESET}  {YELLOW}(skipped — Space not RUNNING){RESET}")

    # ── step 7 ─────────────────────────────────────────────────────────────
    step7_print_checklist(smoke_ok)


if __name__ == "__main__":
    main()
