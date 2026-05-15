"""
upload_to_hub.py
================
Uploads three trained artifacts to Hugging Face Hub:
  1. EfficientNet-B0  →  {HF_USERNAME}/recipe-ingredient-classifier
  2. TinyLlama LoRA merged + GGUF Q4_K_M  →  {HF_USERNAME}/recipe-llm-gguf
  3. FAISS pipeline artifacts  →  {HF_USERNAME}/{HF_SPACE_NAME}  (Space)

Usage (Colab / terminal):
    export HF_TOKEN="hf_..."
    export HF_USERNAME="your_hf_username"
    python upload_to_hub.py

    # Or run individual steps:
    python upload_to_hub.py --step 1
    python upload_to_hub.py --step 2
    python upload_to_hub.py --step 3
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("tqdm not installed. Run: pip install tqdm")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ─ override via env vars or edit directly
# ─────────────────────────────────────────────────────────────────────────────
from recetas.config import DATA, MODELS, FRUITS360_DIR, LLAMA_CPP_DIR, HF as _HF_CFG

HF_TOKEN      = _HF_CFG.token
HF_USERNAME   = _HF_CFG.username
HF_SPACE_NAME = _HF_CFG.space_name

CNN_REPO    = _HF_CFG.cnn_repo
LLM_REPO    = _HF_CFG.llm_repo
SPACE_REPO  = _HF_CFG.space_repo

# Local paths — todos desde recetas.config
EFFICIENTNET_PATH   = MODELS.cnn_weights
LORA_DIR            = MODELS.lora_dir
MERGED_DIR          = MODELS.merged_dir
GGUF_PATH           = MODELS.gguf
LLAMA_CPP_DIR       = LLAMA_CPP_DIR
FRUITS360_TRAIN_DIR = FRUITS360_DIR / "Training"

FAISS_ARTIFACTS = [
    DATA.faiss_index,
    DATA.final_parquet,
    DATA.ingredient_catalog,
    DATA.embeddings_npy,
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _check_credentials() -> None:
    if not HF_TOKEN:
        sys.exit(
            "HF_TOKEN env var is not set.\n"
            "Get a write token at https://huggingface.co/settings/tokens\n"
            "Then: export HF_TOKEN='hf_...'"
        )
    if not HF_USERNAME:
        sys.exit("HF_USERNAME env var is not set. Example: export HF_USERNAME='myname'")


def _get_api() -> HfApi:
    return HfApi(token=HF_TOKEN)


def _ensure_repo(api: HfApi, repo_id: str, repo_type: str = "model") -> None:
    """Create repo if it does not exist yet."""
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"  Repo already exists: {repo_id}")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"  Created repo: {repo_id}  (type={repo_type})")


def _upload_file(
    api: HfApi,
    local_path: Path,
    repo_id: str,
    repo_type: str = "model",
    path_in_repo: str | None = None,
) -> str:
    """Upload a single file and return the URL."""
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    dest = path_in_repo or local_path.name
    size_mb = local_path.stat().st_size / 1_048_576

    print(f"  Uploading '{local_path.name}' ({size_mb:.1f} MB) → {repo_id}/{dest}")
    url = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=dest,
        repo_id=repo_id,
        repo_type=repo_type,
        token=HF_TOKEN,
    )
    return url


def _upload_folder(
    api: HfApi,
    local_dir: Path,
    repo_id: str,
    repo_type: str = "model",
    folder_in_repo: str = "",
) -> None:
    """Upload an entire directory preserving structure."""
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {local_dir}")

    files = list(local_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    total_mb = sum(f.stat().st_size for f in files) / 1_048_576

    print(f"  Uploading folder '{local_dir}' ({len(files)} files, {total_mb:.1f} MB total)")
    for f in tqdm(files, desc="  Files", unit="file", ncols=80):
        rel = f.relative_to(local_dir)
        dest = str(Path(folder_in_repo) / rel) if folder_in_repo else str(rel)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type=repo_type,
            token=HF_TOKEN,
        )


def _verify_repo(api: HfApi, repo_id: str, repo_type: str = "model") -> None:
    """Confirm repo is reachable after upload."""
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"  Verified: {repo_id}  (last modified: {info.lastModified})")
    except Exception as e:
        print(f"  WARNING: could not verify {repo_id}: {e}")


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload EfficientNet-B0
# ─────────────────────────────────────────────────────────────────────────────

def _build_class_labels(fruits360_dir: Path) -> dict[int, str]:
    """
    Build {idx: class_name} from Fruits-360 Training subfolder names.
    Falls back to an empty dict if the directory does not exist — the caller
    should then provide class_labels.json manually before calling this function.
    """
    if not fruits360_dir.is_dir():
        print(
            f"  WARNING: Fruits-360 training dir not found at '{fruits360_dir}'.\n"
            "           Set FRUITS360_TRAIN_DIR to your local path.\n"
            "           Skipping class_labels.json generation."
        )
        return {}

    classes = sorted(
        [d.name for d in fruits360_dir.iterdir() if d.is_dir()]
    )
    return {str(idx): name for idx, name in enumerate(classes)}


def step1_upload_cnn(api: HfApi) -> None:
    print("\n" + "=" * 60)
    print("STEP 1 — EfficientNet-B0  →  " + CNN_REPO)
    print("=" * 60)

    # 1a. Build / load class_labels.json
    labels_path = MODELS.class_labels
    if labels_path.exists():
        print(f"  Found existing class_labels.json at {labels_path} — using it as-is.")
    else:
        print("  Generating class_labels.json from Fruits-360 training folders…")
        labels = _build_class_labels(FRUITS360_TRAIN_DIR)
        if not labels:
            sys.exit(
                "class_labels.json is missing and could not be generated.\n"
                "Create it manually: {\"0\": \"Apple Braeburn\", \"1\": \"Banana\", …}"
            )
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
        n = len(labels)
        print(f"  Written class_labels.json with {n} classes.")

    # 1b. Ensure model weight file exists
    if not EFFICIENTNET_PATH.exists():
        sys.exit(f"Model file not found: {EFFICIENTNET_PATH}")

    # 1c. Create HF repo
    _ensure_repo(api, CNN_REPO, repo_type="model")

    # 1d. Upload model card first so the page looks nice
    model_card = (
        "---\n"
        "license: apache-2.0\n"
        "tags:\n"
        "  - image-classification\n"
        "  - pytorch\n"
        "  - efficientnet\n"
        "  - food\n"
        "  - ingredients\n"
        "pipeline_tag: image-classification\n"
        "---\n\n"
        "# recipe-ingredient-classifier\n\n"
        "EfficientNet-B0 fine-tuned on Fruits-360 + Recipe Ingredients Dataset.\n"
        "Load with `torchvision.models.efficientnet_b0()` and `class_labels.json`.\n"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        card_path = Path(tmp_dir) / "README.md"
        card_path.write_text(model_card)
        _upload_file(api, card_path, CNN_REPO, repo_type="model")

    # 1e. Upload weights and labels
    for f in tqdm(
        [EFFICIENTNET_PATH, labels_path],
        desc="  Uploading CNN artifacts",
        unit="file",
        ncols=80,
    ):
        _upload_file(api, f, CNN_REPO, repo_type="model")

    # 1f. Verify
    _verify_repo(api, CNN_REPO, repo_type="model")
    print(f"  Done. View at: https://huggingface.co/{CNN_REPO}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Merge LoRA, convert to GGUF, upload
# ─────────────────────────────────────────────────────────────────────────────

def _merge_lora() -> None:
    """Merge LoRA adapters into the base TinyLlama model."""
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        sys.exit("Run: pip install peft transformers accelerate torch")

    if MERGED_DIR.exists() and any(MERGED_DIR.iterdir()):
        print(f"  Merged model already exists at '{MERGED_DIR}'. Skipping merge.")
        return

    print("  Loading base TinyLlama-1.1B-Chat-v1.0…")
    base = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print(f"  Loading LoRA adapters from '{LORA_DIR}'…")
    if not LORA_DIR.exists():
        sys.exit(f"LoRA directory not found: {LORA_DIR}")
    model = PeftModel.from_pretrained(base, str(LORA_DIR))

    print("  Merging adapters…")
    merged = model.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR))

    print("  Saving tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"  Merged model saved to '{MERGED_DIR}'")


def _convert_to_gguf() -> None:
    """Convert merged HF model → f16 GGUF → Q4_K_M using llama.cpp (two steps)."""
    if GGUF_PATH.exists():
        print(f"  GGUF already exists at '{GGUF_PATH}'. Skipping conversion.")
        return

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        sys.exit(
            f"llama.cpp convert script not found at '{convert_script}'.\n"
            "Clone it first: git clone https://github.com/ggerganov/llama.cpp"
        )

    # llama-quantize binary (built with cmake)
    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        sys.exit(
            f"llama-quantize not found at '{quantize_bin}'.\n"
            "Build it first:\n"
            "  cmake -B llama.cpp/build -S llama.cpp -DLLAMA_BUILD_TESTS=OFF\n"
            "  cmake --build llama.cpp/build --target llama-quantize -j$(nproc)"
        )

    # llamacpp conda env has transformers>=5.5 needed by convert_hf_to_gguf.py
    llamacpp_python = Path(os.environ.get("CONDA_PREFIX", "")).parent / "llamacpp" / "bin" / "python"
    if not llamacpp_python.exists():
        # fall back to miniconda default location
        llamacpp_python = Path.home() / "miniconda3" / "envs" / "llamacpp" / "bin" / "python"
    if not llamacpp_python.exists():
        sys.exit(
            "llamacpp conda env not found. Create it:\n"
            "  conda create -n llamacpp python=3.12 -y\n"
            "  conda run -n llamacpp pip install transformers==5.5.1 gguf numpy sentencepiece protobuf torch"
        )

    f16_path = GGUF_PATH.with_name("tinyllama-recipes-f16.gguf")

    # Step 1: HF model → f16 GGUF (uses llamacpp env with transformers 5.5)
    print("  Step 1/2: Converting HF model → f16 GGUF…")
    _run([
        str(llamacpp_python),
        str(convert_script),
        str(MERGED_DIR),
        "--outtype", "f16",
        "--outfile", str(f16_path),
    ])

    # Step 2: f16 GGUF → Q4_K_M (uses compiled llama-quantize binary)
    print("  Step 2/2: Quantizing f16 → Q4_K_M…")
    _run([str(quantize_bin), str(f16_path), str(GGUF_PATH), "Q4_K_M"])

    # Clean up intermediate f16 file
    if f16_path.exists():
        f16_path.unlink()
        print(f"  Removed intermediate f16 GGUF")

    if not GGUF_PATH.exists():
        raise FileNotFoundError(f"Conversion produced no output at '{GGUF_PATH}'")
    size_gb = GGUF_PATH.stat().st_size / 1_073_741_824
    print(f"  GGUF created: {GGUF_PATH}  ({size_gb:.2f} GB)")


def step2_merge_and_upload_llm(api: HfApi) -> None:
    print("\n" + "=" * 60)
    print("STEP 2 — TinyLlama LoRA → GGUF  →  " + LLM_REPO)
    print("=" * 60)

    _merge_lora()
    _convert_to_gguf()

    # Create repo
    _ensure_repo(api, LLM_REPO, repo_type="model")

    # Model card
    model_card = (
        "---\n"
        "license: apache-2.0\n"
        "tags:\n"
        "  - llm\n"
        "  - gguf\n"
        "  - tinyllama\n"
        "  - recipe-generation\n"
        "  - lora\n"
        "pipeline_tag: text-generation\n"
        "---\n\n"
        "# recipe-llm-gguf\n\n"
        "TinyLlama-1.1B-Chat fine-tuned with LoRA for recipe generation.\n"
        "Quantized to Q4_K_M GGUF for CPU inference with `llama-cpp-python`.\n\n"
        "```python\n"
        "from llama_cpp import Llama\n"
        "llm = Llama.from_pretrained(\n"
        f"    repo_id=\"{LLM_REPO}\",\n"
        "    filename=\"tinyllama-recipes-q4.gguf\",\n"
        ")\n"
        "```\n"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        card_path = Path(tmp_dir) / "README.md"
        card_path.write_text(model_card)
        _upload_file(api, card_path, LLM_REPO, repo_type="model")

    # Upload GGUF (large file — single call with progress)
    _upload_file(api, GGUF_PATH, LLM_REPO, repo_type="model")

    _verify_repo(api, LLM_REPO, repo_type="model")
    print(f"  Done. View at: https://huggingface.co/{LLM_REPO}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Upload FAISS artifacts to Space repo
# ─────────────────────────────────────────────────────────────────────────────

def step3_upload_faiss(api: HfApi) -> None:
    print("\n" + "=" * 60)
    print("STEP 3 — FAISS artifacts  →  Space: " + SPACE_REPO)
    print("=" * 60)

    # Verify all artifacts exist before starting any upload
    missing = [str(p) for p in FAISS_ARTIFACTS if not p.exists()]
    if missing:
        sys.exit(
            "Missing FAISS artifacts (run DP_embeddings_pipeline.ipynb first):\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    _ensure_repo(api, SPACE_REPO, repo_type="space")

    total_mb = sum(p.stat().st_size for p in FAISS_ARTIFACTS) / 1_048_576
    print(f"  {len(FAISS_ARTIFACTS)} files totalling {total_mb:.1f} MB")

    for artifact in tqdm(FAISS_ARTIFACTS, desc="  Uploading FAISS artifacts", ncols=80):
        _upload_file(api, artifact, SPACE_REPO, repo_type="space")
        # Small pause to avoid rate-limit on free tier
        time.sleep(1)

    _verify_repo(api, SPACE_REPO, repo_type="space")
    print(f"  Done. Space at: https://huggingface.co/spaces/{SPACE_REPO}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Upload recipe artifacts to HF Hub")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only a specific step (1=CNN, 2=LLM, 3=FAISS). Default: all.",
    )
    args = parser.parse_args()

    _check_credentials()
    api = _get_api()

    print(f"\nLogged in as: {HF_USERNAME}")
    print(f"  CNN repo  : {CNN_REPO}")
    print(f"  LLM repo  : {LLM_REPO}")
    print(f"  Space repo: {SPACE_REPO}")

    run_all = args.step is None
    try:
        if run_all or args.step == 1:
            step1_upload_cnn(api)
        if run_all or args.step == 2:
            step2_merge_and_upload_llm(api)
        if run_all or args.step == 3:
            step3_upload_faiss(api)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}")
        raise

    print("\n" + "=" * 60)
    print("All uploads complete.")
    print(f"  CNN:   https://huggingface.co/{CNN_REPO}")
    print(f"  LLM:   https://huggingface.co/{LLM_REPO}")
    print(f"  Space: https://huggingface.co/spaces/{SPACE_REPO}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCCIONES PARA INSTALAR llama.cpp EN WINDOWS (Alienware M16 R2)
# ─────────────────────────────────────────────────────────────────────────────
#
# El Alienware M16 R2 trae GPU RTX 4070 / 4080 — llama.cpp se puede compilar
# con CUDA para conversión mucho más rápida (~3x vs CPU solo).
#
# PRE-REQUISITOS
# ──────────────
# 1. Instalar Visual Studio 2022 Build Tools (no el IDE completo):
#      https://aka.ms/vs/17/release/vs_BuildTools.exe
#      Componentes: "Desarrollo para escritorio con C++" + MSVC + CMake
#
# 2. Instalar CUDA Toolkit 12.x (compatible con RTX 40xx):
#      https://developer.nvidia.com/cuda-downloads
#      Elegir: Windows → x86_64 → 11/12 → exe (local)
#
# 3. Instalar Git for Windows:
#      https://git-scm.com/download/win
#
# COMPILACIÓN CON CUDA
# ─────────────────────
# Abrir "x64 Native Tools Command Prompt for VS 2022" (no el PowerShell normal)
#
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp
#   cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release -j 8
#
# Los binarios quedan en:  llama.cpp\build\bin\Release\
#
# CONVERSIÓN DESDE PYTHON (el script de arriba usa esto)
# ───────────────────────────────────────────────────────
# Dentro del entorno Python del proyecto:
#   cd llama.cpp
#   pip install -r requirements.txt
#   python convert_hf_to_gguf.py ..\tinyllama-recipes-merged\ ^
#         --outtype q4_k_m --outfile ..\tinyllama-recipes-q4.gguf
#
# ALTERNATIVA SIN COMPILAR (más lenta, solo CPU, sin CUDA)
# ─────────────────────────────────────────────────────────
# pip install llama-cpp-python        # wheel precompilado para CPU
# pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
#             # wheel CUDA 12.4 precompilado (más rápido)
#
# VERIFICAR GPU EN llama.cpp (desde terminal, después de compilar)
# ──────────────────────────────────────────────────────────────────
#   build\bin\Release\llama-cli.exe -m tinyllama-recipes-q4.gguf ^
#       -n 64 -p "Suggest a pasta recipe" --n-gpu-layers 32
#
# Si ves "ggml_cuda_init: found X CUDA devices" → GPU activa.
# Ajustar --n-gpu-layers (32 para RTX 4070, 35 para RTX 4080).
#
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
