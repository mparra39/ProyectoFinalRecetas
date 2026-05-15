from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


class _DataPaths:
    root      = ROOT / "data"
    raw       = root / "raw"
    interim   = root / "interim"
    processed = root / "processed"

    raw_parquet         = raw / "df_clean_final.parquet"
    with_images_parquet = interim / "df_clean_with_dish_images.parquet"
    translated_parquet  = interim / "df_clean_final_es.parquet"
    ingredient_coverage = interim / "ingredient_coverage.csv"
    final_parquet       = processed / "df_final_embeddings.parquet"
    faiss_index         = processed / "recipe_faiss.index"
    embeddings_npy      = processed / "recipe_embeddings.npy"
    ingredient_catalog  = processed / "ingredient_catalog.json"

DATA = _DataPaths()


class _ModelPaths:
    root         = ROOT / "models"
    cnn_weights  = root / "efficientnet_ingredients.pth"
    class_labels = root / "class_labels.json"
    lora_dir     = root / "tinyllama-recipes-lora"
    merged_dir   = root / "tinyllama-recipes-merged"
    gguf         = root / "tinyllama-recipes-q4.gguf"

MODELS = _ModelPaths()

NOTEBOOKS = ROOT / "notebooks"


class _ReportPaths:
    root        = ROOT / "reports"
    figures     = root / "figures"
    eval_json   = root / "evaluation_report.json"
    limitations = root / "limitations.csv"

REPORTS = _ReportPaths()

SPACE_REPO_DIR = ROOT / "space_repo"


class _HFConfig:
    token      = os.environ.get("HF_TOKEN",      "")
    username   = os.environ.get("HF_USERNAME",   "ramonsj11")
    space_name = os.environ.get("HF_SPACE_NAME", "ProyectoFinal_recetas")

    @property
    def cnn_repo(self)   -> str: return f"{self.username}/recipe-ingredient-classifier"
    @property
    def llm_repo(self)   -> str: return f"{self.username}/recipe-llm-gguf"
    @property
    def space_repo(self) -> str: return f"{self.username}/{self.space_name}"

HF = _HFConfig()

EMBED_MODEL_NAME     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSLATE_MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
BASE_LLM_NAME        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLAMA_CPP_DIR        = ROOT / "llama.cpp"

FRUITS360_DIR = Path(
    os.environ.get(
        "FRUITS360_DIR",
        str(Path.home() / ".cache/kagglehub/datasets/moltean/fruits"
            "/versions/89/fruits-360_100x100/fruits-360"),
    )
)
