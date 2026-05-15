"""
app_optimized.py  ―  Demo Multimodal de Recomendador de Recetas  (build optimizado)
Hugging Face Space  |  Solo CPU  |  Gradio 4.44
Notas de optimización:
  • CNN + LLM se cargan de forma lazy en el primer uso (lru_cache + threading.Lock).
  • UX en dos fases: Fase 1 (<3 s) = ingredientes + tabla de recetas;
                     Fase 2 (~30 s) = narración LLM, activada por el usuario.
  • Panel de ingredientes gr.HTML — imágenes reales O badges de texto coloreado.
  • Panel de transparencia del pipeline — consulta, puntuaciones, tiempos por etapa.
  • gr.Examples — 5 consultas de texto predefinidas para demos instantáneas.
"""

# ── biblioteca estándar ───────────────────────────────────────────────────────
import base64
import functools
import json
import os
import threading
import time
from pathlib import Path

# ── terceros ──────────────────────────────────────────────────────────────────
import faiss
import gradio as gr

# ── Parche 1: gradio_client 0.6.x — los valores bool de JSON-Schema provocan TypeError ───
import gradio_client.utils as _gcu

_orig_get_type = _gcu.get_type
_orig_jstpt    = _gcu._json_schema_to_python_type

def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)

def _safe_jstpt(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_jstpt(schema, defs)

_gcu.get_type                    = _safe_get_type
_gcu._json_schema_to_python_type = _safe_jstpt

# ── Parche 2: Starlette >=1.0 cambió TemplateResponse(name, ctx) → (req, name) ─
import starlette.templating as _st

_orig_TemplateResponse = _st.Jinja2Templates.TemplateResponse

def _compat_TemplateResponse(self, *args, **kwargs):
    # API antigua (Starlette <1.0): TemplateResponse(name: str, context: dict, ...)
    # API nueva  (Starlette >=1.0): TemplateResponse(request, name: str, context=...)
    if args and isinstance(args[0], str):
        name    = args[0]
        context = args[1] if len(args) > 1 else kwargs.pop("context", {})
        request = context.get("request")
        return _orig_TemplateResponse(self, request, name, context=context, **kwargs)
    return _orig_TemplateResponse(self, *args, **kwargs)

_st.Jinja2Templates.TemplateResponse = _compat_TemplateResponse
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
try:
    from llama_cpp import Llama
    _LLAMA_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[assignment, misc]
    _LLAMA_AVAILABLE = False
    print("llama-cpp-python no disponible — LLM desactivado")
from PIL import Image
from rapidfuzz import process as rfprocess
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
HF_USERNAME   = os.environ.get("HF_USERNAME",   "ramonsj11")
HF_SPACE_NAME = os.environ.get("HF_SPACE_NAME", "ProyectoFinal_recetas")
CNN_REPO      = f"{HF_USERNAME}/recipe-ingredient-classifier"
LLM_REPO      = f"{HF_USERNAME}/recipe-llm-gguf"
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_SHORT  = "multilingual-MiniLM-L12-v2  ·  384-dim"

DIETARY_CHOICES = ["any", "vegetarian", "vegan", "gluten-free", "dairy-free"]
SPEED_CHOICES   = ["any", "fast", "medium", "slow"]

# Si la confianza del modelo está por debajo de este porcentaje, ignoramos ese ingrediente.
# Esto evita ingredientes "alucinados" cuando la imagen no es clara.
CNN_CONFIDENCE_THRESHOLD = 0.15   # 15%

# Si la predicción TOP-1 no alcanza este umbral, el modelo probablemente está confundido con la imagen.
CNN_TOP1_MIN_CONFIDENCE  = 0.35   # 35%

# Número máximo de ingredientes a mostrar en condiciones normales.
CNN_MAX_RESULTS = 5

# Paleta pastel para badges de ingredientes no encontrados en el catálogo
_BADGE_COLORS = [
    "#FFB3B3", "#B3D9FF", "#B3FFB3", "#FFD9B3",
    "#E8B3FF", "#B3FFE8", "#FFE8B3", "#D9B3FF",
]
_GREY = Image.new("RGB", (200, 200), color=(210, 210, 210))

# ─────────────────────────────────────────────────────────────────────────────
# ARTEFACTOS DE ARRANQUE — FAISS + embeddings (rápido, siempre necesario)
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando índice FAISS…")
faiss_index = faiss.read_index("recipe_faiss.index")

print("Cargando dataframe…")
df = pd.read_parquet("df_final_embeddings.parquet").reset_index(drop=True)

with open("ingredient_catalog.json") as _f:
    ingredient_catalog: dict[str, str] = json.load(_f)

try:
    with open("class_labels.json") as _f:
        class_labels: dict[str, str] = json.load(_f)
    print(f"  class_labels.json: {len(class_labels)} clases")
except FileNotFoundError:
    class_labels = {}
    print("  class_labels.json no encontrado — CNN desactivado")

NUM_CLASSES  = len(class_labels)
_catalog_keys = list(ingredient_catalog.keys())

# Compatibilidad de nombres de columna — se prefiere la columna en español si existe
if "ingredient_text_es" in df.columns:
    INGR_COL = "ingredient_text_es"
elif "ingredient_text" in df.columns:
    INGR_COL = "ingredient_text"
else:
    INGR_COL = "ingredients_text_processed"
DIETARY_COL = "dietary_profile" if "dietary_profile" in df.columns else "dietary_profile_updated"
CUISINE_COL = "cuisine_list"    if "cuisine_list"    in df.columns else "cuisine"
DISH_TYPE_COLS = [c for c in ("course_list", "course", "category", "subcategory") if c in df.columns]

print(f"  {len(df):,} recetas | ingr_col={INGR_COL} | dietary_col={DIETARY_COL}")

print("Cargando SentenceTransformer…")
# Modelo multilingüe — soporta consultas en español e inglés (384-dim, igual que antes)
embedding_model = SentenceTransformer(EMBED_MODEL)

print("Artefactos de arranque listos ✅  CNN + LLM se cargarán en el primer uso.")

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZACIÓN 1 + 2 — cargadores lazy con lru_cache y getters seguros para hilos
# ─────────────────────────────────────────────────────────────────────────────
_cnn_lock = threading.Lock()
_llm_lock = threading.Lock()

_cnn_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@functools.lru_cache(maxsize=1)
def _load_cnn_cached() -> torch.nn.Module:
    """Descarga pesos y construye el modelo exactamente una vez; resultado cacheado en el proceso."""
    if NUM_CLASSES == 0:
        raise RuntimeError("class_labels.json no encontrado — CNN no disponible")
    weights_path = hf_hub_download(repo_id=CNN_REPO, filename="efficientnet_ingredients.pth")
    mdl = models.efficientnet_b0(weights=None)
    mdl.classifier[1] = torch.nn.Linear(1280, NUM_CLASSES)
    mdl.load_state_dict(torch.load(weights_path, map_location="cpu"))
    mdl.eval()
    return mdl


@functools.lru_cache(maxsize=1)
def _load_llm_cached() -> "Llama":
    """Descarga GGUF e inicializa Llama exactamente una vez; resultado cacheado en el proceso."""
    if not _LLAMA_AVAILABLE:
        raise RuntimeError("llama-cpp-python no instalado — LLM no disponible")
    gguf_path = hf_hub_download(repo_id=LLM_REPO, filename="tinyllama-recipes-q4.gguf")
    return Llama(model_path=gguf_path, n_ctx=2048, n_threads=4, verbose=False)


def get_cnn() -> torch.nn.Module:
    """Getter lazy seguro para hilos — seguro para llamar desde peticiones Gradio concurrentes."""
    with _cnn_lock:
        return _load_cnn_cached()


def get_llm() -> Llama:
    """Getter lazy seguro para hilos — seguro para llamar desde peticiones Gradio concurrentes."""
    with _llm_lock:
        return _load_llm_cached()


def _cnn_loaded() -> bool:
    return _load_cnn_cached.cache_info().currsize > 0


def _llm_loaded() -> bool:
    return _load_llm_cached.cache_info().currsize > 0


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 1 — clasificación de ingredientes  (CNN lazy)
# ─────────────────────────────────────────────────────────────────────────────
def classify_ingredients(image: Image.Image) -> tuple[list[tuple[str, float]], bool]:
    """
    Devuelve (lista_ingredientes, imagen_poco_confiable).
    - Solo incluye predicciones con confianza >= CNN_CONFIDENCE_THRESHOLD.
    - imagen_poco_confiable = True cuando TOP-1 no supera CNN_TOP1_MIN_CONFIDENCE,
      indicando que la imagen no es ideal para el modelo.
    """
    model  = get_cnn()
    tensor = _cnn_tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top10 = torch.topk(probs, 10)

    all_results = [
        (class_labels.get(str(i.item()), f"clase_{i.item()}"), s.item())
        for i, s in zip(top10.indices, top10.values)
    ]

    # ── Detectar imagen poco confiable ──────────────────────────────────────
    top1_confidence = all_results[0][1] if all_results else 0.0
    low_confidence_image = top1_confidence < CNN_TOP1_MIN_CONFIDENCE

    # ── Filtrar por umbral mínimo ────────────────────────────────────────────
    filtered = [
        (name, conf) for name, conf in all_results
        if conf >= CNN_CONFIDENCE_THRESHOLD
    ]

    # Si ninguno supera el umbral, devolver solo el top-1 como mejor esfuerzo
    if not filtered:
        filtered = all_results[:1]

    return filtered[:CNN_MAX_RESULTS], low_confidence_image


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 5 — búsqueda de imagen de ingrediente
# ─────────────────────────────────────────────────────────────────────────────
def get_ingredient_image(name: str) -> str | None:
    """Búsqueda difusa del nombre en el catálogo (umbral 72); devuelve ruta o None."""
    hit = rfprocess.extractOne(name.lower(), _catalog_keys)
    if hit and hit[1] >= 72:
        return ingredient_catalog[hit[0]]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 2 — recuperación de recetas  (también devuelve cadena de consulta + puntuaciones)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_dietary(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x).lower() for x in raw]
    try:
        return [str(x).lower() for x in json.loads(raw)]
    except Exception:
        return [str(raw).lower()]


def _stringify(val) -> str:
    if isinstance(val, list):
        return ", ".join(str(x) for x in val)
    try:
        return ", ".join(str(x) for x in json.loads(val))
    except Exception:
        return str(val) if pd.notna(val) else ""


def _choice_values_from_columns(frame: pd.DataFrame, columns: list[str], limit: int = 40) -> list[str]:
    values: set[str] = set()
    for col in columns:
        for raw in frame[col].dropna().head(20000):
            text = _stringify(raw) if "_list" in col else str(raw)
            for item in text.split(","):
                item = item.strip()
                if item and item.lower() not in {"nan", "none", "[]"}:
                    values.add(item)
    return ["any"] + sorted(values)[:limit]


DISH_TYPE_CHOICES = _choice_values_from_columns(df, DISH_TYPE_COLS)


def _contains_choice(raw, choice: str) -> bool:
    if choice == "any":
        return True
    return choice.lower() in _stringify(raw).lower()


def _text_blob(row: dict) -> str:
    parts = [
        row.get("recipe_title", ""),
        row.get(INGR_COL, ""),
        row.get("ingredients_text_processed", ""),
        row.get("directions_text", ""),
        row.get("description", ""),
        row.get("category", ""),
        row.get("subcategory", ""),
        row.get("course", ""),
        row.get("course_list", ""),
    ]
    return " ".join(_stringify(p).lower() for p in parts if p is not None)


def _ingredient_overlap(query_terms: list[str], row: dict) -> float:
    terms = [t.lower().strip() for t in query_terms if t and t.strip()]
    if not terms:
        return 0.0
    blob = _text_blob(row)
    return sum(1 for term in terms if term in blob) / len(terms)


def _has_dish_image(row: dict) -> float:
    path = row.get("dish_image_path") or row.get("image_path") or ""
    return 1.0 if path and Path(path).exists() else 0.0


class MLPReranker(torch.nn.Module):
    """Pequeño MLP determinista sobre características de recuperación/filtrado."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(7, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )
        self._init_reasonable_weights()

    def _init_reasonable_weights(self) -> None:
        with torch.no_grad():
            first: torch.nn.Linear = self.net[0]  # type: ignore[assignment]
            second: torch.nn.Linear = self.net[2]  # type: ignore[assignment]
            first.weight.zero_()
            first.bias.zero_()
            for i in range(7):
                first.weight[i, i] = 1.0
            first.weight[7] = torch.tensor([0.8, 1.0, 0.5, 0.35, 0.25, 0.45, 0.2])
            second.weight[:] = torch.tensor([[1.5, 1.2, 0.7, 0.5, 0.35, 0.75, 0.25, 1.0]])
            second.bias.zero_()

    @torch.no_grad()
    def score(self, features: list[list[float]]) -> list[float]:
        if not features:
            return []
        tensor = torch.tensor(features, dtype=torch.float32)
        return self.net(tensor).squeeze(-1).tolist()


reranker = MLPReranker()


def rerank_recipes(
    cands: pd.DataFrame,
    ingredients: list[str],
    dietary_filter: str,
    speed_filter: str,
    dish_type_filter: str,
) -> pd.DataFrame:
    rows = cands.to_dict(orient="records")
    features: list[list[float]] = []
    for row in rows:
        features.append([
            float(row.get("_score", 0.0)),
            _ingredient_overlap(ingredients, row),
            1.0 if dietary_filter == "any" or _contains_choice(row.get(DIETARY_COL, ""), dietary_filter) else 0.0,
            1.0 if speed_filter == "any" or str(row.get("cook_speed", "")).lower() == speed_filter.lower() else 0.0,
            1.0 if dish_type_filter == "any" or any(_contains_choice(row.get(col, ""), dish_type_filter) for col in DISH_TYPE_COLS) else 0.0,
            1.0 if any(term.lower() in str(row.get("recipe_title", "")).lower() for term in ingredients) else 0.0,
            _has_dish_image(row),
        ])
    ranked = cands.copy()
    ranked["_rerank_score"] = reranker.score(features)
    return ranked.sort_values("_rerank_score", ascending=False)


def retrieve_recipes(
    ingredients: list[str],
    dietary_filter: str = "any",
    speed_filter: str = "any",
    dish_type_filter: str = "any",
    k: int = 5,
) -> tuple[list[dict], str, list[float]]:
    """Devuelve (lista_recetas, texto_consulta, puntuaciones_reranker)."""
    query = "ingredients: " + ", ".join(ingredients)
    emb   = embedding_model.encode([query], normalize_embeddings=True).astype("float32")

    dists, idxs = faiss_index.search(emb, 50)
    cands = df.iloc[idxs[0]].copy()
    cands["_score"] = dists[0]

    if dietary_filter != "any":
        mask  = cands[DIETARY_COL].apply(lambda v: dietary_filter.lower() in _parse_dietary(v))
        cands = cands[mask]

    if speed_filter != "any" and "cook_speed" in cands.columns:
        cands = cands[cands["cook_speed"].str.lower() == speed_filter.lower()]

    if dish_type_filter != "any" and DISH_TYPE_COLS:
        mask = cands.apply(
            lambda row: any(_contains_choice(row.get(col, ""), dish_type_filter) for col in DISH_TYPE_COLS),
            axis=1,
        )
        cands = cands[mask]

    ranked = rerank_recipes(cands, ingredients, dietary_filter, speed_filter, dish_type_filter)
    top    = ranked.head(k)
    scores = top["_rerank_score"].tolist()
    top = top.drop_duplicates(subset=["recipe_title"], keep="first")
    return top.to_dict(orient="records"), query, scores


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 3 — narración LLM en streaming  (LLM lazy)
# ─────────────────────────────────────────────────────────────────────────────
def _narration_prompt(row: dict) -> str:
    title = row.get("recipe_title", "Unknown recipe")
    ingr  = row.get(INGR_COL) or row.get("ingredients_text_processed", "")

    # Limitar longitud del contexto de ingredientes para acelerar la generación
    ingr  = ingr[:300]
    ingr  = ingr.replace(" ", ", ") if " " in ingr and "," not in ingr else ingr
    dirs  = row.get("directions_text", "")[:400]

    # Prompt más conciso → respuesta más corta y rápida
    return (
        "<|system|>\n"
        "You are a friendly cooking assistant. Be concise and enthusiastic.\n</s>\n"
        "<|user|>\n"
        f"In 3-4 sentences, describe how to make '{title}'. "
        f"Key ingredients: {ingr}. "
        f"Main steps: {dirs}\n</s>\n"
        "<|assistant|>\n"
    )


def build_recipe_detail_md(row: dict | None, detected_names: list[str] | None = None) -> str:
    if not row:
        return "Selecciona una receta para ver ingredientes y procedimiento."
    title = row.get("recipe_title", "Receta")
    ingredients = row.get(INGR_COL) or row.get("ingredients_text_processed", "")
    ingredients = ingredients.replace(" ", ", ") if " " in ingredients and "," not in ingredients else ingredients
    directions = row.get("directions_text", "") or row.get("directions", "")
    cuisine = _stringify(row.get(CUISINE_COL, ""))
    dietary = _stringify(row.get(DIETARY_COL, ""))
    speed = row.get("cook_speed", "")
    meta = " · ".join(str(x) for x in [cuisine, dietary, speed] if str(x).strip())

    # ── Resaltar ingredientes detectados ──────────────────────────────────
    if detected_names:
        for name in detected_names:
            # Reemplaza solo si el nombre aparece como palabra completa
            # (evita resaltar "egg" dentro de "eggplant")
            import re
            pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
            ingredients = pattern.sub(f"**{name}**", ingredients)
            directions  = pattern.sub(f"**{name}**", directions)

    return (
        f"### {title}\n\n"
        f"{meta}\n\n"
        f"**Ingredientes**\n\n{ingredients or 'No disponible'}\n\n"
        f"**Procedimiento**\n\n{directions or 'No disponible'}"
    )


def generate_recipe(recipe_row: dict | None):
    """Generador — hace streaming de la narración; muestra gr.Info en la primera carga del LLM."""
    if not recipe_row:
        yield "Selecciona una receta de la tabla de arriba y haz clic en 'Narrar'."
        return
    if not _llm_loaded():
        gr.Info("Cargando el modelo de lenguaje por primera vez (~25 s) — por favor espera…")
    model       = get_llm()
    accumulated = ""
    for chunk in model(_narration_prompt(recipe_row), max_tokens=512, temperature=0.7, stream=True):
        accumulated += chunk["choices"][0]["text"]
        yield accumulated


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 4 — chat sobre la receta activa  (LLM lazy)
# max_tokens reducido de 300 → 150
# ─────────────────────────────────────────────────────────────────────────────
def chat_about_recipe(message, history, recipe_state):
    if not message.strip():
        return history, ""
    if recipe_state:
        title   = recipe_state.get("recipe_title", "a recipe")
        ingr    = recipe_state.get(INGR_COL, "")
        sys_msg = (
            f"The user is asking about '{title}'.\nIngredients: {ingr}\n"
            "Answer only questions related to this recipe. Be brief and clear."
        )
    else:
        sys_msg = "You are a helpful cooking assistant. Be brief and clear."

    if not _llm_loaded():
        gr.Info("Cargando el modelo de lenguaje por primera vez (~25s) — por favor espera...")
    model  = get_llm()
    prompt = (
        f"<|system|>\n{sys_msg}\n</s>\n"
        f"<|user|>\n{message}\n</s>\n"
        "<|assistant|>\n"
    )
    reply = model(prompt, max_tokens=150, temperature=0.7, stream=False)["choices"][0]["text"].strip()
    return history + [[message, reply]], ""


# ─────────────────────────────────────────────────────────────────────────────
# CORRECCIÓN — panel HTML de ingredientes  (tarjeta con imagen O badge de texto coloreado)
# colores verde/naranja/rojo según nivel de confianza
# ─────────────────────────────────────────────────────────────────────────────
def _img_to_b64(path: str) -> str | None:
    """Codifica una imagen local como data-URI en base64 para incrustarla en HTML."""
    try:
        ext  = Path(path).suffix.lstrip(".").lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def _confidence_style(conf: float) -> tuple[str, str, str]:
    """
    Devuelve (color_borde, color_fondo_badge, emoji) según el nivel de confianza.
    Verde   = conf >= 0.60  → modelo confiado
    Naranja = 0.30 <= conf < 0.60 → confianza media
    Rojo    = conf < 0.30  → baja confianza, verificar manualmente
    """
    if conf >= 0.60:
        return "#4CAF50", "#e8f5e9", "✅"   # Verde
    elif conf >= 0.30:
        return "#FF9800", "#fff3e0", "⚠️"  # Naranja
    else:
        return "#f44336", "#ffebee", "❓"  # Rojo

def build_ingredient_html(top_ingr: list[tuple[str, float]], low_confidence: bool = False) -> str:
    """
    Panel HTML para ingredientes detectados.
    Sistema de semáforo (verde/naranja/rojo) según confianza.
    Advertencia si la confianza global de la imagen es baja.
    """
    warning_html = ""
    if low_confidence:
        # Advertencia visible cuando la imagen no es ideal
        warning_html = (
            '<div style="background:#fff3e0;border-left:4px solid #FF9800;'
            'padding:8px 12px;margin-bottom:8px;border-radius:4px;color:#7a5c00 !important">'
            '<p style="margin:0;font-size:13px;color:#7a5c00 !important;font-weight:bold">'
            '⚠️ <b>Imagen poco clara para el modelo.</b> '
            'Para mejores resultados, sube una foto de un único ingrediente crudo '
            'sobre fondo claro. Puedes corregir esto usando la búsqueda de texto abajo.'
            '</p>'
            '</div>'
        )

    cards: list[str] = []
    for i, (name, conf) in enumerate(top_ingr):
        pct = f"{conf * 100:.1f}%"
        border_color, bg_color, emoji = _confidence_style(conf)
        path = get_ingredient_image(name)
        src  = _img_to_b64(path) if path and Path(path).exists() else None

        # ── Barra de progreso visual ──────────────────────────────────────
        bar_html = (
            f'<div style="width:90px;height:6px;background:#e0e0e0;'
            f'border-radius:3px;margin:3px auto 1px auto">'
            f'  <div style="width:{conf * 100:.0f}%;height:6px;'
            f'  background:{border_color};border-radius:3px;'
            f'  transition:width 0.4s ease"></div>'
            f'</div>'
            f'<div style="font-size:10px;color:#888;text-align:center">{pct}</div>'
        )
        # ─────────────────────────────────────────────────────────────────

        if src:
            # Tarjeta con imagen real + borde de semáforo
            cards.append(
                f'<div style="text-align:center;margin:6px;width:110px">'
                f'<img src="{src}" style="width:100px;height:100px;'
                f'object-fit:cover;border-radius:10px;'
                f'border:3px solid {border_color}">'
                f'<div style="font-size:12px;margin-top:3px;color:#222 !important">{emoji} {name}</div>'
                f'{bar_html}'
                f'</div>'
            )
        else:
            # Badge de texto con color de semáforo de fondo
            cards.append(
                f'<div style="text-align:center;margin:6px;width:110px;'
                f'display:flex;flex-direction:column;align-items:center;justify-content:center">'
                f'<span style="background:{bg_color};border:2px solid {border_color};'
                f'padding:6px 12px;border-radius:14px;color:#222 !important;'
                f'font-size:13px;font-weight:500;display:inline-block">{emoji} {name}</span>'
                f'{bar_html}'
                f'</div>'
            )

    # Leyenda del semáforo
    legend_html = (
        '<div style="font-size:11px;color:#888;padding:4px 8px;margin-top:4px">'
        '✅ Alta confianza (&ge;60%) &nbsp;|&nbsp; '
        '⚠️ Media (30-60%) &nbsp;|&nbsp; '
        '❓ Baja (&lt;30%) — considera corregir mediante texto'
        '</div>'
    )

    return (
        warning_html
        + '<div style="display:flex;flex-wrap:wrap;gap:4px;'
        'padding:8px;min-height:60px;align-items:flex-start">'
        + "".join(cards)
        + "</div>"
        + legend_html
    )


# ─────────────────────────────────────────────────────────────────────────────
# GALERÍA DE PLATOS
# ─────────────────────────────────────────────────────────────────────────────
def build_dish_gallery(recipes: list[dict]) -> list[tuple[Image.Image, str]]:
    items: list[tuple[Image.Image, str]] = []
    for row in recipes:
        path = row.get("dish_image_path") or row.get("image_path") or ""
        if path and Path(path).exists():
            try:
                img = Image.open(path).convert("RGB").resize((300, 200))
            except Exception:
                img = _GREY
        else:
            img = _GREY
        items.append((img, row.get("recipe_title", "Receta")))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZACIÓN 3 — MANEJADOR DE BÚSQUEDA EN DOS FASES
#   Fase 1 (esta función, <3 s): CNN + FAISS → paneles A, B, debug
#   Fase 2 (clic en narrate_btn, ~30 s): narración LLM bajo demanda
#   advertencia si se proporcionan imagen y texto simultáneamente
#   mensajes de error más amigables para el usuario
# ─────────────────────────────────────────────────────────────────────────────
def find_recipes(image, text_query, dietary, speed, dish_type, progress=gr.Progress()):
    """
    Salidas (9):
        search_status, ingr_html, dish_gallery, recipe_df,
        recipe_detail, recipe_state, results_state, detected_names_state, pipeline_debug
    """
    t_total = time.perf_counter()

    # Validación con mensaje amigable
    if image is None and not (text_query or "").strip():
        raise gr.Error(
            "Por favor sube una foto de un ingrediente O escribe ingredientes "
            "separados por comas (p. ej., tomate, cebolla, ajo)."
        )

    # Advertir si el usuario proporciona TANTO imagen como texto
    if image is not None and (text_query or "").strip():
        gr.Warning(
            "Se han proporcionado imagen y texto. Se usará la imagen para la detección. "
            "Si prefieres buscar por texto, elimina primero la imagen."
        )

    debug: dict = {
        "modelos": {
            "cnn":      f"EfficientNet-B0 ({NUM_CLASSES} clases) | umbral de confianza: {CNN_CONFIDENCE_THRESHOLD*100:.0f}%",
            "embed":    f"{_EMBED_SHORT} (coseno)",
            "reranker": "Reranker MLP sobre puntuación FAISS + solapamiento + filtros + señal de imagen",
            "llm":      "TinyLlama-1.1B-Chat Q4_K_M (lazy — carga al Narrar)",
        },
        "consulta":             "",
        "puntuaciones_reranker":   {},
        "tiempos_ms":         {},
        "cnn_baja_confianza": False,
    }

    # ── Fase 1a: clasificar imagen o parsear texto ────────────────────────
    top_ingr: list[tuple[str, float]]
    low_confidence = False

    if image is not None:
        if not _cnn_loaded():
            gr.Info("Cargando el clasificador de ingredientes por primera vez (~15s)…")
        progress(0.15, desc="Clasificando ingredientes en la imagen…")
        t0 = time.perf_counter()

        top_ingr, low_confidence = classify_ingredients(image)
        debug["tiempos_ms"]["cnn_ms"]         = round((time.perf_counter() - t0) * 1000)
        debug["cnn_baja_confianza"]           = low_confidence
        names = [n for n, _ in top_ingr]
    else:
        names    = [s.strip() for s in text_query.split(",") if s.strip()]
        top_ingr = [(n, 1.0) for n in names[:10]]
        debug["tiempos_ms"]["cnn_ms"] = 0

    if not names:
        raise gr.Error(
            "No se pudieron extraer nombres de ingredientes de la entrada. "
            "Si usas una imagen, prueba con una foto de UN ÚNICO ingrediente crudo "
            "sobre fondo blanco. También puedes escribir los ingredientes "
            "directamente en el campo de texto."
        )

    # ── Fase 1b: embeddings + búsqueda FAISS ─────────────────────────────
    progress(0.45, desc="Buscando entre 64k recetas…")
    t0 = time.perf_counter()
    recipes, query_text, scores = retrieve_recipes(names, dietary, speed, dish_type)
    debug["tiempos_ms"]["faiss_ms"] = round((time.perf_counter() - t0) * 1000)
    debug["consulta"]  = query_text
    debug["puntuaciones_reranker"] = {
        r.get("recipe_title", f"receta_{i}"): round(float(s), 4)
        for i, (r, s) in enumerate(zip(recipes, scores))
    }

    if not recipes:
        raise gr.Error(
            "No se encontraron recetas con estos filtros. "
            "Prueba cambiando 'Preferencia dietética' y/o 'Velocidad de cocción' a 'any'."
        )

    # ── Fase 1c: construir paneles de resultados ─────────────────────────
    progress(0.75, desc="Construyendo resultados…")
    t0 = time.perf_counter()

    ingr_html_str = build_ingredient_html(top_ingr, low_confidence)
    dish_gal      = build_dish_gallery(recipes)

    display = [
        {
            "Título":    r.get("recipe_title", ""),
            "Cocina":   _stringify(r.get(CUISINE_COL, "")),
            "Tipo":     _stringify(r.get("course_list", r.get("course", r.get("category", "")))),
            "Velocidad": r.get("cook_speed", ""),
            "Dieta":    _stringify(r.get(DIETARY_COL, "")),
        }
        for r in recipes
    ]
    recipe_df_data = pd.DataFrame(display)
    recipe_detail  = build_recipe_detail_md(recipes[0], detected_names=names)

    debug["tiempos_ms"]["render_ms"]       = round((time.perf_counter() - t0) * 1000)
    debug["tiempos_ms"]["fase1_total_ms"] = round((time.perf_counter() - t_total) * 1000)

    elapsed_s = debug["tiempos_ms"]["fase1_total_ms"] / 1000
    faiss_s   = debug["tiempos_ms"]["faiss_ms"] / 1000

    conf_note = " ⚠️ Imagen con baja confianza — verifica los ingredientes detectados." if low_confidence else ""
    status = (
        f"**Fase 1 completa** — {len(recipes)} recetas encontradas "
        f"(FAISS {faiss_s:.2f}s · total {elapsed_s:.2f}s){conf_note} "
        f"| Selecciona una receta de la tabla y pulsa **Narrar** para la narración con IA."
    )

    progress(1.0, desc="Fase 1 lista ✅")
    return (
        status,
        ingr_html_str,
        dish_gal,
        recipe_df_data,
        recipe_detail,
        recipes[0],
        recipes,
        names,
        debug,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MANEJADORES DE SELECCIÓN
# ─────────────────────────────────────────────────────────────────────────────
def select_from_df(evt: gr.SelectData, results: list[dict], detected_names: list[str]) -> tuple:
    if not results:
        return None, "", ""
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else 0
    row     = results[min(row_idx, len(results) - 1)]
    return row, f"Seleccionada: **{row.get('recipe_title', 'Receta')}**", build_recipe_detail_md(row, detected_names)



def select_from_gallery(evt: gr.SelectData, results: list[dict], detected_names: list[str]) -> tuple:
    if not results:
        return None, "", ""
    idx = min(int(evt.index), len(results) - 1)
    row = results[idx]
    return row, f"Seleccionada: **{row.get('recipe_title', 'Receta')}**", build_recipe_detail_md(row, detected_names)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAMA DEL PIPELINE — Pestaña 2
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_MD = """\
## Cómo funciona el pipeline
```
  ┌──────────────────────────────────────────────────────────┐
  │                    ENTRADA DEL USUARIO                   │
  │   Foto  ──O──  Consulta de texto  ──O──  Lista ingredientes│
  └──────────────┬───────────────────────────────────────────┘
                 │
        ┌────────▼──────────────────────────────────────────┐
        │  EfficientNet-B0  (carga lazy — solo con foto)    │
        │  imagen → top-10 predicciones de ingredientes     │
        │  Fruits-360 + Dataset de Ingredientes ~150 clases │
        └────────┬──────────────────────────────────────────┘
                 │  lista de ingredientes
        ┌────────▼──────────────────────────────────────────┐
        │  multilingual-MiniLM-L12-v2  ·  384-dim           │
        │  "ingredientes: tomate, cebolla, …" → vec float32 │
        └────────┬──────────────────────────────────────────┘
                 │
        ┌────────▼──────────────────────────────────────────┐
        │  FAISS IndexFlatIP  ·  64k recetas                │
        │  top-50 → filtros: dieta + velocidad + tipo       │
        └────────┬──────────────────────────────────────────┘
                 │
        ┌────────▼──────────────────────────────────────────┐
        │  Reranker MLP                                     │
        │  puntuación FAISS + solapamiento + filtros        │
        └────────┬──────────────────────────────────────────┘
                 │        ← FASE 1 COMPLETA  (< 3 s)
        ┌────────▼──────────────────────────────────────────┐
        │  Top-5 tarjetas de receta                         │
        │  (título · cocina · etiquetas dietéticas · speed) │
        └────────┬──────────────────────────────────────────┘
                 │  usuario hace clic en "Narrar"  ← FASE 2
        ┌────────▼──────────────────────────────────────────┐
        │  TinyLlama-1.1B-Chat  Q4_K_M  (lazy — ~25 s)     │
        │  Genera narración amigable  ·  ~30 s en CPU       │
        └────────┬──────────────────────────────────────────┘
                 │  usuario escribe en el chat
        ┌────────▼──────────────────────────────────────────┐
        │  Modo chat: receta inyectada como contexto        │
        └───────────────────────────────────────────────────┘
```
### Estrategia de carga lazy
| Componente | Se carga en | Tiempo aprox. |
|---|---|---|
| Índice FAISS + dataframe | Arranque de la app | ~2 s |
| SentenceTransformer | Arranque de la app | ~3 s |
| EfficientNet-B0 | Primera foto subida | ~10 s (una vez) |
| TinyLlama GGUF | Primer "Narrar" o Chat | ~25 s (una vez) |
Tras la primera carga, cada modelo queda cacheado en memoria para todas las peticiones siguientes.
### Latencia por etapa (CPU tier gratuito, post-carga)
| Etapa | Tiempo |
|---|---|
| Clasificación CNN | < 1 s |
| Embedding MiniLM | < 0.5 s |
| FAISS top-50 + filtros + reranker | < 1.5 s |
| **Fase 1 total** | **< 3 s** |
| Narración LLM (512 tokens) | 25–40 s |
| Respuesta de chat (300 tokens) | 15–25 s |
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ DE USUARIO
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Recomendador de Recetas (optimizado)", theme=gr.themes.Soft()) as demo:

    # Cabecera con instrucciones paso a paso
    gr.Markdown("# 🍳 Recomendador de Recetas — Demo Multimodal con IA")
    gr.Markdown("""
**Cómo usar la aplicación:**
1. 📸 **Sube una foto** de un **único** ingrediente crudo sobre fondo claro — **O** —
   ✏️ **Escribe** ingredientes separados por comas (p. ej., *tomate, cebolla, ajo*)
2. Ajusta los filtros opcionales (dieta, velocidad, tipo de plato)
3. 🔍 Haz clic en **Buscar Recetas**
4. Selecciona una receta de la galería o la tabla
5. ✨ Haz clic en **Narrar Receta** para la narración con IA (~15-25s en CPU)

> 💡 **Consejo de imagen:** Para mejores resultados, usa fotos de catálogo (fondo blanco,
> ingrediente crudo individual, bien iluminado). Evita envases, platos cocinados o fotos con mucho ruido.
    """)

    # Estado compartido
    recipe_state  = gr.State(None)
    results_state = gr.State([])
    detected_names_state = gr.State([])

    with gr.Tabs():

        # ── PESTAÑA 1 — Buscar Recetas ─────────────────────────────────────────
        with gr.Tab("Buscar Recetas"):
            with gr.Row(equal_height=False):

                # ── COLUMNA IZQUIERDA: entradas ────────────────────────────────
                with gr.Column(scale=1, min_width=300):
                    img_input = gr.Image(
                        label="📸 Foto de UN ingrediente crudo (fondo claro, sin envase)",
                        type="pil",
                        height=220,
                    )
                    gr.Markdown(
                        "<small>💡 Ejemplos ideales: una zanahoria, un tomate, "
                        "una manzana — sobre una mesa o fondo blanco.</small>"
                    )
                    text_input = gr.Textbox(
                        label="✏️ O escribe ingredientes / antojos",
                        placeholder="tomate, cebolla, ajo, albahaca  o  pasta vegana rápida...",
                        lines=2,
                    )
                    dietary_dd = gr.Dropdown(
                        label="Preferencia Dietética",
                        choices=DIETARY_CHOICES,
                        value="any",
                    )
                    speed_dd = gr.Dropdown(
                        label="Velocidad de Cocción",
                        choices=SPEED_CHOICES,
                        value="any",
                    )
                    dish_type_dd = gr.Dropdown(
                        label="Tipo de Plato",
                        choices=DISH_TYPE_CHOICES,
                        value="any",
                    )
                    find_btn = gr.Button("Buscar Recetas 🔍", variant="primary", size="lg")

                    gr.Examples(
                        examples=[
                            ["tomato, mozzarella, basil",    "vegetarian", "any",    "any"],
                            ["chicken, garlic, lemon",       "any",        "medium", "any"],
                            ["oats, banana, honey",          "vegan",      "any",    "any"],
                            ["pasta, eggs, bacon, parmesan", "any",        "medium", "any"],
                            ["black beans, corn, avocado",   "vegan",      "any",    "any"],
                        ],
                        inputs=[text_input, dietary_dd, speed_dd, dish_type_dd],
                        label="Ejemplos Rápidos",
                        examples_per_page=5,
                        cache_examples=False,
                    )

                # ── COLUMNA DERECHA: resultados ────────────────────────────────
                with gr.Column(scale=2):

                    search_status = gr.Markdown(
                        "Sube una foto **o** escribe ingredientes y haz clic en **Buscar Recetas**."
                    )

                    with gr.Accordion("Ingredientes Detectados", open=True):
                        ingr_html = gr.HTML(
                            value='<p style="color:#aaa;padding:8px;font-size:13px">—</p>'
                        )

                    with gr.Accordion("Mejores Recetas", open=True):
                        dish_gallery = gr.Gallery(
                            label="Imágenes de platos — clic para seleccionar",
                            columns=5,
                            height=190,
                            object_fit="cover",
                            show_label=True,
                            allow_preview=False,
                        )
                        recipe_df = gr.Dataframe(
                            headers=["Título", "Cocina", "Tipo", "Velocidad", "Dieta"],
                            interactive=False,
                            wrap=True,
                            row_count=(5, "fixed"),
                        )

                    with gr.Accordion("Ingredientes y Procedimiento de la Receta", open=True):
                        recipe_detail_md = gr.Markdown(
                            "Selecciona una receta para ver ingredientes y procedimiento."
                        )
                        narrate_btn = gr.Button(
                            "✨ Narrar receta seleccionada (~15-25s en CPU) ▶",
                            variant="secondary",
                        )
                        narration_box = gr.Textbox(
                            lines=8,
                            interactive=False,
                            placeholder=(
                                "⏳ Selecciona una receta arriba y haz clic en 'Narrar'.\n\n"
                                "La primera narración tarda ~25s ya que carga el modelo de lenguaje. "
                                "Las siguientes serán más rápidas (~15s). "
                                "Mientras esperas, puedes leer el procedimiento de la receta arriba."
                            ),
                            show_copy_button=True,
                            label="Narración Generada por IA",
                        )

                    with gr.Accordion(
                        "Chat sobre esta receta [TinyLlama · ~10-15s por respuesta en CPU]",
                        open=True,
                    ):
                        chatbot = gr.Chatbot(height=300, bubble_full_width=False)
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Pregúntame lo que quieras sobre esta receta...",
                                show_label=False,
                                scale=5,
                            )
                            chat_btn = gr.Button("Enviar ↩", scale=1, variant="primary")
                        clear_btn = gr.Button("Limpiar chat", size="sm")

                    with gr.Accordion("Transparencia del Pipeline", open=False):
                        gr.Markdown(
                            "_Consulta de embedding, puntuaciones del reranker MLP, "
                            "tiempos por etapa y flag de baja confianza CNN._"
                        )
                        pipeline_debug_json = gr.JSON(label="", value={})

        # ── PESTAÑA 2 — Cómo funciona ──────────────────────────────────────────
        with gr.Tab("Cómo funciona"):
            gr.Markdown(PIPELINE_MD)

# ── MANEJADORES DE EVENTOS ────────────────────────────────────────────────────

    find_btn.click(
        fn=find_recipes,
        inputs=[img_input, text_input, dietary_dd, speed_dd, dish_type_dd],
        outputs=[
            search_status,
            ingr_html,
            dish_gallery,
            recipe_df,
            recipe_detail_md,
            recipe_state,
            results_state,
            detected_names_state,
            pipeline_debug_json,
        ],
    )

    recipe_df.select(
        fn=select_from_df,
        inputs=[results_state, detected_names_state],
        outputs=[recipe_state, search_status, recipe_detail_md],
    )

    dish_gallery.select(
        fn=select_from_gallery,
        inputs=[results_state, detected_names_state],
        outputs=[recipe_state, search_status, recipe_detail_md],
    )

    narrate_btn.click(
        fn=generate_recipe,
        inputs=[recipe_state],
        outputs=[narration_box],
    )

    chat_btn.click(
        fn=chat_about_recipe,
        inputs=[chat_input, chatbot, recipe_state],
        outputs=[chatbot, chat_input],
    )
    chat_input.submit(
        fn=chat_about_recipe,
        inputs=[chat_input, chatbot, recipe_state],
        outputs=[chatbot, chat_input],
    )

    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

# ─────────────────────────────────────────────────────────────────────────────
# LANZAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
demo.queue(max_size=3)
demo.launch()
