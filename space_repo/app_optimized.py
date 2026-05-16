# Recomendador de Recetas — Gradio app para Hugging Face Space

import base64
import functools
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

import faiss
import gradio as gr

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

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from PIL import Image
from rapidfuzz import process as rfprocess
from sentence_transformers import SentenceTransformer

HF_USERNAME   = os.environ.get("HF_USERNAME",   "Grupo-Parquet-ITESO")
HF_SPACE_NAME = os.environ.get("HF_SPACE_NAME", "ProyectoFinal_recetas")
CNN_REPO      = os.environ.get("CNN_REPO",  f"{HF_USERNAME}/recipe-ingredient-classifier")
LLM_REPO      = os.environ.get("LLM_REPO",  "Grupo-Parquet-ITESO/recipe-llm-gguf")
LLM_GGUF_FILE = os.environ.get("LLM_GGUF_FILE", "tinyllama-recipes-q4.gguf")
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_SHORT  = "multilingual-MiniLM-L12-v2  ·  384-dim"

DIETARY_CHOICES = ["any", "vegetarian", "vegan", "gluten-free", "dairy-free"]
SPEED_CHOICES   = ["any", "fast", "medium", "slow"]

CLIP_MODEL_NAME    = "clip-ViT-B-32"
# Escenas multi-ingrediente tienen similitudes bajas (~0.14-0.18) — threshold bajo intencional.
CLIP_MIN_SIMILARITY = 0.15
CLIP_MIN_RESULTS    = 3   # devolver al menos este número aunque estén bajo el threshold
CLIP_MAX_RESULTS    = 7

# Paleta pastel para badges de ingredientes no encontrados en el catálogo
_BADGE_COLORS = [
    "#FFB3B3", "#B3D9FF", "#B3FFB3", "#FFD9B3",
    "#E8B3FF", "#B3FFE8", "#FFE8B3", "#D9B3FF",
]
_GREY = Image.new("RGB", (200, 200), color=(210, 210, 210))

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

print("Artefactos de arranque listos. CLIP y LLM se cargarán en el primer uso.")

_clip_lock = threading.Lock()

# Embeddings de texto pre-computados para todos los ingredientes (se llenan al primer uso de CLIP)
_ingredient_names_cache: list[str] = []
_ingredient_text_embs_cache: "np.ndarray | None" = None

@functools.lru_cache(maxsize=1)
def _load_clip_cached() -> "SentenceTransformer":
    return SentenceTransformer(CLIP_MODEL_NAME)

_llm_lock = threading.Lock()
_llm_instance: "Llama | None" = None

def _get_llm() -> "Llama":
    global _llm_instance
    with _llm_lock:
        if _llm_instance is None:
            token = os.environ.get("HF_TOKEN", "")
            gguf_path = hf_hub_download(
                repo_id=LLM_REPO,
                filename=LLM_GGUF_FILE,
                token=token or None,
            )
            _llm_instance = Llama(
                model_path=gguf_path,
                n_ctx=1024,
                n_threads=4,
                verbose=False,
            )
        return _llm_instance

def _llm_generate(prompt: str, max_new_tokens: int = 200) -> str:
    llm = _get_llm()
    out = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.7,
        stop=["</s>", "<|user|>", "<|system|>"],
        echo=False,
    )
    return out["choices"][0]["text"].strip()

def get_clip() -> "SentenceTransformer":
    with _clip_lock:
        return _load_clip_cached()

def _clip_loaded() -> bool:
    return _load_clip_cached.cache_info().currsize > 0

def _get_ingredient_text_embeddings() -> tuple[list[str], "np.ndarray"]:
    global _ingredient_names_cache, _ingredient_text_embs_cache
    if _ingredient_text_embs_cache is None:
        model  = get_clip()
        seen, unique = set(), []
        for name in class_labels.values():
            key = name.lower()
            if key not in seen:
                seen.add(key)
                unique.append(name)
        _ingredient_names_cache = unique
        # Prompt ensembling: promediar 3 templates mejora detección en escenas multi-ingrediente
        templates = [
            lambda n: f"a photo of {n}",
            lambda n: f"{n}",
            lambda n: f"fresh {n}",
        ]
        embs_per_template = [
            model.encode([t(n) for n in unique], normalize_embeddings=True,
                         batch_size=64, show_progress_bar=False)
            for t in templates
        ]
        avg = np.mean(embs_per_template, axis=0)
        norms = np.linalg.norm(avg, axis=1, keepdims=True)
        _ingredient_text_embs_cache = (avg / norms).astype("float32")
    return _ingredient_names_cache, _ingredient_text_embs_cache

def detect_ingredients_clip(image: Image.Image) -> tuple[list[tuple[str, float]], bool]:

    model = get_clip()
    img_emb = model.encode(image.convert("RGB"), normalize_embeddings=True)

    names, text_embs = _get_ingredient_text_embeddings()
    sims = (img_emb @ text_embs.T).astype(float)

    ranked = sorted(
        ((names[i], float(sims[i])) for i in range(len(names))),
        key=lambda x: x[1], reverse=True,
    )

    detected = [(n, s) for n, s in ranked if s >= CLIP_MIN_SIMILARITY]

    # Garantizar mínimo CLIP_MIN_RESULTS aunque estén bajo el threshold
    low_confidence = len(detected) < CLIP_MIN_RESULTS
    if low_confidence:
        detected = ranked[:CLIP_MIN_RESULTS]

    return detected[:CLIP_MAX_RESULTS], low_confidence

def get_ingredient_image(name: str) -> str | None:
    hit = rfprocess.extractOne(name.lower(), _catalog_keys)
    if hit and hit[1] >= 72:
        return ingredient_catalog[hit[0]]
    return None

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
    if not recipe_row:
        yield "Selecciona una receta de la tabla de arriba y haz clic en 'Narrar'."
        return
    try:
        yield _llm_generate(_narration_prompt(recipe_row), max_new_tokens=200)
    except Exception as e:
        yield f"Error al generar la narración: {e}"

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

    prompt = (
        f"<|system|>\n{sys_msg}\n</s>\n"
        f"<|user|>\n{message}\n</s>\n"
        "<|assistant|>\n"
    )
    try:
        reply = _llm_generate(prompt, max_new_tokens=150).strip()
    except Exception as e:
        reply = f"Error: {e}"
    return history + [[message, reply]], ""

def _img_to_b64(path: str) -> str | None:
    try:
        ext  = Path(path).suffix.lstrip(".").lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def _confidence_style(conf: float) -> tuple[str, str, str]:
    if conf >= 0.60:
        return "#4CAF50", "#e8f5e9", "ok"   # Verde
    elif conf >= 0.30:
        return "#FF9800", "#fff3e0", "!"  # Naranja
    else:
        return "#f44336", "#ffebee", "?"  # Rojo

def build_ingredient_html(top_ingr: list[tuple[str, float]], low_confidence: bool = False) -> str:
    warning_html = ""
    if low_confidence:
        warning_html = (
            '<div style="background:#fff3e0;border-left:4px solid #FF9800;'
            'padding:8px 12px;margin-bottom:8px;border-radius:4px;color:#7a5c00 !important">'
            '<p style="margin:0;font-size:13px;color:#7a5c00 !important;font-weight:bold">'
            '<b>Imagen poco clara para el modelo.</b> '
            'Para mejores resultados, sube una foto con varios ingredientes visibles. '
            'También puedes escribirlos directamente en el campo de texto.'
            '</p>'
            '</div>'
        )

    cards: list[str] = []
    for i, (name, conf) in enumerate(top_ingr):
        pct = f"{conf * 100:.1f}%"
        border_color, bg_color, emoji = _confidence_style(conf)
        path = get_ingredient_image(name)
        src  = _img_to_b64(path) if path and Path(path).exists() else None

        bar_html = (
            f'<div style="width:90px;height:6px;background:#e0e0e0;'
            f'border-radius:3px;margin:3px auto 1px auto">'
            f'  <div style="width:{conf * 100:.0f}%;height:6px;'
            f'  background:{border_color};border-radius:3px;'
            f'  transition:width 0.4s ease"></div>'
            f'</div>'
            f'<div style="font-size:10px;color:#888;text-align:center">{pct}</div>'
        )

        if src:
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
            cards.append(
                f'<div style="text-align:center;margin:6px;width:110px;'
                f'display:flex;flex-direction:column;align-items:center;justify-content:center">'
                f'<span style="background:{bg_color};border:2px solid {border_color};'
                f'padding:6px 12px;border-radius:14px;color:#222 !important;'
                f'font-size:13px;font-weight:500;display:inline-block">{emoji} {name}</span>'
                f'{bar_html}'
                f'</div>'
            )

    legend_html = (
        '<div style="font-size:11px;color:#888;padding:4px 8px;margin-top:4px">'
        'Alta confianza (&ge;60%) &nbsp;|&nbsp; '
        'Media (30-60%) &nbsp;|&nbsp; '
        'Baja (&lt;30%) — considera corregir mediante texto'
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

def find_recipes(image, text_query, dietary, speed, dish_type, progress=gr.Progress()):
    """
    Salidas (9):
        search_status, ingr_html, dish_gallery, recipe_df,
        recipe_detail, recipe_state, results_state, detected_names_state, pipeline_debug
    """
    t_total = time.perf_counter()

    if image is None and not (text_query or "").strip():
        raise gr.Error(
            "Por favor sube una foto de un ingrediente O escribe ingredientes "
            "separados por comas (p. ej., tomate, cebolla, ajo)."
        )

    if image is not None and (text_query or "").strip():
        gr.Warning(
            "Se han proporcionado imagen y texto. Se usará la imagen para la detección. "
            "Si prefieres buscar por texto, elimina primero la imagen."
        )

    debug: dict = {
        "modelos": {
            "clip":     f"clip-ViT-B-32 zero-shot | umbral similitud: {CLIP_MIN_SIMILARITY} | max ingredientes: {CLIP_MAX_RESULTS}",
            "embed":    f"{_EMBED_SHORT} (coseno)",
            "reranker": "Reranker MLP sobre puntuación FAISS + solapamiento + filtros + señal de imagen",
            "llm":      f"HF Inference API → {LLM_REPO}  (~2-5s)",
        },
        "consulta":              "",
        "puntuaciones_reranker": {},
        "tiempos_ms":            {},
        "clip_sin_deteccion":    False,
    }

    top_ingr: list[tuple[str, float]]
    low_confidence = False

    if image is not None:
        if not _clip_loaded():
            gr.Info("Buscando por favor espera…")
        progress(0.15, desc="Detectando ingredientes en la imagen…")
        t0 = time.perf_counter()

        top_ingr, low_confidence = detect_ingredients_clip(image)
        debug["tiempos_ms"]["clip_ms"]       = round((time.perf_counter() - t0) * 1000)
        debug["clip_sin_deteccion"]          = low_confidence
        names = [n for n, _ in top_ingr]
    else:
        names    = [s.strip() for s in text_query.split(",") if s.strip()]
        top_ingr = [(n, 1.0) for n in names[:10]]
        debug["tiempos_ms"]["clip_ms"] = 0

    if not names:
        raise gr.Error(
            "No se pudieron extraer ingredientes de la imagen. "
            "Prueba con una foto más clara o escribe los ingredientes "
            "directamente en el campo de texto."
        )

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

    conf_note = " Imagen con baja confianza — verifica los ingredientes detectados." if low_confidence else ""
    status = (
        f"**Fase 1 completa** — {len(recipes)} recetas encontradas "
        f"(FAISS {faiss_s:.2f}s · total {elapsed_s:.2f}s){conf_note} "
        f"| Selecciona una receta de la tabla y pulsa **Narrar** para la narración con IA."
    )

    progress(1.0, desc="Fase 1 lista")
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

PIPELINE_MD = """\
## Cómo funciona el pipeline
```
  ┌──────────────────────────────────────────────────────────┐
  │                    ENTRADA DEL USUARIO                   │
  │   Foto  ──O──  Consulta de texto  ──O──  Lista ingredientes│
  └──────────────┬───────────────────────────────────────────┘
                 │
        ┌────────▼──────────────────────────────────────────┐
        │  CLIP  clip-ViT-B-32  (carga lazy — solo foto)   │
        │  imagen → similitud vs 394 ingredientes           │
        │  zero-shot multi-label · prompt ensembling x3     │
        │  detecta VARIOS ingredientes en una sola foto     │
        └────────┬──────────────────────────────────────────┘
                 │  lista de ingredientes detectados
        ┌────────▼──────────────────────────────────────────┐
        │  multilingual-MiniLM-L12-v2  ·  384-dim           │
        │  "ingredientes: tomate, cebolla, …" → vec float32 │
        └────────┬──────────────────────────────────────────┘
                 │
        ┌────────▼──────────────────────────────────────────┐
        │  FAISS IndexFlatIP  ·  62k recetas                │
        │  top-50 → filtros: dieta + velocidad + tipo       │
        └────────┬──────────────────────────────────────────┘
                 │
        ┌────────▼──────────────────────────────────────────┐
        │  Reranker MLP                                     │
        │  puntuación FAISS + solapamiento + filtros        │
        └────────┬──────────────────────────────────────────┘
                 │        ← FASE 1 COMPLETA  (< 3 s)
        ┌────────▼──────────────────────────────────────────┐
        │  Top-5 recetas                                    │
        │  (título · cocina · tipo · velocidad · dieta)    │
        └────────┬──────────────────────────────────────────┘
                 │  usuario hace clic en "Narrar"  ← FASE 2
        ┌────────▼──────────────────────────────────────────┐
        │  TinyLlama-1.1B-Chat  (HF Inference API)          │
        │  GPU compartido de HF  ·  ~3-5 s                  │
        └────────┬──────────────────────────────────────────┘
                 │  usuario escribe en el chat
        ┌────────▼──────────────────────────────────────────┐
        │  Chat: receta activa inyectada como contexto      │
        └───────────────────────────────────────────────────┘
```
### Componentes y carga
| Componente | Se carga en | Tiempo |
|---|---|---|
| Índice FAISS + dataframe (62k recetas) | Arranque de la app | ~2 s |
| SentenceTransformer multilingual-MiniLM | Arranque de la app | ~3 s |
| CLIP clip-ViT-B-32 | Primera foto subida | ~20 s (una vez) |
| TinyLlama-1.1B-Chat (HF Inference API) | Por petición vía GPU de HF | ~3-5 s |

CLIP y el índice FAISS quedan cacheados en memoria tras la primera carga.

### Latencia por etapa (post-carga)
| Etapa | Tiempo |
|---|---|
| Detección CLIP multi-ingrediente | < 1 s |
| Embedding MiniLM | < 0.5 s |
| FAISS top-50 + filtros + reranker | < 1.5 s |
| **Fase 1 total** | **< 3 s** |
| Narración TinyLlama (HF Inference API) | ~3-5 s |
| Respuesta de chat | ~3-5 s |

### Vocabulario de ingredientes
394 ingredientes cubiertos, incluyendo frutas, verduras, carnes, lácteos,
especias, chiles secos mexicanos, hierbas, frutos secos y condimentos.
"""

with gr.Blocks(title="Recomendador de Recetas (optimizado)", theme=gr.themes.Soft()) as demo:

    gr.Markdown("Recomendador de Recetas")
    gr.Markdown("""
**Cómo usar la aplicación:**
1. **Sube una foto** con uno o varios ingredientes (tabla de cortar, mercado, nevera) — **O** —
   **Escribe** ingredientes separados por comas (p. ej., *tomate, cebolla, ajo*)
2. Ajusta los filtros opcionales (dieta, velocidad, tipo de plato)
3. Haz clic en **Buscar Recetas**
4. Selecciona una receta de la galería o la tabla
5. Haz clic en **Narrar Receta** para la narración con IA (~3-5s)

> **Consejo:** CLIP detecta múltiples ingredientes en una sola foto.
> Funciona con fotos reales de celular — tabla de cortar, bolsa del super, nevera abierta.
    """)

    recipe_state  = gr.State(None)
    results_state = gr.State([])
    detected_names_state = gr.State([])

    with gr.Tabs():

        with gr.Tab("Buscar Recetas"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=1, min_width=300):
                    img_input = gr.Image(
                        label="Foto con uno o varios ingredientes (tabla de cortar, bolsa del súper, nevera...)",
                        type="pil",
                        height=220,
                    )
                    gr.Markdown(
                        "<small>CLIP detecta múltiples ingredientes en una sola foto. "
                        "Funciona con fotos reales de celular.</small>"
                    )
                    text_input = gr.Textbox(
                        label="O escribe ingredientes / antojos",
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
                    find_btn = gr.Button("Buscar Recetas", variant="primary", size="lg")

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
                            "Narrar receta seleccionada (~3-5s)",
                            variant="secondary",
                        )
                        narration_box = gr.Textbox(
                            lines=8,
                            interactive=False,
                            placeholder=(
                                "⏳ Selecciona una receta arriba y haz clic en 'Narrar'.\n\n"
                                "La narración usa HF Inference API (~3-5s)."
                            ),
                            show_copy_button=True,
                            label="Narración Generada por IA",
                        )

                    with gr.Accordion(
                        "Chat sobre esta receta [~3-5s por respuesta vía HF Inference API]",
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

        with gr.Tab("Cómo funciona"):
            gr.Markdown(PIPELINE_MD)

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

demo.queue(max_size=3)
demo.launch()
