import base64
import functools
import json
import os
import threading
import time
from pathlib import Path

import faiss
import gradio as gr

# gradio_client 0.6.x raises TypeError on bool JSON-Schema values
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

# Starlette >=1.0 changed TemplateResponse(name, ctx) → (request, name)
import starlette.templating as _st

_orig_TemplateResponse = _st.Jinja2Templates.TemplateResponse

def _compat_TemplateResponse(self, *args, **kwargs):
    if args and isinstance(args[0], str):
        name    = args[0]
        context = args[1] if len(args) > 1 else kwargs.pop("context", {})
        request = context.get("request")
        return _orig_TemplateResponse(self, request, name, context=context, **kwargs)
    return _orig_TemplateResponse(self, *args, **kwargs)

_st.Jinja2Templates.TemplateResponse = _compat_TemplateResponse

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
    print("llama-cpp-python not available — LLM disabled")
from PIL import Image
from rapidfuzz import process as rfprocess
from sentence_transformers import SentenceTransformer

HF_USERNAME   = os.environ.get("HF_USERNAME",   "ramonsj11")
HF_SPACE_NAME = os.environ.get("HF_SPACE_NAME", "ProyectoFinal_recetas")
CNN_REPO      = f"{HF_USERNAME}/recipe-ingredient-classifier"
LLM_REPO      = f"{HF_USERNAME}/recipe-llm-gguf"
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_SHORT  = "multilingual-MiniLM-L12-v2  ·  384-dim"

DIETARY_CHOICES = ["any", "vegetarian", "vegan", "gluten-free", "dairy-free"]
SPEED_CHOICES   = ["any", "fast", "medium", "slow"]

_BADGE_COLORS = [
    "#FFB3B3", "#B3D9FF", "#B3FFB3", "#FFD9B3",
    "#E8B3FF", "#B3FFE8", "#FFE8B3", "#D9B3FF",
]
_GREY = Image.new("RGB", (200, 200), color=(210, 210, 210))

print("Loading FAISS index…")
faiss_index = faiss.read_index("recipe_faiss.index")

print("Loading dataframe…")
df = pd.read_parquet("df_final_embeddings.parquet").reset_index(drop=True)

with open("ingredient_catalog.json") as _f:
    ingredient_catalog: dict[str, str] = json.load(_f)

try:
    with open("class_labels.json") as _f:
        class_labels: dict[str, str] = json.load(_f)
    print(f"  class_labels.json: {len(class_labels)} classes")
except FileNotFoundError:
    class_labels = {}
    print("  class_labels.json not found — CNN disabled")

NUM_CLASSES   = len(class_labels)
_catalog_keys = list(ingredient_catalog.keys())

if "ingredient_text_es" in df.columns:
    INGR_COL = "ingredient_text_es"
elif "ingredient_text" in df.columns:
    INGR_COL = "ingredient_text"
else:
    INGR_COL = "ingredients_text_processed"
DIETARY_COL    = "dietary_profile" if "dietary_profile" in df.columns else "dietary_profile_updated"
CUISINE_COL    = "cuisine_list"    if "cuisine_list"    in df.columns else "cuisine"
DISH_TYPE_COLS = [c for c in ("course_list", "course", "category", "subcategory") if c in df.columns]

print(f"  {len(df):,} recipes | ingr_col={INGR_COL} | dietary_col={DIETARY_COL}")

print("Loading SentenceTransformer…")
embedding_model = SentenceTransformer(EMBED_MODEL)

print("Ready.")

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
    if NUM_CLASSES == 0:
        raise RuntimeError("class_labels.json not found — CNN unavailable")
    weights_path = hf_hub_download(repo_id=CNN_REPO, filename="efficientnet_ingredients.pth")
    mdl = models.efficientnet_b0(weights=None)
    mdl.classifier[1] = torch.nn.Linear(1280, NUM_CLASSES)
    mdl.load_state_dict(torch.load(weights_path, map_location="cpu"))
    mdl.eval()
    return mdl


@functools.lru_cache(maxsize=1)
def _load_llm_cached() -> "Llama":
    if not _LLAMA_AVAILABLE:
        raise RuntimeError("llama-cpp-python not installed — LLM unavailable")
    gguf_path = hf_hub_download(repo_id=LLM_REPO, filename="tinyllama-recipes-q4.gguf")
    return Llama(model_path=gguf_path, n_ctx=2048, n_threads=4, verbose=False)


def get_cnn() -> torch.nn.Module:
    with _cnn_lock:
        return _load_cnn_cached()


def get_llm() -> Llama:
    with _llm_lock:
        return _load_llm_cached()


def _cnn_loaded() -> bool:
    return _load_cnn_cached.cache_info().currsize > 0


def _llm_loaded() -> bool:
    return _load_llm_cached.cache_info().currsize > 0


def classify_ingredients(image: Image.Image) -> list[tuple[str, float]]:
    model  = get_cnn()
    tensor = _cnn_tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top10 = torch.topk(probs, 10)
    return [
        (class_labels.get(str(i.item()), f"class_{i.item()}"), s.item())
        for i, s in zip(top10.indices, top10.values)
    ]


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
            first: torch.nn.Linear  = self.net[0]  # type: ignore[assignment]
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
    return top.to_dict(orient="records"), query, scores


def _narration_prompt(row: dict) -> str:
    title   = row.get("recipe_title", "Unknown recipe")
    ingr    = row.get(INGR_COL) or row.get("ingredients_text_processed", "")
    ingr    = ingr.replace(" ", ", ") if " " in ingr and "," not in ingr else ingr
    dirs    = row.get("directions_text", "")[:800]
    dietary = _stringify(row.get(DIETARY_COL) or row.get("dietary_profile_updated", ""))
    return (
        "<|system|>\n"
        "You are a helpful cooking assistant. Narrate recipes clearly and engagingly.\n</s>\n"
        "<|user|>\n"
        "Please narrate this recipe in a friendly way:\n"
        f"Title: {title}\nIngredients: {ingr}\nInstructions: {dirs}\nDietary: {dietary}\n</s>\n"
        "<|assistant|>\n"
    )


def build_recipe_detail_md(row: dict | None) -> str:
    if not row:
        return "Select a recipe to see ingredients and procedure."
    title       = row.get("recipe_title", "Recipe")
    ingredients = row.get(INGR_COL) or row.get("ingredients_text_processed", "")
    ingredients = ingredients.replace(" ", ", ") if " " in ingredients and "," not in ingredients else ingredients
    directions  = row.get("directions_text", "") or row.get("directions", "")
    cuisine     = _stringify(row.get(CUISINE_COL, ""))
    dietary     = _stringify(row.get(DIETARY_COL, ""))
    speed       = row.get("cook_speed", "")
    meta        = " · ".join(str(x) for x in [cuisine, dietary, speed] if str(x).strip())
    return (
        f"### {title}\n\n"
        f"{meta}\n\n"
        f"**Ingredients**\n\n{ingredients or 'Not available'}\n\n"
        f"**Procedure**\n\n{directions or 'Not available'}"
    )


def generate_recipe(recipe_row: dict | None):
    if not recipe_row:
        yield "Select a recipe from the table above, then click 'Narrate'."
        return
    if not _llm_loaded():
        gr.Info("Loading language model for the first time (~25 s) — please wait…")
    model       = get_llm()
    accumulated = ""
    for chunk in model(_narration_prompt(recipe_row), max_tokens=512, temperature=0.7, stream=True):
        accumulated += chunk["choices"][0]["text"]
        yield accumulated


def chat_about_recipe(
    message: str,
    history: list[list[str | None]],
    recipe_state: dict | None,
) -> tuple[list, str]:
    if not message.strip():
        return history, ""
    if recipe_state:
        title   = recipe_state.get("recipe_title", "a recipe")
        ingr    = recipe_state.get(INGR_COL, "")
        sys_msg = (
            f"The user is asking about '{title}'.\nIngredients: {ingr}\n"
            "Answer only questions related to this recipe."
        )
    else:
        sys_msg = "You are a helpful cooking assistant."

    if not _llm_loaded():
        gr.Info("Loading language model for the first time (~25 s) — please wait…")
    model  = get_llm()
    prompt = (
        f"<|system|>\n{sys_msg}\n</s>\n"
        f"<|user|>\n{message}\n</s>\n"
        "<|assistant|>\n"
    )
    reply = model(prompt, max_tokens=300, temperature=0.7, stream=False)["choices"][0]["text"].strip()
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


def build_ingredient_html(top_ingr: list[tuple[str, float]]) -> str:
    cards: list[str] = []
    for i, (name, conf) in enumerate(top_ingr):
        pct   = f"{conf * 100:.1f}%"
        color = _BADGE_COLORS[i % len(_BADGE_COLORS)]
        path  = get_ingredient_image(name)
        src   = _img_to_b64(path) if path and Path(path).exists() else None

        if src:
            cards.append(
                f'<div style="text-align:center;margin:6px;width:110px">'
                f'<img src="{src}" style="width:100px;height:100px;'
                f'object-fit:cover;border-radius:10px;border:1px solid #ddd">'
                f'<div style="font-size:12px;margin-top:3px;color:#333">{name}</div>'
                f'<div style="font-size:11px;color:#888">{pct}</div>'
                f'</div>'
            )
        else:
            cards.append(
                f'<div style="text-align:center;margin:6px;width:110px;'
                f'display:flex;flex-direction:column;align-items:center;justify-content:center">'
                f'<span style="background:{color};padding:6px 12px;border-radius:14px;'
                f'font-size:13px;font-weight:500;display:inline-block">🥬 {name}</span>'
                f'<div style="font-size:11px;color:#888;margin-top:4px">{pct}</div>'
                f'</div>'
            )

    return (
        '<div style="display:flex;flex-wrap:wrap;gap:4px;'
        'padding:8px;min-height:60px;align-items:flex-start">'
        + "".join(cards)
        + "</div>"
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
        items.append((img, row.get("recipe_title", "Recipe")))
    return items


def find_recipes(
    image: Image.Image | None,
    text_query: str,
    dietary: str,
    speed: str,
    dish_type: str,
    progress=gr.Progress(),
):
    t_total = time.perf_counter()

    if image is None and not (text_query or "").strip():
        raise gr.Error("Please upload a photo or type ingredient names.")

    debug: dict = {
        "models": {
            "cnn":      f"EfficientNet-B0  ({NUM_CLASSES} classes)",
            "embed":    f"{_EMBED_SHORT}  (cosine)",
            "reranker": "MLP reranker over FAISS score + overlap + filters + image signal",
            "llm":      "TinyLlama-1.1B-Chat Q4_K_M  (lazy — loads on Narrate)",
        },
        "query":           "",
        "reranker_scores": {},
        "timing_ms":       {},
    }

    top_ingr: list[tuple[str, float]]

    if image is not None:
        if not _cnn_loaded():
            gr.Info("Loading ingredient classifier for the first time (~15 s)…")
        progress(0.15, desc="Running ingredient classifier…")
        t0 = time.perf_counter()
        top_ingr = classify_ingredients(image)
        debug["timing_ms"]["cnn_ms"] = round((time.perf_counter() - t0) * 1000)
        names = [n for n, _ in top_ingr[:5]]
    else:
        names    = [s.strip() for s in text_query.split(",") if s.strip()]
        top_ingr = [(n, 1.0) for n in names[:10]]
        debug["timing_ms"]["cnn_ms"] = 0

    if not names:
        raise gr.Error("Could not extract any ingredient names from the input.")

    progress(0.45, desc="Searching 64k recipes…")
    t0 = time.perf_counter()
    recipes, query_text, scores = retrieve_recipes(names, dietary, speed, dish_type)
    debug["timing_ms"]["faiss_ms"] = round((time.perf_counter() - t0) * 1000)
    debug["query"] = query_text
    debug["reranker_scores"] = {
        r.get("recipe_title", f"recipe_{i}"): round(float(s), 4)
        for i, (r, s) in enumerate(zip(recipes, scores))
    }

    if not recipes:
        raise gr.Error(
            "No recipes matched those filters. "
            "Try setting Dietary and/or Cook speed to 'any'."
        )

    progress(0.75, desc="Building result panels…")
    t0 = time.perf_counter()

    ingr_html_str = build_ingredient_html(top_ingr)
    dish_gal      = build_dish_gallery(recipes)

    display = [
        {
            "Title":   r.get("recipe_title", ""),
            "Cuisine": _stringify(r.get(CUISINE_COL, "")),
            "Type":    _stringify(r.get("course_list", r.get("course", r.get("category", "")))),
            "Speed":   r.get("cook_speed", ""),
            "Dietary": _stringify(r.get(DIETARY_COL, "")),
        }
        for r in recipes
    ]
    recipe_df_data = pd.DataFrame(display)
    recipe_detail  = build_recipe_detail_md(recipes[0])

    debug["timing_ms"]["render_ms"]      = round((time.perf_counter() - t0) * 1000)
    debug["timing_ms"]["phase1_total_ms"] = round((time.perf_counter() - t_total) * 1000)

    elapsed_s = debug["timing_ms"]["phase1_total_ms"] / 1000
    faiss_s   = debug["timing_ms"]["faiss_ms"] / 1000
    status = (
        f"**Phase 1 complete** — found **{len(recipes)}** recipes "
        f"(FAISS {faiss_s:.2f}s · total {elapsed_s:.2f}s)  "
        f"| Click a row or dish image to select, then press **Narrate** for AI narration."
    )

    progress(1.0, desc="Done")
    return (
        status,
        ingr_html_str,
        dish_gal,
        recipe_df_data,
        recipe_detail,
        recipes[0],
        recipes,
        debug,
    )


def select_from_df(evt: gr.SelectData, results: list[dict]) -> tuple[dict, str, str]:
    if not results:
        return None, "", ""
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else 0
    row     = results[min(row_idx, len(results) - 1)]
    return row, f"Seleccionada: **{row.get('recipe_title', 'Receta')}**", build_recipe_detail_md(row)


def select_from_gallery(evt: gr.SelectData, results: list[dict]) -> tuple[dict, str, str]:
    if not results:
        return None, "", ""
    idx = min(int(evt.index), len(results) - 1)
    row = results[idx]
    return row, f"Seleccionada: **{row.get('recipe_title', 'Receta')}**", build_recipe_detail_md(row)


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


with gr.Blocks(title="Recomendador de Recetas", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🍳 Recomendador de Recetas")
    gr.Markdown(

    )

    recipe_state  = gr.State(None)
    results_state = gr.State([])

    with gr.Tabs():

        with gr.Tab("Buscar recetas"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=1, min_width=300):
                    img_input = gr.Image(
                        label="Foto de tu ingrediente (opcional)",
                        type="pil",
                        height=220,
                    )
                    text_input = gr.Textbox(
                        label="O describe ingredientes / antojo",
                        placeholder="tomate, cebolla, ajo, albahaca o pasta vegana rápida…",
                        lines=2,
                    )
                    dietary_dd = gr.Dropdown(
                        label="Preferencia dietética",
                        choices=DIETARY_CHOICES,
                        value="any",
                    )
                    speed_dd = gr.Dropdown(
                        label="Velocidad de cocción",
                        choices=SPEED_CHOICES,
                        value="any",
                    )
                    dish_type_dd = gr.Dropdown(
                        label="Tipo de platillo",
                        choices=DISH_TYPE_CHOICES,
                        value="any",
                    )
                    find_btn = gr.Button("Buscar Recetas 🔍", variant="primary", size="lg")

                    gr.Examples(
                        examples=[
                            ["tomate, mozzarella, albahaca",     "vegetarian", "any",    "any"],
                            ["pollo, ajo, limón",                "any",        "medium", "any"],
                            ["avena, plátano, miel",             "vegan",      "any",    "any"],
                            ["pasta, huevos, tocino, queso",     "any",        "medium", "any"],
                            ["frijoles negros, maíz, aguacate",  "vegan",      "any",    "any"],
                        ],
                        inputs=[text_input, dietary_dd, speed_dd, dish_type_dd],
                        label="Prueba un ejemplo",
                        examples_per_page=5,
                        cache_examples=False,
                    )

                with gr.Column(scale=2):

                    search_status = gr.Markdown(
                        "Sube una foto **o** escribe ingredientes, luego haz clic en **Buscar Recetas**."
                    )

                    with gr.Accordion("Ingredientes detectados", open=True):
                        ingr_html = gr.HTML(
                            value='<p style="color:#aaa;padding:8px;font-size:13px">—</p>'
                        )

                    with gr.Accordion("Mejores recetas", open=True):
                        dish_gallery = gr.Gallery(
                            label="Imágenes del platillo — haz clic para seleccionar",
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

                    with gr.Accordion("Procedimiento e ingredientes", open=True):
                        recipe_detail_md = gr.Markdown(
                            "Selecciona una receta para ver ingredientes y procedimiento."
                        )
                        narrate_btn = gr.Button(
                            "Narrar receta seleccionada ▶", variant="secondary"
                        )
                        narration_box = gr.Textbox(
                            lines=12,
                            interactive=False,
                            placeholder=(
                                "Selecciona una receta en la tabla de arriba, "
                                "luego haz clic en 'Narrar receta seleccionada'…"
                            ),
                            show_copy_button=True,
                            label="",
                        )

                    with gr.Accordion(
                        "Chat sobre esta receta  [TinyLlama · ~20 s por respuesta]",
                        open=True,
                    ):
                        chatbot = gr.Chatbot(height=300, bubble_full_width=False)
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Pregúntame lo que quieras sobre esta receta…",
                                show_label=False,
                                scale=5,
                            )
                            chat_btn = gr.Button("Enviar ↩", scale=1, variant="primary")
                        clear_btn = gr.Button("Limpiar chat", size="sm")

                    with gr.Accordion("Transparencia del pipeline", open=False):
                        gr.Markdown(
                            "_Texto de embedding, puntuaciones del reranker MLP "
                            "y tiempos por etapa de la última búsqueda._"
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
            pipeline_debug_json,
        ],
    )

    recipe_df.select(
        fn=select_from_df,
        inputs=[results_state],
        outputs=[recipe_state, search_status, recipe_detail_md],
    )

    dish_gallery.select(
        fn=select_from_gallery,
        inputs=[results_state],
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
