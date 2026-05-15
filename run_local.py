"""
Launches the recipe recommender locally with graceful degradation.

Works with whatever models are available:
  - FAISS + text search  → always on if .parquet/.index exist
  - CNN (photo input)    → requires efficientnet_ingredients.pth + class_labels.json
  - LLM (narration)      → requires tinyllama-recipes-q4.gguf

Usage:
    python run_local.py
    python run_local.py --port 7861
    python run_local.py --no-llm
    python run_local.py --share
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from recetas.config import DATA, MODELS, EMBED_MODEL_NAME

FAISS_PATH  = DATA.faiss_index
PARQUET     = DATA.final_parquet
CATALOG     = DATA.ingredient_catalog
CNN_WEIGHTS = MODELS.cnn_weights
LABELS_FILE = MODELS.class_labels
GGUF_PATH   = MODELS.gguf

DIETARY_CHOICES = ["any", "vegetarian", "vegan", "gluten-free", "dairy-free"]
SPEED_CHOICES   = ["any", "fast", "medium", "slow"]
_BADGE_COLORS   = ["#FFB3B3", "#B3D9FF", "#B3FFB3", "#FFD9B3",
                   "#E8B3FF", "#B3FFE8", "#FFE8B3", "#D9B3FF"]


def _print_status(use_cnn: bool, use_llm: bool) -> None:
    GREEN, YELLOW, RESET, BOLD = "\033[32m", "\033[33m", "\033[0m", "\033[1m"
    ok   = f"{GREEN}enabled{RESET}"
    warn = f"{YELLOW}disabled{RESET}"
    print(f"\n{BOLD}Recipe Recommender — Local Dev Server{RESET}")
    print("─" * 44)
    print(f"  FAISS retrieval (text):   {ok}")
    print(f"  CNN (ingredient photo):   {ok if use_cnn else warn}")
    if not use_cnn:
        print(f"    missing: {CNN_WEIGHTS.name}  and/or  {LABELS_FILE.name}")
    print(f"  LLM narration (TinyLlama): {ok if use_llm else warn}")
    if not use_llm:
        print(f"    missing: {GGUF_PATH.name}")
    print("─" * 44 + "\n")


def _check_core_imports() -> None:
    missing = []
    for pkg, imp in [
        ("gradio",                "gradio"),
        ("faiss",                 "faiss"),
        ("pandas",                "pandas"),
        ("sentence_transformers", "sentence_transformers"),
        ("rapidfuzz",             "rapidfuzz"),
        ("PIL",                   "PIL"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing packages:")
        for m in missing:
            print(f"  pip install {m}")
        sys.exit(1)


def load_core(parquet=None, faiss_index=None, embed_model=None):
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    faiss_path   = Path(faiss_index) if faiss_index else FAISS_PATH
    parquet_path = Path(parquet)     if parquet     else PARQUET
    model_name   = embed_model or EMBED_MODEL_NAME

    for path in [faiss_path, parquet_path, CATALOG]:
        if not path.exists():
            print(f"ERROR: missing {path.name} in {path.parent}")
            sys.exit(1)

    print("Loading FAISS index…")
    index = faiss.read_index(str(faiss_path))

    print("Loading dataframe…")
    df = pd.read_parquet(str(parquet_path)).reset_index(drop=True)

    with open(CATALOG) as f:
        catalog: dict[str, str] = json.load(f)

    print(f"Loading SentenceTransformer ({model_name})…")
    embedder = SentenceTransformer(model_name)

    if "ingredient_text_es" in df.columns:
        ingr_col = "ingredient_text_es"
    elif "ingredient_text" in df.columns:
        ingr_col = "ingredient_text"
    else:
        ingr_col = "ingredients_text_processed"
    dietary_col = "dietary_profile" if "dietary_profile" in df.columns else "dietary_profile_updated"
    cuisine_col = "cuisine_list"    if "cuisine_list"    in df.columns else "cuisine"

    print(f"  {len(df):,} recipes | ingr_col={ingr_col}\n")
    return index, df, catalog, embedder, ingr_col, dietary_col, cuisine_col


def load_cnn_if_available(skip: bool):
    if skip or not CNN_WEIGHTS.exists() or not LABELS_FILE.exists():
        return None, None, None
    try:
        import torch
        import torchvision.models as tv_models
        import torchvision.transforms as T
    except ImportError:
        print("  torch/torchvision not installed — CNN disabled")
        return None, None, None

    print("Loading EfficientNet-B0…")
    with open(LABELS_FILE) as f:
        labels: dict[str, str] = json.load(f)
    num_cls = len(labels)
    mdl = tv_models.efficientnet_b0(weights=None)
    mdl.classifier[1] = torch.nn.Linear(1280, num_cls)
    mdl.load_state_dict(torch.load(str(CNN_WEIGHTS), map_location="cpu"))
    mdl.eval()
    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"  CNN ready — {num_cls} classes\n")
    return mdl, labels, tf


def load_llm_if_available(skip: bool):
    if skip or not GGUF_PATH.exists():
        return None
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  llama-cpp-python not installed — LLM disabled")
        return None
    print("Loading TinyLlama GGUF…")
    llm = Llama(model_path=str(GGUF_PATH), n_ctx=2048, n_threads=4, verbose=False)
    size_gb = GGUF_PATH.stat().st_size / 1_073_741_824
    print(f"  LLM ready ({size_gb:.2f} GB)\n")
    return llm


def _parse_dietary(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x).lower() for x in raw]
    try:
        return [str(x).lower() for x in json.loads(raw)]
    except Exception:
        return [str(raw).lower()]


def _stringify(val) -> str:
    import pandas as _pd
    if isinstance(val, list):
        return ", ".join(str(x) for x in val)
    try:
        return ", ".join(str(x) for x in json.loads(val))
    except Exception:
        return str(val) if _pd.notna(val) else ""


def _fuzzy_image(name: str, catalog: dict, catalog_keys: list) -> str | None:
    from rapidfuzz import process as rfp
    hit = rfp.extractOne(name.lower(), catalog_keys)
    if hit and hit[1] >= 72:
        return catalog[hit[0]]
    return None


def _img_to_b64(path: str) -> str | None:
    try:
        ext  = Path(path).suffix.lstrip(".").lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def build_ingredient_html(top_ingr, catalog, catalog_keys) -> str:
    cards = []
    for i, (name, conf) in enumerate(top_ingr):
        pct   = f"{conf*100:.1f}%"
        color = _BADGE_COLORS[i % len(_BADGE_COLORS)]
        path  = _fuzzy_image(name, catalog, catalog_keys)
        src   = _img_to_b64(path) if path and Path(path).exists() else None
        if src:
            cards.append(
                f'<div style="text-align:center;margin:6px;width:110px">'
                f'<img src="{src}" style="width:100px;height:100px;'
                f'object-fit:cover;border-radius:10px;border:1px solid #ddd">'
                f'<div style="font-size:12px;margin-top:3px">{name}</div>'
                f'<div style="font-size:11px;color:#888">{pct}</div></div>'
            )
        else:
            cards.append(
                f'<div style="text-align:center;margin:6px;width:110px;'
                f'display:flex;flex-direction:column;align-items:center">'
                f'<span style="background:{color};padding:6px 12px;'
                f'border-radius:14px;font-size:13px;font-weight:500">🥬 {name}</span>'
                f'<div style="font-size:11px;color:#888;margin-top:4px">{pct}</div></div>'
            )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:4px;padding:8px">'
        + "".join(cards) + "</div>"
    )


def build_app(faiss_index, df, catalog, embedder,
              ingr_col, dietary_col, cuisine_col,
              cnn_model, cnn_labels, cnn_tf,
              llm):
    import gradio as gr
    import pandas as pd
    from PIL import Image

    catalog_keys = list(catalog.keys())
    USE_CNN = cnn_model is not None
    USE_LLM = llm is not None

    _GREY = Image.new("RGB", (200, 200), color=(210, 210, 210))

    def classify_ingredients(image: Image.Image) -> list[tuple[str, float]]:
        import torch
        tensor = cnn_tf(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(cnn_model(tensor), dim=1)[0]
        top10 = torch.topk(probs, 10)
        return [(cnn_labels.get(str(i.item()), f"cls_{i.item()}"), s.item())
                for i, s in zip(top10.indices, top10.values)]

    def retrieve(names, dietary, speed, k=5):
        query = "ingredients: " + ", ".join(names)
        emb   = embedder.encode([query], normalize_embeddings=True).astype("float32")
        dists, idxs = faiss_index.search(emb, 20)
        cands = df.iloc[idxs[0]].copy()
        cands["_score"] = dists[0]
        if dietary != "any":
            mask  = cands[dietary_col].apply(lambda v: dietary.lower() in _parse_dietary(v))
            cands = cands[mask]
        if speed != "any" and "cook_speed" in cands.columns:
            cands = cands[cands["cook_speed"].str.lower() == speed.lower()]
        top    = cands.head(k)
        scores = top["_score"].tolist()
        return top.to_dict(orient="records"), query, scores

    def dish_gallery(recipes):
        items = []
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

    def find_recipes(image, text_query, dietary, speed, progress=gr.Progress()):
        t0 = time.perf_counter()
        if image is None and not (text_query or "").strip():
            raise gr.Error("Upload a photo or type ingredient names.")

        top_ingr: list[tuple[str, float]]
        if image is not None and USE_CNN:
            progress(0.2, desc="Classifying ingredient…")
            top_ingr = classify_ingredients(image)
            names    = [n for n, _ in top_ingr[:5]]
        elif image is not None and not USE_CNN:
            gr.Warning("CNN not available — using text input instead.")
            names    = [s.strip() for s in (text_query or "").split(",") if s.strip()]
            top_ingr = [(n, 1.0) for n in names[:10]]
        else:
            names    = [s.strip() for s in text_query.split(",") if s.strip()]
            top_ingr = [(n, 1.0) for n in names[:10]]

        if not names:
            raise gr.Error("No ingredients found in the input.")

        progress(0.5, desc="Searching 64k recipes…")
        recipes, query, scores = retrieve(names, dietary, speed)

        if not recipes:
            raise gr.Error("No results with those filters. Try setting filters to 'any'.")

        progress(0.8, desc="Building panels…")
        ingr_html_val = build_ingredient_html(top_ingr, catalog, catalog_keys)
        dgal          = dish_gallery(recipes)
        display       = [{
            "Title":   r.get("recipe_title", ""),
            "Cuisine": _stringify(r.get(cuisine_col, "")),
            "Speed":   r.get("cook_speed", ""),
            "Dietary": _stringify(r.get(dietary_col, "")),
        } for r in recipes]
        recipe_df_val = pd.DataFrame(display)

        debug = {
            "query": query,
            "scores": {r.get("recipe_title", f"r{i}"): round(float(s), 4)
                       for i, (r, s) in enumerate(zip(recipes, scores))},
            "timing_ms": {"total": round((time.perf_counter() - t0) * 1000)},
        }
        status = (
            f"**{len(recipes)} recipes found** in "
            f"{debug['timing_ms']['total']/1000:.2f}s  "
            f"| Select a row and press **Narrate** for LLM narration."
        )
        progress(1.0)
        return status, ingr_html_val, dgal, recipe_df_val, recipes[0], recipes, json.dumps(debug, indent=2)

    def narrate(recipe_row):
        if not recipe_row:
            yield "Select a recipe first."
            return
        if not USE_LLM:
            yield (
                f"**LLM not available locally.**\n\n"
                f"**Recipe:** {recipe_row.get('recipe_title', '')}\n\n"
                f"**Ingredients:** {recipe_row.get(ingr_col, '')}\n\n"
                f"**Instructions:** {recipe_row.get('directions_text', '')[:500]}…"
            )
            return
        ingr  = recipe_row.get(ingr_col, "")
        ingr  = ingr.replace(" ", ", ") if " " in ingr and "," not in ingr else ingr
        dirs  = recipe_row.get("directions_text", "")[:800]
        prompt = (
            "<|system|>\nYou are a helpful cooking assistant.\n</s>\n"
            "<|user|>\nPlease narrate this recipe in a friendly way:\n"
            f"Title: {recipe_row.get('recipe_title', '')}\n"
            f"Ingredients: {ingr}\nInstructions: {dirs}\n</s>\n"
            "<|assistant|>\n"
        )
        out = ""
        for chunk in llm(prompt, max_tokens=512, temperature=0.7, stream=True):
            out += chunk["choices"][0]["text"]
            yield out

    def chat(message, history, recipe):
        if not message.strip():
            return history, ""
        if not USE_LLM:
            reply = (
                f"LLM not available. Active recipe: "
                f"**{recipe.get('recipe_title', 'none') if recipe else 'none'}**"
            )
            return history + [[message, reply]], ""
        ctx = (f"The user is asking about '{recipe.get('recipe_title', '')}'. "
               f"Ingredients: {recipe.get(ingr_col, '')}"
               if recipe else "You are a helpful cooking assistant.")
        prompt = (f"<|system|>\n{ctx}\n</s>\n"
                  f"<|user|>\n{message}\n</s>\n<|assistant|>\n")
        reply = llm(prompt, max_tokens=300, temperature=0.7)["choices"][0]["text"].strip()
        return history + [[message, reply]], ""

    def sel_df(evt: gr.SelectData, results):
        if not results:
            return None, ""
        row = results[min(evt.index[0] if isinstance(evt.index, (list, tuple)) else 0,
                          len(results) - 1)]
        return row, f"Selected: **{row.get('recipe_title', '')}**"

    def sel_gal(evt: gr.SelectData, results):
        if not results:
            return None, ""
        row = results[min(int(evt.index), len(results) - 1)]
        return row, f"Selected: **{row.get('recipe_title', '')}**"

    cnn_label = "CNN active" if USE_CNN else "CNN unavailable (missing .pth)"
    llm_label = "LLM active" if USE_LLM else "LLM unavailable (missing .gguf)"

    with gr.Blocks(title="Recipe Recommender [LOCAL]", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# 🍳 Recipe Recommender — Local Dev Server")
        gr.Markdown(f"`FAISS 64k recipes`  |  `{cnn_label}`  |  `{llm_label}`")

        recipe_state  = gr.State(None)
        results_state = gr.State([])

        with gr.Tabs():
            with gr.Tab("Find recipes"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        img_in = gr.Image(
                            label="Ingredient photo (requires CNN)",
                            type="pil", height=200,
                            interactive=USE_CNN,
                        )
                        txt_in = gr.Textbox(
                            label="Or type ingredients (comma-separated)",
                            placeholder="tomato, onion, garlic, basil…",
                            lines=2,
                        )
                        diet_dd = gr.Dropdown(
                            label="Dietary preference",
                            choices=DIETARY_CHOICES, value="any",
                        )
                        speed_dd = gr.Dropdown(
                            label="Cook speed",
                            choices=SPEED_CHOICES, value="any",
                        )
                        find_btn = gr.Button("Search Recipes", variant="primary", size="lg")
                        gr.Examples(
                            examples=[
                                ["tomato, mozzarella, basil",    "vegetarian", "fast"],
                                ["chicken, garlic, lemon",       "any",        "medium"],
                                ["oats, banana, honey",          "vegan",      "fast"],
                                ["pasta, eggs, bacon, parmesan", "any",        "medium"],
                                ["black beans, corn, avocado",   "vegan",      "fast"],
                            ],
                            inputs=[txt_in, diet_dd, speed_dd],
                            label="Quick examples",
                            examples_per_page=5,
                        )

                    with gr.Column(scale=2):
                        status_md = gr.Markdown(
                            "Type ingredients or upload a photo, then click **Search Recipes**."
                        )
                        with gr.Accordion("Detected ingredients", open=True):
                            ingr_html = gr.HTML(value='<p style="color:#aaa;padding:8px">—</p>')
                        with gr.Accordion("Top recipes", open=True):
                            dish_gal = gr.Gallery(
                                label="Dish images",
                                columns=5, height=190,
                                object_fit="cover", allow_preview=False,
                            )
                            recipe_df = gr.Dataframe(
                                headers=["Title", "Cuisine", "Speed", "Dietary"],
                                interactive=False, wrap=True, row_count=(5, "fixed"),
                            )
                        narrate_label = (
                            "LLM Narration" if USE_LLM
                            else "LLM Narration (disabled — missing .gguf)"
                        )
                        with gr.Accordion(narrate_label, open=False):
                            narrate_btn = gr.Button("Narrate selected recipe", variant="secondary")
                            narration_box = gr.Textbox(
                                lines=10, interactive=False,
                                show_copy_button=True, label="",
                                placeholder="Select a recipe and click Narrate…",
                            )
                        with gr.Accordion("Chat about this recipe", open=False):
                            chatbot   = gr.Chatbot(height=280, bubble_full_width=False)
                            with gr.Row():
                                chat_in  = gr.Textbox(show_label=False, scale=5,
                                                      placeholder="Ask about this recipe…")
                                chat_btn = gr.Button("Send", scale=1, variant="primary")
                            clear_btn = gr.Button("Clear chat", size="sm")
                        with gr.Accordion("Pipeline debug", open=False):
                            debug_json = gr.Textbox(label="", lines=8, interactive=False, value="")

            with gr.Tab("How it works"):
                gr.Markdown("""
## Pipeline

```
Text  ───────────────────────────────┐
                                     ▼
Photo → EfficientNet-B0 → names → MiniLM → FAISS 64k recipes
                                     │
                                     ▼ (< 3 s)
                               Top-5 recipes
                                     │ click Narrate
                                     ▼ (~ 30 s on CPU)
                          TinyLlama Q4_K_M → narration
```

| Component | Local status |
|---|---|
| FAISS retrieval | active |
| EfficientNet-B0 CNN | """ + ("active" if USE_CNN else "needs `efficientnet_ingredients.pth`") + """ |
| TinyLlama LLM | """ + ("active" if USE_LLM else "needs `tinyllama-recipes-q4.gguf`") + """ |
""")

        find_btn.click(
            fn=find_recipes,
            inputs=[img_in, txt_in, diet_dd, speed_dd],
            outputs=[status_md, ingr_html, dish_gal, recipe_df,
                     recipe_state, results_state, debug_json],
        )
        recipe_df.select(fn=sel_df, inputs=[results_state],
                         outputs=[recipe_state, status_md])
        dish_gal.select(fn=sel_gal, inputs=[results_state],
                        outputs=[recipe_state, status_md])
        narrate_btn.click(fn=narrate, inputs=[recipe_state], outputs=[narration_box])
        chat_btn.click(fn=chat, inputs=[chat_in, chatbot, recipe_state],
                       outputs=[chatbot, chat_in])
        chat_in.submit(fn=chat, inputs=[chat_in, chatbot, recipe_state],
                       outputs=[chatbot, chat_in])
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, chat_in])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",         type=int,  default=7860)
    parser.add_argument("--share",        action="store_true",
                        help="Generate a temporary public URL via ngrok")
    parser.add_argument("--no-llm",       action="store_true",
                        help="Skip LLM even if .gguf exists")
    parser.add_argument("--no-cnn",       action="store_true",
                        help="Skip CNN even if .pth exists")
    parser.add_argument("--parquet",      default=None)
    parser.add_argument("--faiss-index",  default=None)
    parser.add_argument("--embed-model",  default=None)
    args = parser.parse_args()

    _check_core_imports()

    faiss_index, df, catalog, embedder, ingr_col, dietary_col, cuisine_col = load_core(
        parquet=args.parquet,
        faiss_index=args.faiss_index,
        embed_model=args.embed_model,
    )
    cnn_model, cnn_labels, cnn_tf = load_cnn_if_available(args.no_cnn)
    llm = load_llm_if_available(args.no_llm)

    _print_status(use_cnn=cnn_model is not None, use_llm=llm is not None)

    demo = build_app(
        faiss_index, df, catalog, embedder,
        ingr_col, dietary_col, cuisine_col,
        cnn_model, cnn_labels, cnn_tf,
        llm,
    )
    demo.queue(max_size=2)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
