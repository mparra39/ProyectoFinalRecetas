---
title: ProyectoFinal Recetas
emoji: 🍳
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
python_version: '3.12'
app_file: app_optimized.py
pinned: false
license: mit
models:
  - ramonsj11/recipe-ingredient-classifier
  - ramonsj11/recipe-llm-gguf
datasets:
  - wafaaelhusseini/extended-recipes-dataset-64k-dishes
---

# Recipe Recommender — Multimodal AI Demo

An end-to-end recipe recommendation system that accepts ingredient photos,
free-text cravings, comma-separated ingredient lists, and optional filters. It
returns matching recipes with dish images, full ingredients/procedure, and
TinyLlama-powered narration/chat.

---

## What It Does

The pipeline offers **three input modes** and returns results across **four output panels**:

| Input mode | How to use |
|---|---|
| **Ingredient photo** | Upload a photo of an ingredient; EfficientNet-B0 identifies it automatically |
| **Text query** | Type a free-text request: *"quick vegan pasta under 30 min"* |
| **Optional filters** | Choose diet, cook speed, and dish type |

| Output panel | Content |
|---|---|
| **Detected ingredients** | Ingredient predictions with catalog images when available |
| **Top-5 recipes** | Dish images plus title, cuisine, type, speed, and dietary tags |
| **Procedure and ingredients** | Selected recipe ingredients, directions, and TinyLlama narration |
| **Recipe chat** | Questions answered with the active recipe as context |

---

## Technical Stack

| Component | Model / Library | Role |
|---|---|---|
| **Visual classifier** | EfficientNet-B0 (fine-tuned) | Ingredient recognition from photos |
| **Semantic retrieval** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | Query → FAISS IndexFlatIP search (Spanish & English) |
| **Vector index** | FAISS `IndexFlatIP` | 64k recipe embeddings, cosine similarity |
| **Reranker** | Small MLP over retrieval/filter features | Reorder FAISS candidates before Top-5 display |
| **Recipe generation** | TinyLlama-1.1B-Chat Q4\_K\_M (GGUF) | Recipe narration and active-recipe chat |
| **Fuzzy matching** | RapidFuzz | Ingredient name normalization |
| **Interface** | Gradio 4.44 | Web UI |

All inference runs on **CPU only** — no GPU required.

---

## How to Use

1. **Text query** — Type your craving or dietary constraint in the search box and
   press *Search*. Example: `"spicy Thai noodles, no nuts"`.

2. **Photo upload** — Click *Upload Ingredient Photo*, choose a clear photo of a
   single ingredient (fruit, vegetable, spice, meat, or dairy item), and press
   *Identify & Search*. The classifier will label it and run a retrieval
   automatically.

3. **Filters** — Optionally select dietary preference, cook speed, and dish type
   before pressing *Find Recipes*.

Results appear in the four panels on the right. Click any recipe row or dish
image to set the active recipe, then use the narration button or chat panel.

---

## Data Sources

| Dataset | Use |
|---|---|
| `wafaaelhusseini/extended-recipes-dataset-64k-dishes` | Main cleaned 64k recipe base |
| Epicurious subset | Finished-dish images where available |
| Fruits-360 | Fruit/vegetable ingredient images and CNN training coverage |
| Recipe Ingredients Image Dataset | Spices, meats, dairy, and non-fruit ingredients |

---

## Limitations

- **CPU-only inference**: The LLM step (TinyLlama) takes **~25–40 seconds** on
  the free-tier CPU. Retrieval and classification are fast (<2 s).
- **Image coverage**: Ingredient images are available for roughly **60–70 %** of
  ingredients. Missing images display a placeholder.
- **Classifier scope**: The visual classifier was trained on Fruits-360 (120
  classes) + Recipe Ingredients Dataset (~30 categories). Uncommon or composite
  ingredients may be misidentified.
- **LLM quality**: TinyLlama at Q4\_K\_M is a small model. Generated recipes are
  plausible but not professionally tested. Always verify quantities.
- **Language**: The retrieval index and ingredients are in **Spanish**. The LLM
  (TinyLlama) was fine-tuned on English recipes; its output may mix languages.
- **Dataset**: Recipes come from the
  [extended-recipes-dataset-64k-dishes](https://huggingface.co/datasets/wafaaelhusseini/extended-recipes-dataset-64k-dishes).
  Coverage skews toward Western and South/East Asian cuisines.

---

## Local Development

```bash
git clone https://huggingface.co/spaces/{HF_USERNAME}/ProyectoFinal_recetas
cd ProyectoFinal_recetas
pip install -r requirements.txt
python app_optimized.py
```

The first run downloads model weights from the Hub automatically (~700 MB total).
