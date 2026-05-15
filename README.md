# Recipe Recommendation System — ITESO Grupo Parquet

Sistema multimodal de recomendación de recetas que combina búsqueda semántica (FAISS), clasificación de imágenes (EfficientNet-B0) y generación de texto (TinyLlama QLoRA).

Demo: [Grupo-Parquet-ITESO/ProyectoFinal_recetas](https://huggingface.co/spaces/Grupo-Parquet-ITESO/ProyectoFinal_recetas)

---

## Estructura

```
├── data/
│   ├── raw/          ← dataset original (df_clean_final.parquet)
│   ├── interim/      ← artefactos intermedios (traducción, imágenes)
│   └── processed/    ← listos para inferencia (FAISS index, embeddings)
├── models/           ← pesos entrenados (.pth, .gguf, class_labels.json)
├── notebooks/        ← exploración y pipeline (0–3_DP_*.ipynb)
├── recetas/          ← paquete Python del proyecto
│   ├── config.py     ← rutas y constantes centralizadas
│   └── modeling/     ← scripts de entrenamiento y preprocesamiento
├── reports/          ← métricas de evaluación y figuras
├── space_repo/       ← archivos del Hugging Face Space
├── run_local.py      ← lanzar la app localmente
├── deploy_to_hf.py   ← desplegar al Space de HF
└── upload_to_hub.py  ← subir modelos al HF Hub
```

---

## Setup

```bash
conda create -n recipes python=3.12 -y
conda activate recipes

# Opción A — instalar todas las dependencias directamente
pip install -r requirements.txt

# Opción B — instalar como paquete editable
pip install -e ".[train]"
```

---

## Variables de entorno

```bash
export HF_TOKEN="hf_..."           # requerido para deploy y subida de modelos
export HF_USERNAME="tu-usuario"    # default: Grupo-Parquet-ITESO
```

---

## Flujo completo — desde cero

### 1. Descargar el dataset de recetas

El dataset se descarga automáticamente al correr el notebook `0_DP_dataset_versionfinal.ipynb`, o manualmente:

```bash
python -c "import kagglehub; kagglehub.dataset_download('wafaaelhusseini/extended-recipes-dataset-64k-dishes')"
```

Colocar el parquet resultante en `data/raw/df_clean_final.parquet`.

### 2. Descargar Fruits-360 (para entrenar la CNN)

```bash
python -c "import kagglehub; kagglehub.dataset_download('moltean/fruits')"
```

O configurar la ruta manualmente:

```bash
export FRUITS360_DIR="/ruta/a/fruits-360_100x100/fruits-360"
```

### 3. Traducir y construir el índice FAISS

```bash
python recetas/modeling/translate_and_rebuild.py --batch-size 128
```

Genera en `data/processed/`: `df_final_embeddings.parquet`, `recipe_faiss.index`, `recipe_embeddings.npy`

### 4. Entrenar la CNN

```bash
python recetas/modeling/train_cnn.py --epochs 15 --batch-size 128
```

Genera: `models/efficientnet_ingredients.pth` y `models/class_labels.json`

### 5. Entrenar el LLM (requiere GPU ≥ 8 GB VRAM)

```bash
python recetas/modeling/train_llm.py --epochs 3 --batch-size 4
```

Convertir a GGUF para inferencia:

```bash
python llama.cpp/convert_hf_to_gguf.py models/tinyllama-recipes-merged/ --outtype q4_k_m
```

### 6. Subir modelos al Hub

```bash
python upload_to_hub.py
```

### 7. Ejecutar localmente

```bash
# Modo mínimo — solo FAISS + texto
python run_local.py --no-cnn --no-llm

# Completo — auto-detecta modelos disponibles
python run_local.py

# Opciones adicionales
python run_local.py --port 7861 --share
```

### 8. Desplegar al Space de Hugging Face

```bash
export HF_TOKEN="hf_..."
python deploy_to_hf.py

# Omitir smoke test si el LLM tarda en arrancar
python deploy_to_hf.py --skip-smoke
```

---

## Docker

La opción más rápida para correr la app sin instalar nada manualmente.

**Requisitos previos:** Docker instalado y `data/processed/` poblado (pasos 1–3 del flujo).

El Dockerfile usa `requirements-inference.txt` con torch CPU-only (~500 MB vs ~2 GB del build CUDA). El build completo tarda ~5 min en la primera vez; las reconstrucciones posteriores son rápidas gracias al caché de capas.

```bash
# Construir la imagen (primera vez ~5 min, luego usa caché)
docker compose build

# Modo mínimo — solo FAISS + búsqueda de texto
docker compose up app

# Modo completo — CNN + LLM (requiere models/ con .pth y .gguf)
docker compose --profile full up app-full
```

La app queda disponible en `http://localhost:7860`.

`data/processed/` y `models/` se montan como volúmenes de solo lectura — no se copian dentro de la imagen. Al actualizar los artefactos localmente, basta con reiniciar el contenedor sin reconstruir.

```bash
# Pasar variables de entorno en el mismo comando
HF_TOKEN=hf_... HF_USERNAME=tu-usuario docker compose up app
```

**Archivos Docker:**

| Archivo | Propósito |
|---|---|
| `Dockerfile` | Imagen base con dependencias de inferencia |
| `docker-compose.yml` | Servicios `app` (mínimo) y `app-full` (CNN + LLM) |
| `requirements-inference.txt` | Deps exactas para el contenedor (CPU torch, sin training tools) |
| `.dockerignore` | Excluye `data/`, `models/`, `notebooks/` del contexto de build |

---

## Dependencias por etapa

| Etapa | Instalación |
|---|---|
| Inferencia + app local | `pip install -r requirements.txt` |
| Entrenamiento | `pip install -e ".[train]"` |
| HF Space | `space_repo/requirements.txt` (HF lo instala automáticamente) |

---

## Recursos en Hugging Face

| Recurso | ID |
|---|---|
| Space (demo) | `Grupo-Parquet-ITESO/ProyectoFinal_recetas` |
| CNN | `Grupo-Parquet-ITESO/recipe-ingredient-classifier` |
| LLM GGUF | `Grupo-Parquet-ITESO/recipe-llm-gguf` |
| Dataset | `wafaaelhusseini/extended-recipes-dataset-64k-dishes` |
