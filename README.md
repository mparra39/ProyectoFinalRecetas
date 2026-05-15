# ProyectoFinal Recetas — ITESO Grupo Parquet

Sistema multimodal de recomendación de recetas que combina búsqueda semántica FAISS, clasificación de imágenes (EfficientNet-B0) y generación de texto (TinyLlama QLoRA).

## Estructura del proyecto

```
├── data/
│   ├── raw/          ← dato original inmutable (df_clean_final.parquet)
│   ├── interim/      ← transformaciones intermedias (imágenes, traducción)
│   └── processed/    ← listos para modelado (FAISS index, embeddings, parquet final)
├── models/           ← modelos entrenados (.pth, .gguf, class_labels.json)
├── notebooks/        ← exploración y pipeline (0–3_DP_*.ipynb)
├── recetas/          ← paquete Python fuente
│   ├── config.py     ← todas las rutas y constantes centralizadas
│   └── modeling/     ← scripts de entrenamiento y traducción
├── reports/
│   └── figures/      ← visualizaciones (.png, .json, .csv)
├── references/       ← documentación de apoyo
├── space_repo/       ← archivos para Hugging Face Space
├── run_local.py      ← lanzar app localmente
├── deploy_to_hf.py   ← desplegar al Space de HF
└── upload_to_hub.py  ← subir modelos al HF Hub
```

## Setup

```bash
conda create -n recipes python=3.12 -y
conda activate recipes
pip install -e ".[train]"
```

## Flujo de trabajo

### 1. Traducción y FAISS (español)
```bash
python recetas/modeling/translate_and_rebuild.py --batch-size 128
```

### 2. Entrenar CNN
```bash
python recetas/modeling/train_cnn.py --epochs 15 --batch-size 128
```

### 3. Entrenar LLM
```bash
python recetas/modeling/train_llm.py --epochs 3 --batch-size 4
```

### 4. Correr localmente
```bash
python run_local.py --no-cnn --no-llm
```

### 5. Desplegar a Hugging Face
```bash
export HF_TOKEN="hf_..."
python deploy_to_hf.py
```

## Hugging Face Space

[Grupo-Parquet-ITESO/ProyectoFinal_recetas](https://huggingface.co/spaces/Grupo-Parquet-ITESO/ProyectoFinal_recetas)
# ProyectoFinalRecetas
# ProyectoFinalRecetas
