PYTHON := $(shell which python)
CONDA_ENV := recipes

.PHONY: setup install lab run train-cnn train-llm translate deploy clean

## Instalar el paquete en modo editable (conda activate recipes primero)
install:
	pip install -e ".[train]"

## Arrancar JupyterLab apuntando a notebooks/
lab:
	jupyter lab --notebook-dir=notebooks/ --no-browser

## Correr la app localmente (solo FAISS, sin CNN ni LLM)
run:
	python run_local.py --no-cnn --no-llm

## Correr la app con todos los modelos disponibles
run-full:
	python run_local.py

## Traducir ingredientes y reconstruir FAISS en español
translate:
	python recetas/modeling/translate_and_rebuild.py --batch-size 128

## Entrenar EfficientNet-B0 (CNN clasificador de ingredientes)
train-cnn:
	python recetas/modeling/train_cnn.py --epochs 15 --batch-size 128

## Fine-tune TinyLlama con QLoRA
train-llm:
	python recetas/modeling/train_llm.py --epochs 3 --batch-size 4 --grad-accum 8

## Subir modelos al HF Hub (requiere HF_TOKEN)
upload:
	python upload_to_hub.py

## Desplegar Space completo a HF (requiere HF_TOKEN)
deploy:
	python deploy_to_hf.py

## Mostrar estructura del proyecto
tree:
	find . -not -path "*/.git/*" -not -path "*/__pycache__/*" \
	       -not -path "*/\.*" -not -path "*/data/raw/*" \
	       -not -path "*/data/processed/*" | sort | head -60

## Limpiar archivos temporales
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Limpio."
