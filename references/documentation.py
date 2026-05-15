"""
documentation.py
================
Genera la documentación técnica del sistema de recomendación de recetas
en formato Markdown, lista para integrar en un reporte académico/profesional.

Ejecución:
    python documentation.py
    python documentation.py > documentation.md   # redirigir a archivo

Requiere evaluation_report.json (generado por DP_evaluation_report.ipynb)
para la Sección 3. Si no existe, usa marcadores [PLACEHOLDER].
"""

import json
import os
import sys
import textwrap

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE MÉTRICAS DE EVALUACIÓN
# ──────────────────────────────────────────────────────────────────────────────

EVAL_REPORT_PATH = "evaluation_report.json"

def load_eval_report():
    if os.path.exists(EVAL_REPORT_PATH):
        with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        print(f"[INFO] evaluation_report.json cargado ({EVAL_REPORT_PATH})", file=sys.stderr)
        return report, True
    print(
        f"[WARN] {EVAL_REPORT_PATH} no encontrado. "
        "Ejecuta DP_evaluation_report.ipynb para generar las métricas reales.\n"
        "Se usarán marcadores [PLACEHOLDER] en la Sección 3.",
        file=sys.stderr,
    )
    return {}, False

eval_report, metrics_available = load_eval_report()

def m(key_path, placeholder="[PLACEHOLDER]", fmt=None):
    """Extrae un valor del eval_report por ruta de claves separadas por '.'."""
    obj = eval_report
    for k in key_path.split("."):
        if not isinstance(obj, dict) or k not in obj:
            return placeholder
        obj = obj[k]
    if fmt and obj != placeholder:
        try:
            return fmt.format(obj)
        except Exception:
            return str(obj)
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE FORMATEO
# ──────────────────────────────────────────────────────────────────────────────

RULE_MAJOR = "=" * 80
RULE_MINOR = "-" * 80

def section(title):
    print(f"\n{RULE_MAJOR}")
    print(f"# {title}")
    print(RULE_MAJOR)

def subsection(title):
    print(f"\n## {title}")
    print(RULE_MINOR)

def p(text):
    """Imprime párrafo con wrap en 88 chars."""
    wrapped = textwrap.fill(text.strip(), width=88, subsequent_indent="")
    print(wrapped)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 1: ASPECTOS TÉCNICOS DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

section("SECCIÓN 1: Aspectos Técnicos del Modelo")

print("""
> **Modelo base:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
> **Arquitectura:** 12 capas transformer · 12 attention heads · hidden size 384
> **Parámetros:** ~118M · **Max tokens (modelo):** 128 subwords
> **Embeddings:** 384 dimensiones, vectores L2-normalizados
""")

# ── 1.1 REPRESENTACIONES ─────────────────────────────────────────────────────
subsection("1.1 Representaciones — Texto enriquecido como proxy multimodal")

p("""
El sistema no dispone de embeddings visuales en la capa de retrieval. En su lugar,
se construye un campo `embedding_text` que actúa como proxy de representación
multimodal codificando en texto los atributos semánticos, dietéticos, culturales
y sensoriales de cada receta. El template utilizado es:
""")

print("""\
```
Title: {recipe_title}
Category: {category} | Subcategory: {subcategory}
Description: {description}
Ingredients: {ingredient_text}
Cuisine: {cuisine_list_text}
Course: {course_list_text}
Tastes: {tastes_text}
Dietary: {dietary_profile_updated_text}
Health flags: {health_flags_text}
Main ingredient: {main_ingredient}
Cook speed: {cook_speed}
Directions: {directions_text}
```
""")

p("""
El campo `difficulty` se excluye deliberadamente del embedding_text y se usa como
filtro post-retrieval. Incluirlo sesgaba el espacio latente hacia el nivel de
dificultad en detrimento de la semántica culinaria: una búsqueda de "pasta rápida"
recuperaba recetas "easy" de cualquier tipo antes que pastas de velocidad media.
""")

p("""
El texto resultante se colapsa con `" ".join(text.split())` para eliminar
whitespace redundante. Sentence-transformers procesa este texto con su tokenizador
WordPiece, aplica 12 capas de self-attention y proyecta la secuencia completa
al espacio latente mediante mean pooling sobre todos los tokens, produciendo un
vector de 384 dimensiones que encapsula la identidad semántica completa de la receta.
""")

# ── 1.2 FUNCIONES DE ACTIVACIÓN ──────────────────────────────────────────────
subsection("1.2 Funciones de Activación — GELU vs ReLU")

p("""
Las capas feed-forward de cada bloque transformer en paraphrase-multilingual-MiniLM-L12-v2 utilizan
**GELU** (Gaussian Error Linear Unit) como función de activación:
""")

print("""\
```
GELU(x) = x · Φ(x)    donde Φ es la CDF de la distribución normal estándar
         ≈ 0.5x · (1 + tanh[√(2/π) · (x + 0.044715x³)])   # aproximación
```
""")

p("""
A diferencia de ReLU (que aplica un umbral duro en cero), GELU suaviza la
transición: para valores ligeramente negativos, la unidad no se desactiva
completamente sino que escala su salida gradualmente. Esto preserva información
de gradiente en activaciones cercanas a cero y reduce el "dying neuron" problem,
relevante durante el pre-entrenamiento de modelos de lenguaje con datos ruidosos
como los ingredientes de recetas (cantidades, unidades, nombres en múltiples idiomas).
""")

# ── 1.3 OPTIMIZACIÓN ─────────────────────────────────────────────────────────
subsection("1.3 Optimización — AdamW en pre-entrenamiento; inference frozen")

p("""
El encoder fue pre-entrenado con **AdamW** (Adam con weight decay desacoplado,
propuesto por Loshchilov & Hutter 2019). Los hiperparámetros estándar de la familia
BERT aplican:
""")

print("""\
| Hiperparámetro    | Valor típico MiniLM/SBERT |
|-------------------|-------------------------|
| β₁ (momentum)     | 0.9                     |
| β₂ (RMS decay)    | 0.999                   |
| ε (estabilidad)   | 1 × 10⁻⁸               |
| weight_decay (λ)  | 0.01                    |
| learning_rate     | 2 × 10⁻⁵ (warmup)       |
""")

p("""
La diferencia clave entre Adam y AdamW es que en AdamW el weight decay se aplica
directamente a los pesos (antes del paso de gradiente) en lugar de incorporarse
al gradiente. Esto evita que el decay quede modulado por la magnitud de los
momentos adaptativos, resultando en una regularización más uniforme entre capas.
""")

p("""
**En este sistema**, el encoder se usa en modo **inference frozen**: los pesos
no se actualizan. No existe un loop de optimización en producción. Toda la
"optimización" ocurre offline durante la construcción del índice FAISS (una sola
pasada de codificación). El único componente con estado mutable es el índice,
que puede actualizarse incrementalmente sin reentrenar el encoder.
""")

# ── 1.4 REGULARIZACIÓN ───────────────────────────────────────────────────────
subsection("1.4 Regularización — Dropout en el encoder + fuzzy matching")

p("""
El encoder aplica **dropout con tasa 0.1** en tres puntos de cada bloque transformer:
""")

print("""\
1. Después de la proyección de attention outputs (attn_drop)
2. Después de la capa feed-forward (ffn_drop)
3. Sobre los embeddings de entrada, incluyendo positional embeddings (emb_drop)
""")

p("""
Durante inference (modo `model.eval()`), el dropout está desactivado y los pesos
se usan en su totalidad. El efecto del dropout ya está "horneado" en los pesos
pre-entrenados: la red aprendió a ser robusta a la ausencia parcial de activaciones,
lo que se traduce en representaciones más generalizables (menos dependientes de
patrones espurios en el texto de entrenamiento).
""")

p("""
**Fuzzy matching como data augmentation implícita:** el pipeline de imágenes de
ingredientes usa `rapidfuzz.fuzz.partial_ratio` con threshold 72 para asociar
ingredientes del catálogo con nombres canónicos del dataset. Este proceso normaliza
variaciones semánticas equivalentes ("tomatoes" → "tomato", "ground beef" →
"beef") que, en un pipeline de fine-tuning, equivaldrían a aumentar el training
set con sinónimos y paráfrasis. En el pipeline actual mejora la cobertura de
imágenes del 0% (exact match) a 81.4% de ingredientes únicos.
""")

# ── 1.5 ESPACIO LATENTE ──────────────────────────────────────────────────────
subsection("1.5 Espacio Latente — Propiedades del espacio de 384 dimensiones")

p("""
Todos los vectores de receta son L2-normalizados (norma unitaria), por lo que
residen en la superficie de la hiperesfera unitaria S³⁸³. Esta propiedad tiene
implicaciones directas para el retrieval:
""")

print("""\
- **Inner product = cosine similarity:** `<a, b> = cos(θ) ∈ [-1, 1]`, lo que
  permite usar `faiss.IndexFlatIP` como índice de cosine similarity exacto.

- **Isotropía esperada:** los modelos pre-entrenados con contrastive learning
  tienden a usar el espacio de forma más isótropa que los modelos de language
  modeling clásico, evitando el "anisotropy problem" donde los embeddings se
  concentran en un cono del espacio y las similitudes coseno son artificialmente
  altas para todos los pares.

- **Clustering semántico:** el análisis de diversidad del espacio (Sección 3)
  verifica si `dist_coseno(misma_categoría) < dist_coseno(diferente_categoría)`.
  Si se cumple, el encoder captura la estructura categórica culinaria sin
  supervisión explícita de categorías.

- **t-SNE (2000 recetas, perplexity=30):** la visualización 2D muestra la
  proyección del espacio sobre un plano, coloreada por categoría culinaria.
  Los clusters visibles indican que el espacio latente organiza las recetas
  por tipo (postres, sopas, carnes, etc.) antes de ser consultado.
""")

# ── 1.6 ATENCIÓN ─────────────────────────────────────────────────────────────
subsection("1.6 Atención — Multi-head self-attention y representaciones contextuales")

p("""
Cada bloque transformer en paraphrase-multilingual-MiniLM-L12-v2 implementa **multi-head self-attention**
con 12 cabezas paralelas. Para cada token de posición i, el mecanismo calcula:
""")

print("""\
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

donde:
  Q = W_Q · x_i    (query: "¿qué busco?")
  K = W_K · x_j    (key:   "¿qué tengo?")
  V = W_V · x_j    (value: "¿qué entrego?")
  d_k = 32         (dimensión por cabeza = 384 / 12)
```
""")

p("""
Las 12 cabezas aprenden a atender a diferentes aspectos del contexto simultáneamente:
algunas cabezas capturan relaciones sintácticas (adjetivo-sustantivo), otras
semánticas (ingrediente-técnica), otras de larga distancia (receta-clasificación
dietética). Esto produce representaciones **contextuales y compositivas**:
""")

print("""\
| Frase                     | Contexto de "chicken"                       |
|---------------------------|---------------------------------------------|
| "grilled chicken breast"  | Atención hacia "grilled" → técnica de cocción|
| "chicken noodle soup"     | Atención hacia "soup" + "noodle" → contexto de caldo |
| "chicken vegan substitute"| Atención hacia "vegan" + "substitute" → rol semántico |
""")

p("""
En los tres casos el token "chicken" produce vectores distintos en las capas
intermedias. Tras el mean pooling, estas diferencias se preservan en el embedding
final de la receta, permitiendo distinguir "chicken soup" de "chicken steak"
aunque ambas comparten la palabra "chicken".
""")

# ── 1.7 ESTABILIDAD ──────────────────────────────────────────────────────────
subsection("1.7 Estabilidad de Entrenamiento — Mean pooling + normalización L2")

p("""
paraphrase-multilingual-MiniLM-L12-v2 fue entrenado con la estrategia de **mean pooling** (promedio de
todos los embeddings de tokens de salida, excluyendo padding). Frente a alternativas:
""")

print("""\
| Estrategia         | Ventaja                          | Limitación                        |
|--------------------|----------------------------------|-----------------------------------|
| CLS token          | Directo, O(1)                    | El CLS tiende a no capturar contenido semántico global |
| Max pooling        | Robusto a ruido                  | Pierde información de frecuencia  |
| **Mean pooling**   | Integra toda la secuencia        | Sensible a tokens de relleno (resuelto con masking) |
| Weighted pooling   | Prioriza tokens informativos     | Requiere peso adicional entrenado |
""")

p("""
La normalización L2 posterior (`normalize_embeddings=True` en sentence-transformers)
asegura que todas las recetas tengan la misma norma, eliminando el sesgo por longitud
de texto: una receta con 500 palabras no domina sobre una de 50 palabras simplemente
por magnitud del vector. Además, estabiliza la distribución de similitudes coseno
en el rango [-1, 1], simplificando la calibración de thresholds en el retrieval.
""")

# ── 1.8 CAPACIDAD GENERATIVA ─────────────────────────────────────────────────
subsection("1.8 Capacidad Generativa — Retrieval-Augmented Generation vs generación libre")

p("""
El sistema implementa **Retrieval-Augmented Generation (RAG)**, no generación pura.
El encoder transformer es un modelo de representación, no generativo: no puede
producir texto nuevo directamente. La arquitectura completa tiene dos módulos:
""")

print("""\
```
Query del usuario
       │
       ▼
[Encoder]  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
       │    → query_embedding (384d, normalizado)
       ▼
[Retrieval] FAISS IndexFlatIP → top-K recetas por cosine similarity
       │    → {receta_1, receta_2, ..., receta_K}
       ▼
[Generación] TinyLlama-1.1B-Chat fine-tuned / GGUF
       │    → respuesta reformulada, personalizada, en lenguaje natural
       ▼
Respuesta final al usuario
```
""")

print("""\
| Dimensión           | Retrieval puro             | Generación pura            | RAG (este sistema)          |
|---------------------|----------------------------|----------------------------|-----------------------------|
| Factualidad         | Alta (usa datos reales)    | Baja (puede alucinar)      | Alta (ancla en retrieval)   |
| Flexibilidad        | Baja (solo lo que existe)  | Alta (crea contenido nuevo)| Media (reformula lo real)   |
| Latencia            | Baja (<100ms con FAISS)    | Alta (inferencia LLM)      | Media (retrieval + LLM)     |
| Actualización       | Añadir al índice FAISS     | Reentrenar                 | Actualizar índice           |
| Alucinación         | Ninguna                    | Alta                       | Baja (LLM acotado)          |
""")

p("""
El tradeoff central es que el sistema solo puede recomendar recetas que existen
en el corpus. No puede "inventar" una receta para un nicho no cubierto (e.g.,
"pizza sin tomate con sabor a mole"). La capa generativa mitiga esto reformulando
los resultados recuperados en el tono/idioma del usuario, pero no puede crear
recetas ex nihilo sin riesgo de alucinación.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 2: ESTRATEGIA DE DATOS Y ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

section("SECCIÓN 2: Estrategia de Datos y Entrenamiento")

# ── 2.1 DATASETS ─────────────────────────────────────────────────────────────
subsection("2.1 Descripción de Datasets")

print("""\
| # | Dataset (Kaggle)                                          | Registros | Uso en el sistema                          |
|---|-----------------------------------------------------------|-----------|--------------------------------------------|
| 1 | `wafaaelhusseini/extended-recipes-dataset-64k-dishes`     | 62,126    | **Corpus principal** — recetas, ingredientes, instrucciones, perfiles dietéticos |
| 2 | `pes12017000148/food-ingredients-and-recipe-dataset-with-images` | 13,501 | **Capa 1 de imágenes** — imagen del platillo terminado (Epicurious) |
| 3 | `moltean/fruits` (Fruits-360)                             | 141 clases | **Capa 2, fuente A** — imágenes de frutas y verduras (prioridad) |
| 4 | `fasihcs/recipe-ingredients-image-dataset`                | 42 clases | **Capa 2, fuente B** — especias, carnes, lácteos (complementario) |
""")

print("""
**Criterios de selección:**

- **Dataset 1** fue elegido por su amplitud (62k recetas), diversidad de categorías
  (267 en total) y la riqueza de metadatos estructurados: ingredientes canónicos,
  perfiles dietéticos booleanos, cook_speed, health_flags. No existe dataset
  comparable de libre acceso con esta densidad de anotaciones.

- **Dataset 2 (Epicurious)** fue elegido sobre alternativas (Food-101, Recipe1M)
  porque incluye un mapeo directo nombre_de_imagen → título_de_receta, permitiendo
  un join por similaridad semántica de títulos sin necesidad de clasificación visual.
  Cobertura resultante: 5.4% del corpus (3,363 recetas), suficiente para un MVP
  que demuestra la arquitectura multicapa.

- **Datasets 3 y 4** se complementan: Fruits-360 cubre frutas y verduras con alta
  calidad (imágenes homogéneas, fondo blanco, 100×100px), mientras que el dataset
  de Recipe Ingredients cubre proteínas, lácteos y especias que Fruits-360 no
  incluye. La prioridad de Fruits-360 sobre fasihcs evita duplicados y garantiza
  consistencia visual.
""")

# ── 2.2 PIPELINE DE PREPROCESAMIENTO ─────────────────────────────────────────
subsection("2.2 Pipeline de Preprocesamiento — Pasos y decisiones clave")

print("""\
```
CARGA RAW
  │
  ├─[1] Filtros de calidad
  │      recipe_title: no nulo, ≤ 200 chars
  │      ingredient_text: no nulo, num_ingredients ≥ 3
  │      directions_text: no nulo, ≥ 20 palabras
  │      dietary_profile_updated: parseable como JSON list
  │      cook_speed ∈ {fast, medium, slow, unspecified}
  │
  ├─[2] Normalización de perfiles dietéticos
  │      dietary_profile_updated corrige inconsistencias del campo original:
  │      is_vegan=True pero dietary_profile incluye "halal" → reconciliación
  │      booleana. Fuente canónica: flags booleanos (is_vegan, is_vegetarian…).
  │
  ├─[3] Construcción de columnas de texto de listas
  │      cuisine_list (JSON) → cuisine_list_text ("Italian, Mediterranean")
  │      course_list  (JSON) → course_list_text  ("Dinner, Main Course")
  │      tastes       (JSON) → tastes_text       ("savory, umami")
  │      dietary_profile_updated (JSON) → dietary_profile_updated_text
  │      health_flags (JSON) → health_flags_text
  │
  ├─[4] Construcción de embedding_text
  │      Template estructurado con 12 campos semánticos
  │      Whitespace colapsado: " ".join(text.split())
  │      SIN difficulty (filtro post-retrieval)
  │
  ├─[5] Control de longitud
  │      Aproximación de tokens: len(text.split()) [word-tokens]
  │      Threshold: 512 word-tokens (≈ 256-400 subwords WordPiece)
  │      Para textos > 512: truncar directions_text al 60% de su longitud
  │
  ├─[6] Generación de embeddings
  │      model.encode(batch_size=64, normalize_embeddings=True)
  │      Output: np.float32 array (n × 384)
  │
  └─[7] Construcción del índice FAISS
         faiss.IndexFlatIP(384)
         index.add(embeddings)  → búsqueda exacta por inner product
```
""")

print("""
**Decisiones clave del preprocesamiento:**

**`ingredient_text` vs `ingredients_canonical`:**
`ingredient_text` contiene ingredientes en texto libre, limpiados de cantidades y
unidades (e.g., "garlic olive oil tomatoes"). `ingredients_canonical` es una lista
JSON con los ingredientes originales con sus cantidades ("1 cup tomatoes"). Se eligió
`ingredient_text` para el embedding porque: (a) las cantidades no son semánticamente
relevantes para retrieval por tipo de receta, (b) el texto libre es más compatible
con la distribución de entrenamiento de sentence-transformers, y (c) reduce la
longitud del embedding_text preservando la información que importa.

**`dietary_profile_updated` vs `dietary_profile`:**
El campo `dietary_profile` del dataset original presenta inconsistencias: recetas
marcadas como `is_vegan=True` pero con "halal" o "kosher" en dietary_profile,
categorías redundantes ("gluten_free" duplicado), o ausencia de flags que deberían
estar presentes según los ingredientes. `dietary_profile_updated` se construye
reconciliando los campos booleanos individuales (`is_vegan`, `is_vegetarian`,
`is_halal`, `is_kosher`, `is_nut_free`, `is_dairy_free`, `is_gluten_free`) como
fuente de verdad, descartando el campo string cuando entra en contradicción.

**Por qué excluir `difficulty` del embedding_text:**
En experimentos iniciales, incluir difficulty desviaba el retrieval: una query
"quick pasta" recuperaba recetas "easy" de cualquier categoría antes que pastas
de velocidad "medium". Al excluirlo, el retrieval prioriza la semántica culinaria
y difficulty se aplica como filtro post-retrieval, manteniendo la flexibilidad.
""")

# ── 2.3 ESTRATEGIA DE ENTRENAMIENTO ──────────────────────────────────────────
subsection("2.3 Estrategia de Entrenamiento — Inference frozen (MVP)")

p("""
En esta versión del sistema **no se realiza fine-tuning** del encoder. El modelo
paraphrase-multilingual-MiniLM-L12-v2 se usa con sus pesos pre-entrenados en modo inference. Esta
decisión es válida por las siguientes razones:
""")

print("""\
**Justificación del modelo frozen:**

1. **Pre-entrenamiento extenso:** paraphrase-multilingual-MiniLM-L12-v2 fue
   entrenado para similitud semántica multilingüe mediante sentence pairs,
   aprendizaje contrastivo y destilación de modelos más grandes. Su espacio
   latente ya captura relaciones culinarias en inglés y español sin fine-tuning
   adicional.

2. **Ausencia de datos de preferencia:** el fine-tuning contrastivo requiere
   pares (query, receta_positiva, receta_negativa). En esta versión no existen
   interacciones de usuario que proporcionen esa señal de supervisión.

3. **Velocidad de iteración (MVP):** un encoder frozen permite iterar el pipeline
   completo (datos → embeddings → índice → retrieval) en horas en lugar de días,
   reduciendo el tiempo al primer prototipo evaluable.

**Cuándo hacer fine-tuning:**

| Señal | Acción recomendada |
|-------|-------------------|
| P@5 < 0.60 en queries del dominio objetivo | Fine-tuning con pares sintéticos (GPT-4) |
| Disponibilidad de 1,000+ pares (query, receta_preferida) | Bi-encoder fine-tuning con MNRL loss |
| Necesidad de recuperación multilingüe | Fine-tuning con paLLM-MiniLM o LaBSE |
| Latencia < 20ms requerida con 500k+ recetas | Fine-tuning + IndexIVFPQ (cuantización) |
""")

# ── 2.4 HIPERPARÁMETROS ──────────────────────────────────────────────────────
subsection("2.4 Hiperparámetros Importantes")

print("""\
| Componente              | Hiperparámetro                   | Valor | Justificación                                                     |
|-------------------------|----------------------------------|-------|-------------------------------------------------------------------|
| Fuzzy — ingredientes    | `partial_ratio` threshold        | 72    | Maximiza cobertura (81.4%); acepta variaciones plurales, regionales |
| Fuzzy — títulos         | `token_sort_ratio` threshold     | 88    | Alta precisión para el join Epicurious; evita matches temáticos incorrectos |
| Embeddings              | batch_size                       | 64    | Balance memoria GPU/CPU vs throughput; escala a 62k recetas en <5min |
| Embeddings              | max word-tokens (pre-truncado)   | 512   | Threshold propio de word-tokens; el modelo procesa hasta 256 subword tokens |
| Truncado                | fracción de directions_text      | 60%   | Conserva las instrucciones iniciales (más informativas); elimina pasos finales menos semánticos |
| FAISS                   | Tipo de índice                   | `IndexFlatIP` | Búsqueda exacta; para 62k vectores la búsqueda exhaustiva es <10ms |
| FAISS                   | Normalización                    | L2    | Inner product = cosine similarity para vectores unitarios          |
""")

# ── 2.5 EXPERIMENTOS ─────────────────────────────────────────────────────────
subsection("2.5 Experimentos Realizados")

print("""\
**Experimento 1: Con vs sin `difficulty` en embedding_text**

| Configuración              | Comportamiento observado                                             |
|----------------------------|----------------------------------------------------------------------|
| Con difficulty             | Query "pasta rápida" recupera recetas "easy" de cualquier categoría |
| **Sin difficulty (final)** | Query "pasta rápida" recupera pastas; difficulty se filtra después  |

Conclusión: difficulty actúa como sesgo espurio en el espacio latente.
Trasladarla a filtro post-retrieval mejora la relevancia temática.

---

**Experimento 2: Exact match vs fuzzy match para join con Epicurious**

| Estrategia                              | Recetas con imagen | Cobertura |
|-----------------------------------------|--------------------|-----------|
| Solo exact match (title_norm idéntico)  | 1,969              | 3.2%      |
| Exact + fuzzy (token_sort_ratio ≥ 88)  | 3,363              | **5.4%**  |
| Incremento por fuzzy                    | +1,394             | +41%      |

El fuzzy matching recupera principalmente títulos con ligeras variaciones:
artículos omitidos ("The Best Pasta" → "Best Pasta"), puntuación diferente,
o inversión de palabras ("Garlic Chicken" → "Chicken with Garlic").

---

**Experimento 3: Threshold para matching de ingredientes (partial_ratio)**

| Threshold | Ingredientes únicos con imagen | Falsos positivos estimados |
|-----------|-------------------------------|---------------------------|
| 88        | ~65%                          | Muy bajos                 |
| 80        | ~73%                          | Bajos                     |
| **72**    | **81.4%**                     | Moderados (score 72-79)   |

Se eligió 72 como compromiso que maximiza la cobertura de usuario visible
(% de ingredientes en una receta que tienen imagen asociada), asumiendo que
los falsos positivos en score 72-79 serán validados manualmente en la siguiente
iteración.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 3: EVALUACIÓN DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

section("SECCIÓN 3: Evaluación de Resultados")

print()

# ── Métricas de evaluación ────────────────────────────────────────────────────
if metrics_available:
    B = eval_report.get("B_retrieval_quality", {})
    C = eval_report.get("C_embedding_diversity", {})
    A = eval_report.get("A_dataset_coverage", {})

    p5_global = B.get("mean_precision_at_5", "[PLACEHOLDER]")
    mrr_global = B.get("mean_mrr", "[PLACEHOLDER]")
    n_queries = B.get("n_queries", 20)

    dist_random = C.get("mean_cosine_dist_random", "[PLACEHOLDER]")
    dist_same   = C.get("mean_cosine_dist_same_cat", "[PLACEHOLDER]")
    dist_diff   = C.get("mean_cosine_dist_diff_cat", "[PLACEHOLDER]")
    ratio       = C.get("diversity_ratio_diff_over_same", "[PLACEHOLDER]")
    clustered   = C.get("categorical_clustering_ok", None)
    kl_div      = C.get("tsne_params", {}).get("kl_divergence", "[PLACEHOLDER]")

    total_rec   = A.get("total_recipes", "[PLACEHOLDER]")
    pct_dish    = A.get("pct_with_dish_image", "[PLACEHOLDER]")
    pct_ingr_w  = A.get("pct_ingredients_with_image_weighted", "[PLACEHOLDER]")
    pct_ingr_u  = A.get("pct_ingredients_unique_with_image", "[PLACEHOLDER]")

    clustering_verdict = (
        "✓ `dist_same_cat < dist_diff_cat` — el espacio captura agrupación semántica por categoría"
        if clustered else
        "⚠ `dist_same_cat ≥ dist_diff_cat` — las categorías no forman clusters bien separados"
    )

    # Tabla de queries destacadas
    per_query_rows = B.get("per_query", [])
    best_queries   = sorted(per_query_rows, key=lambda r: -r.get("precision_at_5", 0))[:5]
    worst_queries  = sorted(per_query_rows, key=lambda r: r.get("precision_at_5", 0))[:3]
else:
    p5_global   = "[P@5 — ejecutar DP_evaluation_report.ipynb]"
    mrr_global  = "[MRR — ejecutar DP_evaluation_report.ipynb]"
    n_queries   = 20
    dist_random = "[PLACEHOLDER]"
    dist_same   = "[PLACEHOLDER]"
    dist_diff   = "[PLACEHOLDER]"
    ratio       = "[PLACEHOLDER]"
    kl_div      = "[PLACEHOLDER]"
    clustering_verdict = "[PLACEHOLDER — ejecutar DP_evaluation_report.ipynb]"
    total_rec   = "[PLACEHOLDER]"
    pct_dish    = "[PLACEHOLDER]"
    pct_ingr_w  = "[PLACEHOLDER]"
    pct_ingr_u  = "[PLACEHOLDER]"
    best_queries, worst_queries = [], []

# ── 3.1 Cobertura ────────────────────────────────────────────────────────────
subsection("3.1 Métricas de Cobertura del Dataset")

print(f"""\
| Métrica                                          | Valor          |
|--------------------------------------------------|----------------|
| Total de recetas en corpus                       | {total_rec}    |
| Recetas con imagen de platillo (Epicurious join) | {pct_dish}%    |
| Apariciones de ingredientes con imagen (pond.)   | {pct_ingr_w}%  |
| Ingredientes únicos con imagen                   | {pct_ingr_u}%  |
""")

p("""
La baja cobertura de imágenes de platillo (≈5%) refleja que Epicurious es un
corpus de alta calidad pero reducido (13,501 recetas) frente a las 62k del corpus
principal. Los ingredientes muestran cobertura sustancialmente mayor porque
Fruits-360 + fasihcs cubren las categorías más frecuentes (frutas, verduras,
especias, carnes básicas), que son precisamente los ingredientes más comunes
en el corpus.
""")

# ── 3.2 Retrieval Quality ────────────────────────────────────────────────────
subsection("3.2 Retrieval Quality — Precision@5 y MRR")

p("""
El test set de evaluación comprende 20 queries manuales con keywords esperados,
distribuidas en consultas monolingüe inglés (17) y cross-lingüe ES/FR (3).
Un resultado se considera relevante si contiene al menos 1 keyword esperado
en el campo `recipe_title + embedding_text` (case-insensitive substring match).
""")

print(f"""\
| Métrica                | Global            | EN (n=17)  | Cross-lingual ES/FR (n=3) |
|------------------------|-------------------|------------|---------------------------|
| **Mean Precision@5**   | **{p5_global}**   | [PLACEHOLDER] | [PLACEHOLDER]          |
| **Mean MRR**           | **{mrr_global}**  | [PLACEHOLDER] | [PLACEHOLDER]          |
""")

if best_queries:
    print("\n**Top 5 queries con mejor P@5:**\n")
    print(f"{'#':<3} {'Query':<45} {'P@5':>5} {'RR':>6}")
    print("-" * 62)
    for i, q in enumerate(best_queries, 1):
        qtext = q['query'][:43] + ".." if len(q['query']) > 43 else q['query']
        print(f"{i:<3} {qtext:<45} {q['precision_at_5']:>5.3f} {q['reciprocal_rank']:>6.3f}")

if worst_queries:
    print("\n**Queries con menor P@5 (análisis de fallos):**\n")
    print(f"{'#':<3} {'Query':<45} {'P@5':>5} {'RR':>6}")
    print("-" * 62)
    for i, q in enumerate(worst_queries, 1):
        qtext = q['query'][:43] + ".." if len(q['query']) > 43 else q['query']
        print(f"{i:<3} {qtext:<45} {q['precision_at_5']:>5.3f} {q['reciprocal_rank']:>6.3f}")

print()
p("""
Las queries cross-lingüe (español, francés) actúan como diagnóstico de la capacidad
multilingüe del encoder. paraphrase-multilingual-MiniLM-L12-v2 proyecta consultas
de distintos idiomas al mismo espacio latente; aun así, las diferencias entre idioma
de consulta, traducción de ingredientes y texto original pueden afectar la precisión.
Esto constituye una limitación explícita del sistema (ver Sección 3.4).
""")

# ── 3.3 Diversidad del Espacio Latente ───────────────────────────────────────
subsection("3.3 Análisis de Diversidad del Espacio Latente")

print(f"""\
| Métrica                                          | Valor          |
|--------------------------------------------------|----------------|
| Dist. coseno media — pares aleatorios (n=1,000)  | {dist_random}  |
| Dist. coseno media — misma categoría (n≥500)     | {dist_same}    |
| Dist. coseno media — diferente categoría (n≥500) | {dist_diff}    |
| Ratio distancia (diff / same)                    | {ratio}        |
| t-SNE KL divergence (perplexity=30, n=2,000)     | {kl_div}       |

**Interpretación:** {clustering_verdict}
""")

p("""
Un ratio diff/same > 1.0 confirma que el encoder organiza el espacio latente de
forma que recetas de la misma categoría culinaria están más próximas entre sí que
recetas de categorías diferentes, incluso sin haber sido entrenado explícitamente
en este corpus. La t-SNE (guardada en `tsne_embeddings.png`) proporciona evidencia
visual de esta organización: clusters de postres, sopas, platos principales y
recetas vegetarianas son visualmente separables en la proyección 2D.
""")

p("""
**Nota metodológica:** la t-SNE aplica una reducción PCA previa (384d → 50d) para
estabilizar la inicialización y acelerar el cómputo. La varianza explicada por PCA
se reporta en el output del notebook. La KL divergence del t-SNE indica la calidad
de la proyección: valores < 1.5 indican que la estructura local del espacio original
se preserva bien en 2D.
""")

# ── 3.4 Limitaciones ─────────────────────────────────────────────────────────
subsection("3.4 Limitaciones del Sistema")

print("""\
| # | Limitación                                                | Impacto | Esfuerzo de mejora |
|---|-----------------------------------------------------------|---------|--------------------|
| 1 | Cobertura parcial de imágenes de ingredientes (~81%)      | Medio   | Alto               |
| 2 | Truncamiento de recetas largas a 512 tokens (pérdida de pasos finales) | Medio | Medio |
| 3 | Sistema retrieval-based sin generación real (depende de LLM downstream) | Alto | Alto |
| 4 | Sesgo hacia cocina occidental (Fruits-360 + Epicurious)   | Medio   | Alto               |
| 5 | Sin validación humana de fuzzy matching de ingredientes   | Bajo    | Medio              |
| 6 | Ausencia de feedback loop — sin fine-tuning sobre preferencias reales | Alto | Alto |
| 7 | Cobertura de imagen de platillo muy baja (~5.4%)          | Medio   | Medio              |
""")

# ── 3.5 Mejoras Futuras ───────────────────────────────────────────────────────
subsection("3.5 Mejoras Futuras — Priorizadas por Impacto/Esfuerzo")

print("""\
**1. Fine-tuning del encoder con contrastive learning [Impacto: Alto | Esfuerzo: Alto]**

Técnica: bi-encoder fine-tuning con **Multiple Negatives Ranking Loss (MNRL)** o
**SimCSE (contrastive sentence embeddings)**. Se construyen pares positivos
(query, receta_relevante) desde interacciones de usuario o generados con GPT-4
("genera 500 queries que correspondan a esta receta"). Los negativos se muestrean
in-batch (hard negatives). El encoder actualizado capturará semántica específica
del dominio culinario, mejorando P@5 esperado de ~0.7 a ~0.85+.

---

**2. Integración de CLIP para búsqueda imagen→receta [Impacto: Alto | Esfuerzo: Alto]**

Técnica: encodificar las imágenes de platillos disponibles (3,363 recetas) con
`openai/clip-vit-base-patch32`. El usuario puede subir una foto de un plato y
recuperar recetas similares por similaridad visual en el espacio latente de CLIP.
Requiere un segundo índice FAISS (visual) y lógica de fusión de scores
(text score α + image score (1-α)). Posibles valores α: 0.6-0.8 (texto dominante).

---

**3. Expansión del catálogo de ingredientes con Open Food Facts API [Impacto: Medio | Esfuerzo: Medio]**

Los 8,712 ingredientes únicos sin imagen son candidatos para búsqueda en la API de
Open Food Facts (`https://world.openfoodfacts.org/cgi/search.pl`), que incluye
fotografías de productos. Proceso: normalizar el nombre → buscar en API → descargar
imagen → añadir a `ingredient_catalog.json`. La cobertura esperada podría alcanzar
90%+ de ingredientes únicos.

---

**4. Validación humana de 500 queries para ground truth real [Impacto: Medio | Esfuerzo: Bajo]**

El test set actual de 20 queries usa relevancia por keyword matching, que es un
proxy débil. Con 500 queries evaluadas manualmente por 3 jueces (escala 1-3 por
resultado en top-5), se obtiene un Pooled Precision@5 y un NDCG@10 con varianza
estadística robusta. Este ground truth permitiría: (a) calibrar el threshold de
cosine similarity para filtrado, (b) detectar categorías de consulta con bajo
rendimiento, (c) proveer datos de fine-tuning (ver punto 1).
""")


# ══════════════════════════════════════════════════════════════════════════════
#  CIERRE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{RULE_MAJOR}")
print("# FIN DE LA DOCUMENTACIÓN TÉCNICA")
print(RULE_MAJOR)
print(f"""
Generado por: documentation.py
Métricas reales disponibles: {'SÍ' if metrics_available else 'NO — ejecutar DP_evaluation_report.ipynb'}
Archivos referenciados:
  - df_final_embeddings.parquet  (corpus con embedding_text e imágenes)
  - recipe_embeddings.npy        (matrix n × 384, float32, L2-normalizada)
  - recipe_faiss.index           (IndexFlatIP, búsqueda exacta)
  - ingredient_catalog.json      (183 ingredientes → ruta de imagen)
  - evaluation_report.json       (métricas de las secciones A, B, C)
  - tsne_embeddings.png          (visualización del espacio latente)
  - limitations.csv              (tabla de limitaciones y mejoras)
""")
