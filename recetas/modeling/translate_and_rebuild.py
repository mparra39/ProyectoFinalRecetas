import argparse
import time

import numpy as np
import pandas as pd
import torch
import faiss
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from recetas.config import DATA, EMBED_MODEL_NAME, TRANSLATE_MODEL_NAME

INPUT_PARQUET  = DATA.raw_parquet
OUTPUT_PARQUET = DATA.final_parquet
FAISS_INDEX    = DATA.faiss_index
EMBEDDINGS_NPY = DATA.embeddings_npy

TRANSLATE_MODEL = TRANSLATE_MODEL_NAME
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_translator(device):
    print(f"Loading translator {TRANSLATE_MODEL}…")
    tok = MarianTokenizer.from_pretrained(TRANSLATE_MODEL)
    mdl = MarianMTModel.from_pretrained(TRANSLATE_MODEL).to(device)
    mdl.eval()
    return tok, mdl


def translate_batch(texts: list[str], tok, mdl, device, max_len=128) -> list[str]:
    inputs = tok(texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_length=max_len)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


def translate_series(series: pd.Series, tok, mdl, device,
                     batch_size: int = 64) -> pd.Series:
    """Translates a string Series; deduplicates before calling the model."""
    unique_texts = series.dropna().unique().tolist()
    cache: dict[str, str] = {}

    print(f"  Unique texts to translate: {len(unique_texts):,}")
    for i in tqdm(range(0, len(unique_texts), batch_size), unit="batch"):
        batch = unique_texts[i : i + batch_size]
        translations = translate_batch(batch, tok, mdl, device)
        cache.update(zip(batch, translations))

    return series.map(lambda x: cache.get(x, x) if pd.notna(x) else x)


def rebuild_faiss(texts: pd.Series, embed_model_name: str, device: str,
                  batch_size: int = 256):
    print(f"\nLoading embedding model: {embed_model_name}…")
    model = SentenceTransformer(embed_model_name, device=device)

    print("Generating embeddings…")
    embeddings = model.encode(
        texts.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype("float32")

    print(f"Embeddings shape: {embeddings.shape}")
    np.save(EMBEDDINGS_NPY, embeddings)
    print(f"Saved → {EMBEDDINGS_NPY}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX))
    print(f"FAISS index saved → {FAISS_INDEX}  ({index.ntotal:,} vectors)")

    return embeddings, index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",       type=int, default=64)
    parser.add_argument("--no-rebuild-faiss", action="store_true")
    parser.add_argument("--no-translate",     action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading {INPUT_PARQUET.name}…")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  {len(df):,} recipes | columns: {list(df.columns)[:8]}…")

    if not args.no_translate:
        tok, mdl = load_translator(device)

        print("\n[1/2] Translating ingredient_text…")
        t0 = time.time()
        df["ingredient_text_es"] = translate_series(
            df["ingredient_text"], tok, mdl, device, args.batch_size
        )
        print(f"  done ({time.time()-t0:.0f}s)")

        print("\nExamples:")
        for en, es in zip(df["ingredient_text"].head(3), df["ingredient_text_es"].head(3)):
            print(f"  EN: {en[:80]}")
            print(f"  ES: {es[:80]}")
            print()

        print(f"Saving {OUTPUT_PARQUET.name}…")
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"  saved → {OUTPUT_PARQUET}")
    else:
        print(f"Loading translated parquet {OUTPUT_PARQUET.name}…")
        df = pd.read_parquet(OUTPUT_PARQUET)

    if not args.no_rebuild_faiss:
        embed_col = "ingredient_text_es" if "ingredient_text_es" in df.columns else "ingredient_text"
        print(f"\n[2/2] Rebuilding FAISS on column '{embed_col}'…")
        embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        rebuild_faiss(df[embed_col].fillna(""), EMBED_MODEL, embed_device)

    print("\nDone.")
    print(f"  Parquet:    {OUTPUT_PARQUET}")
    print(f"  FAISS:      {FAISS_INDEX}")
    print(f"  Embeddings: {EMBEDDINGS_NPY}")


if __name__ == "__main__":
    main()
