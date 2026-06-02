"""
I/O em disco — parquet, json, sparse npz, modelos joblib.

Toda persistência do pipeline passa por aqui. Centralizar facilita trocar
formatos depois (ex: parquet → feather) sem mexer em cada módulo.
"""

import json
from pathlib import Path
import pandas as pd


# ── Parquet ──────────────────────────────────────────────────────────────────
def salvar_parquet(df: pd.DataFrame, caminho, **kwargs):
    """Salva DataFrame em parquet com snappy. Cria diretório-pai se preciso."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(caminho, compression="snappy", index=False, **kwargs)
    return caminho


def ler_parquet(caminho, colunas=None) -> pd.DataFrame:
    """Lê parquet. Aceita lista de colunas para economizar RAM."""
    return pd.read_parquet(caminho, columns=colunas)


def iter_parquet(caminho, batch_size=50_000):
    """
    Itera um parquet em batches (PyArrow). Usar para arquivos grandes
    onde não cabe ler tudo em RAM.
    """
    import pyarrow.parquet as pq
    arquivo = pq.ParquetFile(caminho)
    for batch in arquivo.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


# ── JSON (métricas, config, glossário) ───────────────────────────────────────
def salvar_json(obj, caminho):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return caminho


def ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


# ── Sparse (TF-IDF) ──────────────────────────────────────────────────────────
def salvar_sparse(matriz, caminho):
    """Salva scipy.sparse em .npz."""
    from scipy import sparse
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(caminho, matriz)
    return caminho


def ler_sparse(caminho):
    from scipy import sparse
    return sparse.load_npz(caminho)


# ── Modelos sklearn ──────────────────────────────────────────────────────────
def salvar_modelo(modelo, caminho):
    import joblib
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, caminho, compress=3)
    return caminho


def ler_modelo(caminho):
    import joblib
    return joblib.load(caminho)


# ── Embeddings densos ────────────────────────────────────────────────────────
def salvar_npy(arr, caminho):
    import numpy as np
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    np.save(caminho, arr)
    return caminho


def ler_npy(caminho, mmap=True):
    """mmap=True não carrega tudo em RAM — cada slice traz só o que precisa."""
    import numpy as np
    return np.load(caminho, mmap_mode="r" if mmap else None)


# ── Concatenar parquets parciais (usado pela coleta multi-ano) ───────────────
def concatenar_parquets(padrao_glob, caminho_saida) -> Path:
    """
    Lê todos parquets que casam com o glob e concatena num único arquivo.
    Usa PyArrow Dataset para não materializar tudo em RAM ao mesmo tempo.

    Aceita glob absoluto (ex: '/dados/coleta/contratos_SP_*.parquet')
    OU relativo (ex: 'dados/coleta/contratos_*.parquet').
    """
    import glob as _glob
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    arquivos = [Path(p) for p in _glob.glob(str(padrao_glob))]
    if not arquivos:
        raise FileNotFoundError(f"nenhum arquivo casa com '{padrao_glob}'")
    dataset = ds.dataset(arquivos, format="parquet")
    tabela = dataset.to_table()
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tabela, caminho_saida, compression="snappy")
    return caminho_saida
