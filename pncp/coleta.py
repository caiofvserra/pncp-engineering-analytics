"""
Coleta de contratos via API PNCP.

Para evitar OOM em coletas multi-ano (2023–2025 ≈ 200–300k contratos),
salva um parquet parcial por (UF, ano) e libera RAM antes de baixar o próximo.
No fim, concatena tudo num único `contratos.parquet`.

Endpoints relevantes (https://pncp.gov.br/api/consulta):
  /v1/contratacoes/publicacao  — contratações publicadas
"""

import time
from pathlib import Path

import pandas as pd
import requests

from pncp import config
from pncp.io_disco import (
    salvar_parquet, ler_parquet, concatenar_parquets,
)
from pncp.ram import liberar, monitorar_ram


def _baixar_pagina(uf, data_ini, data_fim, pagina, tamanho):
    """Uma chamada à API. Retorna lista de contratos (pode ser vazia)."""
    params = {
        "dataInicial": data_ini,
        "dataFinal": data_fim,
        "uf": uf,
        "pagina": pagina,
        "tamanhoPagina": tamanho,
    }
    url = f"{config.API_BASE}/v1/contratacoes/publicacao"
    for tentativa in range(config.TENTATIVAS_HTTP):
        try:
            r = requests.get(url, params=params, timeout=config.TIMEOUT_HTTP)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as exc:
            if tentativa == config.TENTATIVAS_HTTP - 1:
                print(f"[coleta] falha definitiva pág={pagina}: {exc}")
                return []
            time.sleep(2 ** tentativa)
    return []


def _coletar_ano(uf, ano, tamanho, max_paginas):
    """Coleta um ano inteiro (todas as páginas até esvaziar)."""
    data_ini = f"{ano}0101"
    data_fim = f"{ano}1231"
    contratos = []
    for pagina in range(1, max_paginas + 1):
        lote = _baixar_pagina(uf, data_ini, data_fim, pagina, tamanho)
        if not lote:
            break
        contratos.extend(lote)
        time.sleep(config.PAUSA_PAGINA)
        if pagina % 10 == 0:
            print(f"[coleta] {uf}/{ano} pág {pagina} → {len(contratos):,} acum")
    return contratos


def _normalizar(contratos):
    """Lista de dicts da API → DataFrame com colunas estáveis e dtypes leves."""
    df = pd.DataFrame(contratos)
    if df.empty:
        return df

    # Coluna do rótulo (Lei 14.133)
    if "categoriaProcessoId" in df.columns:
        df["categoria_id"] = df["categoriaProcessoId"].astype("Int16")
    df["rotulo"] = df["categoria_id"].map(config.rotular).astype("category")

    # Texto-alvo da classificação
    for col in ("objetoCompra", "informacaoComplementar"):
        if col not in df.columns:
            df[col] = ""
    df["objeto"] = (df["objetoCompra"].fillna("") + " "
                    + df["informacaoComplementar"].fillna(""))

    # Ano (para filtros temporais)
    if "dataPublicacaoPncp" in df.columns:
        df["anoPublicacao"] = (
            pd.to_datetime(df["dataPublicacaoPncp"], errors="coerce")
            .dt.year.astype("Int16")
        )

    # Ajusta dtypes para reduzir RAM
    for col in ("unidadeOrgao", "orgaoEntidade"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    if "valorTotalEstimado" in df.columns:
        df["valor"] = pd.to_numeric(df["valorTotalEstimado"], errors="coerce") \
                      .astype("float32")

    return df


def coletar(uf, anos, tamanho=500, max_paginas=400, sobrescrever=False):
    """
    Coleta contratos para a UF e anos dados, salvando parquet por ano e
    consolidando em `dados/coleta/contratos.parquet`.

    Args:
        uf: sigla da UF (ex: "SP")
        anos: iterável de anos (ex: range(2023, 2026))
        tamanho: tamanho de página da API (máx 500)
        max_paginas: corta após N páginas por ano (segurança)
        sobrescrever: se False, pula anos já baixados

    Returns:
        Path do parquet consolidado.
    """
    anos = list(anos)
    monitorar_ram("início coleta")

    for ano in anos:
        parcial = config.caminho(config.SUB_COLETA, f"contratos_{uf}_{ano}.parquet")
        if parcial.exists() and not sobrescrever:
            print(f"[coleta] {uf}/{ano} já existe, pulando ({parcial.name})")
            continue

        print(f"[coleta] baixando {uf}/{ano}...")
        contratos = _coletar_ano(uf, ano, tamanho, max_paginas)
        df = _normalizar(contratos)
        if df.empty:
            print(f"[coleta] {uf}/{ano} retornou vazio")
            continue
        salvar_parquet(df, parcial)
        print(f"[coleta] {uf}/{ano} → {len(df):,} contratos em {parcial.name}")
        liberar(df, contratos)
        monitorar_ram(f"após {uf}/{ano}")

    # Consolida todos os parciais
    consolidado = config.caminho(config.SUB_COLETA, "contratos.parquet")
    padrao = str(config.PASTA_DADOS / config.SUB_COLETA / f"contratos_{uf}_*.parquet")
    concatenar_parquets(padrao, consolidado)
    print(f"[coleta] consolidado em {consolidado}")
    return consolidado


def filtrar_anos(caminho_parquet, ano_minimo=None, ano_maximo=None):
    """Reescreve o parquet aplicando filtro de anos (útil após EDA)."""
    df = ler_parquet(caminho_parquet)
    n_antes = len(df)
    if ano_minimo is not None:
        df = df[df["anoPublicacao"] >= ano_minimo]
    if ano_maximo is not None:
        df = df[df["anoPublicacao"] <= ano_maximo]
    salvar_parquet(df, caminho_parquet)
    print(f"[coleta] filtro temporal: {n_antes:,} → {len(df):,} contratos")
    liberar(df)
    return caminho_parquet
