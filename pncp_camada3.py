"""
pncp_camada3.py — Camada 3 do TCC: Termos Aditivos

Projeto TCC: Identificação de Serviços de Engenharia em Contratações Públicas
Autor: Caio Serra

────────────────────────────────────────────────────────────────────────────
MOTIVAÇÃO

O contrato original pode ser legitimamente "geral" no início, mas durante
a execução receber TERMOS ADITIVOS (tipoDocumentoId=14 do PNCP) que mudam
o escopo. Exemplo real do TCC:

    "Contrato original: Serviço de pintura predial"  → categoria 8 (geral) ✓
    "Aditivo nº 3: Inclusão de execução de muro de   → ...mas o aditivo
     arrimo de 12 metros, com fundação..."             é OBRA pura.

Sem termos aditivos NÃO temos como pegar essa mutação de escopo. A Camada 3
captura exatamente esses casos.

────────────────────────────────────────────────────────────────────────────
ARQUITETURA

A Camada 3 reaproveita TUDO da Camada 2 — apenas filtra por
`tipoDocumentoId == 14` (Termo Aditivo) ao invés dos tipos de TR/PB/Edital.

Estrutura:
   1. Coleta de aditivos via mesma API /arquivos da Camada 2
   2. Extração de texto (PyMuPDF + pdfplumber + OCR)
   3. Detecção de marcadores de engenharia nos aditivos (ART, RRT, etc.)
   4. ALERTA ESPECIAL: contrato 'geral' com aditivo de engenharia
      = MUDANÇA DE ESCOPO indevida (sinal forte de subenquadramento)
"""

# ════════════════════════════════════════════════════════════════════════════
# Imports — reaproveita Camada 2
# ════════════════════════════════════════════════════════════════════════════
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reaproveita TUDO da Camada 2: API client, extração de texto,
# detecção de marcadores, comparações.
try:
    from pncp_camada2 import (
        # API e descoberta
        API_INTEGRACAO,
        MAPA_TIPO_DOCUMENTO,
        _decompor_numero_controle_pncp,
        listar_documentos_compra,
        listar_documentos_contrato,
        baixar_documento,
        descobrir_documentos,
        # Coleta com cache
        PASTA_CACHE_PDF,
        _path_cache,
        # Extração de texto
        extrair_texto_robusto,
        extrair_textos_em_lote,
        # Marcadores
        MARCADORES_ENGENHARIA,
        detectar_marcadores,
        construir_features_camada2,
        # Helpers
        TEM_PYMUPDF, TEM_PDFPLUMBER, TEM_OCR,
        tqdm,
    )
except ImportError as e:
    raise ImportError(
        f"pncp_camada3.py exige pncp_camada2.py no mesmo diretório.\n"
        f"Erro original: {e}"
    )

try:
    from pncp_analise import (
        _get_com_retry, KEYWORDS_ENG,
        tokenizar, _normalizar,
        _salvar, _anotar_barras,
        PALETA, EM_COLAB,
    )
    if EM_COLAB:
        from IPython.display import Image, display
except ImportError as e:
    raise ImportError(f"pncp_camada3.py exige pncp_analise.py: {e}")

import warnings
import requests

warnings.filterwarnings("ignore")
print("✅ Camada 3 (Termos Aditivos) carregada.")


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C3.1 — Coleta de termos aditivos
# ════════════════════════════════════════════════════════════════════════════

# tipoDocumentoId 14 = Termo Aditivo (manual PNCP §5.12)
TIPO_DOC_TERMO_ADITIVO = 14
TIPO_DOC_TERMO_ADITIVO_NOMES = {
    "termo aditivo", "aditivo", "termo de aditamento",
}


def _eh_aditivo(d: dict) -> bool:
    """
    Identifica se um documento da API é termo aditivo.

    Funciona mesmo quando tipoDocumentoId não vem (PNCP às vezes só envia
    tipoDocumentoNome): faz fallback via nome normalizado.
    """
    tipo_id = d.get("tipoDocumentoId")
    if tipo_id == TIPO_DOC_TERMO_ADITIVO:
        return True
    nome = _normalizar(str(d.get("tipoDocumentoNome", ""))).lower()
    return any(t in nome for t in TIPO_DOC_TERMO_ADITIVO_NOMES)


def coletar_aditivos(df: pd.DataFrame,
                      max_contratos: int = None,
                      apenas_geral: bool = True,
                      pasta_cache: str = "pdfs_aditivos_cache") -> pd.DataFrame:
    """
    Coleta APENAS termos aditivos (tipo 14) anexados aos contratos.

    Por padrão prioriza contratos rotulados 'geral' — o foco do TCC é
    encontrar aditivos com conteúdo de engenharia em contratos que
    deveriam ser puramente "serviço geral".

    Parâmetros
    ──────────
    df            : DataFrame da Camada 1
    max_contratos : limite (None = todos)
    apenas_geral  : se True, só busca aditivos de contratos 'geral'
                     (recomendado — gera o sinal-chave do TCC)
    pasta_cache   : pasta para PDFs baixados

    Retorna
    ───────
    DataFrame com 1 linha por aditivo encontrado.
    """
    Path(pasta_cache).mkdir(exist_ok=True)

    df_use = df.copy()
    if apenas_geral and "rotulo" in df_use.columns:
        df_use = df_use[df_use["rotulo"] == "geral"]
        print(f"   Foco: {len(df_use):,} contratos rotulados 'geral'")
    if max_contratos:
        df_use = df_use.head(max_contratos)

    print(f"\n📄 Buscando termos aditivos em {len(df_use):,} contratos...")

    registros = []
    n_404 = n_ok = n_cache = 0

    for _, row in tqdm(df_use.iterrows(), total=len(df_use),
                        desc="📄 Aditivos"):
        num_ctrl = row["numeroControlePNCP"]
        info = _decompor_numero_controle_pncp(num_ctrl)
        if info is None or info["tipo"] not in (1, 2):
            continue

        if info["tipo"] == 1:
            docs = listar_documentos_compra(info["cnpj"], info["ano"], info["sequencial"])
            recurso = "compras"
        else:
            docs = listar_documentos_contrato(info["cnpj"], info["ano"], info["sequencial"])
            recurso = "contratos"

        if not docs:
            n_404 += 1
            continue

        # Filtra apenas aditivos
        aditivos = [d for d in docs if _eh_aditivo(d)]
        if not aditivos:
            continue

        for d in aditivos:
            seq_doc = d.get("sequencialDocumento")
            if seq_doc is None:
                continue

            # Cache por (numeroControlePNCP, seq_doc) — reutiliza o helper da C2
            safe = num_ctrl.replace("/", "_")
            caminho = Path(pasta_cache) / f"{safe}__aditivo_{seq_doc}.pdf"

            if caminho.exists() and caminho.stat().st_size > 0:
                n_cache += 1
            else:
                conteudo = baixar_documento(
                    info["cnpj"], info["ano"], info["sequencial"],
                    seq_doc, recurso
                )
                if conteudo is None:
                    continue
                caminho.write_bytes(conteudo)
                n_ok += 1
                time.sleep(0.4)

            registros.append({
                "numeroControlePNCP":  num_ctrl,
                "rotulo":              row.get("rotulo"),
                "objeto_original":     row.get("objeto", "")[:200],
                "valor_original":      row.get("valorTotalEstimado"),
                "sequencialDocumento": seq_doc,
                "tipoDocumentoId":     d.get("tipoDocumentoId", TIPO_DOC_TERMO_ADITIVO),
                "tipoDocumentoNome":   d.get("tipoDocumentoNome", "Termo Aditivo"),
                "titulo":              d.get("titulo", "")[:200],
                "dataPublicacaoPncp":  d.get("dataPublicacaoPncp", ""),
                "caminho_local":       str(caminho),
                "tamanho_bytes":       caminho.stat().st_size if caminho.exists() else 0,
            })

    print(f"\n✅ Coleta de aditivos:")
    print(f"   • Downloads:  {n_ok:,}")
    print(f"   • Cache:      {n_cache:,}")
    print(f"   • Sem PDF:    {n_404:,}")
    print(f"   • Aditivos:   {len(registros):,}")

    return pd.DataFrame(registros)


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C3.2 — Análise de mudança de escopo
# ════════════════════════════════════════════════════════════════════════════

def analisar_mudanca_escopo(df_aditivos: pd.DataFrame,
                              df_contratos: pd.DataFrame,
                              pasta_saida: str) -> pd.DataFrame:
    """
    Análise central da Camada 3: detecta mudança de escopo INDEVIDA.

    Critério de mudança de escopo suspeita:
    • Contrato original rotulado 'geral'
    • Aditivo contém marcadores de engenharia (ART, RRT, "obra", etc.)
    → o aditivo introduziu serviço de engenharia em contrato que era geral.

    Esse é o caso mais SUSPEITO porque:
    • A licitação original NÃO seguiu rito de engenharia (era "geral")
    • Mas a execução acabou incluindo trabalho de engenharia
    • Indica burla do rito de licitação

    Para o TCC, esses casos são "subenquadramento confirmado pela execução".
    """
    if df_aditivos.empty:
        print("   [pulado] sem aditivos coletados.")
        return pd.DataFrame()

    # Junta com objeto + rótulo do contrato original
    cols_orig = ["numeroControlePNCP", "objeto", "rotulo",
                  "valorTotalEstimado", "razaoSocialOrgao",
                  "nomeRazaoSocialFornecedor"]
    cols_orig = [c for c in cols_orig if c in df_contratos.columns]
    df_join = df_aditivos.merge(
        df_contratos[cols_orig].drop_duplicates("numeroControlePNCP"),
        on="numeroControlePNCP", how="left", suffixes=("_aditivo", "")
    )

    # Marca contratos suspeitos (geral + aditivo com sinal de engenharia)
    df_join["mudanca_escopo_suspeita"] = (
        (df_join["rotulo"] == "geral") &
        (df_join.get("mk_score_engenharia", 0) >= 2)
    )

    n_suspeitos = df_join["mudanca_escopo_suspeita"].sum()
    n_total_geral = (df_join["rotulo"] == "geral").sum()

    print(f"\n── ANÁLISE DE MUDANÇA DE ESCOPO ──")
    print(f"   Total de aditivos analisados:        {len(df_join):,}")
    print(f"   Aditivos em contratos 'geral':       {n_total_geral:,}")
    print(f"   Mudança de escopo suspeita:          {n_suspeitos:,}  "
          f"({n_suspeitos/max(n_total_geral,1)*100:.1f}% dos 'geral' com aditivo)")
    print(f"\n   Interpretação: contratos rotulados 'geral' que receberam")
    print(f"   aditivo com marcadores de engenharia — indica que o escopo")
    print(f"   real envolveu engenharia (subenquadramento confirmado pela")
    print(f"   EXECUÇÃO, mesmo se a licitação original parecia legítima).")

    if n_suspeitos > 0:
        print(f"\n── Top-15 mudanças de escopo suspeitas ──")
        susp = df_join[df_join["mudanca_escopo_suspeita"]].sort_values(
            "mk_score_engenharia", ascending=False
        )
        cols_show = [c for c in [
            "numeroControlePNCP", "objeto", "razaoSocialOrgao",
            "nomeRazaoSocialFornecedor",
            "valor_original", "mk_score_engenharia"
        ] if c in susp.columns]
        print(susp[cols_show].head(15).to_string(index=False))

        arq = os.path.join(pasta_saida, "c3_mudanca_escopo_suspeita.csv")
        susp.to_csv(arq, index=False, encoding="utf-8-sig")
        print(f"\n   💾 {arq}")

    # Gráfico
    if len(df_join) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Painel 1: marcadores por rótulo no aditivo
        cols_pres = [c for c in df_join.columns if c.endswith("_presente")]
        if cols_pres:
            res = df_join.groupby("rotulo")[cols_pres].mean().mul(100).round(1)
            res.columns = [c.replace("mk_", "").replace("_presente", "")
                            for c in res.columns]
            res.T.plot(kind="bar", ax=axes[0],
                        color=[PALETA.get(c, "#aaa") for c in res.index],
                        edgecolor="white")
            axes[0].set_title("Marcadores nos ADITIVOS, por rótulo do contrato",
                                fontweight="bold")
            axes[0].set_xlabel("Marcador"); axes[0].set_ylabel("% aditivos")
            axes[0].legend(title="Rótulo do contrato")
            axes[0].tick_params(axis="x", rotation=30)
            sns.despine(ax=axes[0])

        # Painel 2: distribuição de score nos aditivos
        for rot, cor in PALETA.items():
            sub = df_join[df_join["rotulo"] == rot]["mk_score_engenharia"].dropna()
            if len(sub) == 0: continue
            axes[1].hist(sub, bins=10, alpha=0.7, label=rot.capitalize(),
                          color=cor, edgecolor="white")
        axes[1].set_title("Score de engenharia nos ADITIVOS",
                            fontweight="bold")
        axes[1].set_xlabel("Marcadores de eng. encontrados (0-9)")
        axes[1].set_ylabel("Nº aditivos")
        axes[1].legend()
        sns.despine(ax=axes[1])

        fig.tight_layout()
        _salvar(fig, "c3_01_aditivos_mudanca_escopo.png", pasta_saida)

    return df_join[df_join["mudanca_escopo_suspeita"]]


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C3.3 — Orquestrador da Camada 3
# ════════════════════════════════════════════════════════════════════════════

def executar_camada3(df: pd.DataFrame,
                      pasta_saida: str = None,
                      max_contratos: int = 200,
                      apenas_geral: bool = True) -> dict:
    """
    Pipeline completo da Camada 3 — Termos Aditivos.

    1. Coleta termos aditivos dos contratos (foca em 'geral' por padrão)
    2. Extrai texto dos PDFs (mesmo método da Camada 2)
    3. Detecta marcadores de engenharia (ART, RRT, "obra", etc.)
    4. Identifica mudança de escopo suspeita: 'geral' + aditivo de eng.

    Parâmetros
    ──────────
    df             : DataFrame da Camada 1
    pasta_saida    : pasta para gráficos/CSVs
    max_contratos  : limite de contratos a processar
    apenas_geral   : foco em contratos 'geral' (recomendado)
    """
    print("\n" + "█"*62)
    print("  CAMADA 3 — TERMOS ADITIVOS")
    print("█"*62)

    if pasta_saida is None:
        uf  = df["ufSigla"].mode()[0]       if "ufSigla" in df.columns else "xx"
        ano = (df["anoPublicacao"].mode()[0] if "anoPublicacao" in df.columns
               else "xxxx")
        pasta_saida = f"graficos_pncp_{uf}_{ano}"
    os.makedirs(pasta_saida, exist_ok=True)

    # 1. Coleta
    print("\n[1] Coletando termos aditivos...")
    df_aditivos = coletar_aditivos(
        df, max_contratos=max_contratos,
        apenas_geral=apenas_geral
    )
    if df_aditivos.empty:
        print("\n⚠ Nenhum aditivo coletado.")
        return {"df_aditivos": df_aditivos}

    # 2. Extração de texto
    print("\n[2] Extraindo texto dos aditivos...")
    df_aditivos = extrair_textos_em_lote(df_aditivos, usar_ocr_se_vazio=True)

    # 3. Detecção de marcadores
    print("\n[3] Detectando marcadores de engenharia...")
    df_aditivos = construir_features_camada2(df_aditivos)

    arq = os.path.join(pasta_saida, "c3_aditivos_completos.parquet")
    df_aditivos.to_parquet(arq, index=False)
    print(f"   💾 {arq}")

    # 4. Análise de mudança de escopo
    print("\n[4] Analisando mudança de escopo suspeita...")
    df_suspeitos = analisar_mudanca_escopo(df_aditivos, df, pasta_saida)

    print("\n" + "█"*62)
    print(f"  CAMADA 3 ✅  {len(df_aditivos):,} aditivos | "
          f"{len(df_suspeitos):,} mudanças de escopo suspeitas")
    print("█"*62)

    return {
        "df_aditivos":           df_aditivos,
        "df_suspeitos":          df_suspeitos,
    }


# ════════════════════════════════════════════════════════════════════════════
# REFERÊNCIA RÁPIDA
# ════════════════════════════════════════════════════════════════════════════
#
# Pré-requisitos: já ter rodado a Camada 1 e ter pncp_camada2.py + pncp_analise.py
#
# Pipeline completo:
#   from pncp_camada3 import executar_camada3
#   c3 = executar_camada3(df, max_contratos=200, apenas_geral=True)
#
# Resultados:
#   c3["df_aditivos"]   → todos os aditivos com texto e marcadores
#   c3["df_suspeitos"]  → contratos 'geral' com aditivo de engenharia
#                          (mudança de escopo confirmada pela execução)
