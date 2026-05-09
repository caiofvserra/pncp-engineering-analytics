"""
Exploratory Data Analysis (EDA).

Gera gráficos inline no Colab/Jupyter (não só em disco — corrige o
problema de "não vi gráfico nenhum"). Os PNGs ficam salvos em
dados/eda/ para incluir no TCC.

Inspirado no pncp_analise.py original (23+ gráficos), mantém os mais
importantes para análise descritiva e detecção de viés.
"""

from collections import Counter
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from pncp import config
from pncp._plot import salvar_e_mostrar
from pncp.io_disco import ler_parquet, salvar_json
from pncp.ram import liberar


# ── 1. Distribuição de rótulos ───────────────────────────────────────────────
def g_distribuicao_rotulos(df, pasta):
    """Quantos contratos por rótulo, com anotações nas barras."""
    ct = df["rotulo"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cores = {"engenharia": "#1f77b4", "obras": "#2ca02c", "geral": "#ff7f0e"}
    bars = ax.bar(ct.index, ct.values,
                   color=[cores.get(r, "#888") for r in ct.index])
    for b, v in zip(bars, ct.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Distribuição de contratos por rótulo (Lei 14.133/2021)")
    ax.set_ylabel("nº contratos")
    return salvar_e_mostrar(fig, pasta / "01_distribuicao_rotulos.png")


# ── 2. Série temporal por rótulo ─────────────────────────────────────────────
def g_serie_temporal(df, pasta):
    if "anoPublicacao" not in df.columns:
        return None
    df = df.dropna(subset=["anoPublicacao"]).copy()
    df["anoPublicacao"] = df["anoPublicacao"].astype(int)
    if "mesPublicacao" in df.columns:
        df = df.dropna(subset=["mesPublicacao"]).copy()
        df["periodo"] = (df["anoPublicacao"].astype(str) + "-"
                          + df["mesPublicacao"].astype(int).astype(str).str.zfill(2))
    else:
        df["periodo"] = df["anoPublicacao"].astype(str)
    cross = (df.groupby(["periodo", "rotulo"]).size()
             .unstack(fill_value=0).sort_index())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cross.plot(ax=ax, marker="o", markersize=4, linewidth=1.5)
    ax.set_title("Evolução de contratos por rótulo")
    ax.set_xlabel("período")
    ax.set_ylabel("nº contratos")
    plt.xticks(rotation=45)
    ax.legend(title="rótulo", loc="best")
    return salvar_e_mostrar(fig, pasta / "02_serie_temporal.png")


# ── 3. Boxplot de valor por rótulo ───────────────────────────────────────────
def g_boxplot_valor(df, pasta):
    if "valor" not in df.columns:
        return None
    df_v = df[df["valor"] > 0].copy()
    if df_v.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Esquerda: linear, com outliers
    axes[0].set_yscale("symlog")
    df_v.boxplot(column="valor", by="rotulo", ax=axes[0])
    axes[0].set_title("Valor (escala log)")
    axes[0].set_ylabel("R$")
    plt.suptitle("")
    # Direita: até percentil 95 (zoom)
    p95 = df_v["valor"].quantile(0.95)
    df_v[df_v["valor"] <= p95].boxplot(column="valor", by="rotulo", ax=axes[1])
    axes[1].set_title(f"Valor até P95 (R$ {p95:,.0f})")
    return salvar_e_mostrar(fig, pasta / "03_boxplot_valor.png")


# ── 4. Top municípios ────────────────────────────────────────────────────────
def g_top_municipios(df, pasta, top_n=15):
    if "municipioNome" not in df.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Total
    top_total = df["municipioNome"].value_counts().head(top_n)
    top_total.iloc[::-1].plot.barh(ax=axes[0], color="#1f77b4")
    axes[0].set_title(f"Top {top_n} municípios — total")
    # Engenharia
    eng = df[df["rotulo"] == "engenharia"]
    if not eng.empty:
        top_eng = eng["municipioNome"].value_counts().head(top_n)
        top_eng.iloc[::-1].plot.barh(ax=axes[1], color="#2ca02c")
        axes[1].set_title(f"Top {top_n} municípios — engenharia")
    return salvar_e_mostrar(fig, pasta / "04_top_municipios.png")


# ── 5. Top órgãos ────────────────────────────────────────────────────────────
def g_top_orgaos(df, pasta, top_n=15):
    col = "razaoSocialOrgao" if "razaoSocialOrgao" in df.columns \
          else "nomeUnidade" if "nomeUnidade" in df.columns else None
    if col is None:
        return None
    top = df[col].dropna().value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(11, 6))
    top.iloc[::-1].plot.barh(ax=ax, color="#9467bd")
    ax.set_title(f"Top {top_n} órgãos contratantes")
    ax.set_xlabel("nº contratos")
    return salvar_e_mostrar(fig, pasta / "05_top_orgaos.png")


# ── 6. Comprimento do objeto ─────────────────────────────────────────────────
def g_comprimento_objeto(df, pasta):
    if "objeto" not in df.columns:
        return None
    df = df.copy()
    df["len_obj"] = df["objeto"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for rot, sub in df.groupby("rotulo"):
        if sub.empty:
            continue
        ax.hist(sub["len_obj"].clip(upper=2000), bins=40, alpha=0.5,
                label=str(rot))
    ax.set_xlabel("comprimento do objeto (caracteres, capped 2000)")
    ax.set_ylabel("frequência")
    ax.set_title("Comprimento do objeto por rótulo")
    ax.legend()
    return salvar_e_mostrar(fig, pasta / "06_comprimento_objeto.png")


# ── 7. Frequência de palavras (top-30) ───────────────────────────────────────
_RX_TOK = re.compile(r"[a-zA-ZÀ-ÿ]+")


def _tokens(texto):
    return [t.lower() for t in _RX_TOK.findall(str(texto)) if len(t) >= 3]


def g_frequencia_palavras(df, pasta, top_n=30, max_amostra=50_000):
    if "objeto_limpo" in df.columns:
        col = "objeto_limpo"
    elif "objeto" in df.columns:
        col = "objeto"
    else:
        return None
    sample = df[col].dropna().sample(n=min(max_amostra, len(df)),
                                       random_state=config.SEED)
    contador = Counter()
    for txt in sample:
        contador.update(_tokens(txt))
    mais = pd.Series(dict(contador.most_common(top_n)))
    fig, ax = plt.subplots(figsize=(9, 8))
    mais.iloc[::-1].plot.barh(ax=ax, color="#17becf")
    ax.set_title(f"Top {top_n} palavras (amostra de {len(sample):,})")
    ax.set_xlabel("frequência")
    return salvar_e_mostrar(fig, pasta / "07_frequencia_palavras.png")


# ── 8. Bigramas mais frequentes ──────────────────────────────────────────────
def g_bigramas(df, pasta, top_n=20, max_amostra=50_000):
    col = "objeto_limpo" if "objeto_limpo" in df.columns else "objeto"
    if col not in df.columns:
        return None
    sample = df[col].dropna().sample(n=min(max_amostra, len(df)),
                                       random_state=config.SEED)
    contador = Counter()
    for txt in sample:
        toks = _tokens(txt)
        contador.update(zip(toks, toks[1:]))
    mais = pd.Series({" ".join(k): v
                      for k, v in contador.most_common(top_n)})
    fig, ax = plt.subplots(figsize=(9, 8))
    mais.iloc[::-1].plot.barh(ax=ax, color="#bcbd22")
    ax.set_title(f"Top {top_n} bigramas (amostra de {len(sample):,})")
    return salvar_e_mostrar(fig, pasta / "08_bigramas.png")


# ── 9. Termos de engenharia em "geral" (sinal de subenquadramento) ──────────
def g_termos_eng_em_geral(df, pasta):
    """
    Contagem de contratos rotulados 'geral' que contêm termos típicos
    de engenharia. Cada barra alta = sinal de potencial subenquadramento.
    """
    if "n_termos_eng" not in df.columns:
        return None
    geral = df[df["rotulo"] == "geral"].copy()
    if geral.empty:
        return None
    contagens = (geral["n_termos_eng"].clip(upper=10)
                 .value_counts().sort_index())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(contagens.index.astype(str), contagens.values, color="#d62728")
    for b, v in zip(bars, contagens.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Contratos 'geral' por nº de termos de engenharia no objeto")
    ax.set_xlabel("nº de termos de engenharia (capped em 10)")
    ax.set_ylabel("nº contratos")
    return salvar_e_mostrar(fig, pasta / "09_termos_eng_em_geral.png")


# ── 10. Esfera de governo ────────────────────────────────────────────────────
def g_por_esfera(df, pasta):
    if "esferaNome" not in df.columns:
        return None
    cross = (df.groupby(["esferaNome", "rotulo"]).size()
             .unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cross.plot.bar(ax=ax, stacked=False)
    ax.set_title("Contratos por esfera de governo × rótulo")
    ax.set_xlabel("esfera")
    plt.xticks(rotation=0)
    return salvar_e_mostrar(fig, pasta / "10_por_esfera.png")


# ── Pipeline ─────────────────────────────────────────────────────────────────
def executar(caminho_parquet=None, mostrar_inline=True):
    """
    Lê o parquet de coleta, gera gráficos e relatorio.json.
    Retorna dict com paths salvos. Gráficos são EXIBIDOS inline em Colab.
    """
    from pncp.ram import precisa_de
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")
    if not precisa_de(caminho_parquet, "eda",
                       "rode pncp.coleta.coletar(...) primeiro"):
        return None

    df = ler_parquet(caminho_parquet)
    if df.empty:
        print("[eda] parquet vazio — nada a fazer")
        return None
    print(f"[eda] {len(df):,} contratos")

    pasta = config.caminho(config.SUB_EDA, ".").parent

    def _safe_dict(serie):
        return {str(k): int(v) for k, v in serie.items()}

    relatorio = {
        "n_contratos": int(len(df)),
        "rotulos": _safe_dict(df["rotulo"].value_counts()),
        "anos": (_safe_dict(df["anoPublicacao"].dropna().astype(int)
                              .value_counts().sort_index())
                 if "anoPublicacao" in df.columns else {}),
        "valor_total_estimado": (float(df["valor"].sum())
                                 if "valor" in df.columns else None),
        "graficos": {},
    }

    # Gera todos os gráficos
    for nome, fn in [
        ("distribuicao_rotulos", g_distribuicao_rotulos),
        ("serie_temporal", g_serie_temporal),
        ("boxplot_valor", g_boxplot_valor),
        ("top_municipios", g_top_municipios),
        ("top_orgaos", g_top_orgaos),
        ("comprimento_objeto", g_comprimento_objeto),
        ("frequencia_palavras", g_frequencia_palavras),
        ("bigramas", g_bigramas),
        ("termos_eng_em_geral", g_termos_eng_em_geral),
        ("por_esfera", g_por_esfera),
    ]:
        try:
            p = fn(df, pasta)
            if p is not None:
                relatorio["graficos"][nome] = str(p)
        except Exception as e:
            print(f"[eda] {nome} falhou: {type(e).__name__}: {e}")

    # Detecta viés temporal
    if "anoPublicacao" in df.columns and df["anoPublicacao"].notna().any():
        df_tmp = df[df["anoPublicacao"].notna()].copy()
        df_tmp["anoPublicacao"] = df_tmp["anoPublicacao"].astype(int)
        por_ano = df_tmp.groupby("anoPublicacao")["rotulo"] \
                         .value_counts(normalize=True)
        suspeitos = por_ano[por_ano.index.get_level_values("rotulo") == "geral"]
        suspeitos = suspeitos[suspeitos > 0.7]
        if len(suspeitos):
            relatorio["alerta_temporal"] = {
                "anos_suspeitos": [int(a) for a, _ in suspeitos.index],
                "mensagem": ("Anos com >70% de 'geral' podem indicar mudança "
                             "de critério — considerar filtrar."),
            }

    saida = salvar_json(relatorio, config.caminho(config.SUB_EDA, "relatorio.json"))
    print(f"\n[eda] {len(relatorio['graficos'])} gráfico(s) gerados em "
          f"{pasta}/{config.SUB_EDA}/")
    print(f"[eda] relatório JSON em {saida}")
    liberar(df)
    return relatorio


def mostrar():
    """Re-exibe os gráficos do EDA já gerados (use no Colab)."""
    try:
        from IPython.display import Image, display
    except ImportError:
        print("[eda.mostrar] IPython não disponível")
        return
    from pncp.io_disco import ler_json
    rel_path = config.caminho(config.SUB_EDA, "relatorio.json")
    if not rel_path.exists():
        print("[eda.mostrar] rode pncp.eda.executar() primeiro")
        return
    rel = ler_json(rel_path)
    print(f"\n📊 EDA — {rel['n_contratos']:,} contratos")
    print(f"   rótulos: {rel.get('rotulos')}")
    print(f"   anos: {rel.get('anos')}")
    if "alerta_temporal" in rel:
        print(f"   ⚠ {rel['alerta_temporal']['mensagem']}")
    for nome, p in (rel.get("graficos") or {}).items():
        if Path(p).exists():
            print(f"\n— {nome} —")
            display(Image(p))
