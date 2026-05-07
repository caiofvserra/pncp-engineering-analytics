"""
Exploratory Data Analysis (EDA).

Salva PNGs em dados/eda/ e um relatorio.json com as estatísticas.
Não retorna objetos pesados — tudo vai para disco.
"""

from collections import Counter

import matplotlib
matplotlib.use("Agg")  # backend sem display, evita travar Colab
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from pncp import config
from pncp.io_disco import ler_parquet, salvar_json
from pncp.ram import liberar


def _salvar_fig(fig, nome):
    saida = config.caminho(config.SUB_EDA, nome)
    fig.tight_layout()
    fig.savefig(saida, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return saida


def _grafico_distribuicao(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    df["rotulo"].value_counts().plot.bar(ax=ax)
    ax.set_title("Contratos por rótulo (Lei 14.133/2021)")
    ax.set_ylabel("nº contratos")
    return _salvar_fig(fig, "01_distribuicao_rotulos.png")


def _grafico_temporal(df):
    if "anoPublicacao" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    cross = df.groupby(["anoPublicacao", "rotulo"]).size().unstack(fill_value=0)
    cross.plot(ax=ax, marker="o")
    ax.set_title("Evolução por rótulo ao longo dos anos")
    ax.set_ylabel("nº contratos")
    return _salvar_fig(fig, "02_temporal.png")


def _grafico_termos(df, n=20):
    if "objeto_limpo" not in df.columns:
        return None
    contador = Counter()
    # Sample para não estourar RAM em 300k linhas
    sample = df["objeto_limpo"].dropna().sample(
        n=min(50_000, len(df)), random_state=config.SEED
    )
    for txt in sample:
        contador.update(txt.split())
    mais = pd.Series(dict(contador.most_common(n)))
    fig, ax = plt.subplots(figsize=(7, 6))
    mais.iloc[::-1].plot.barh(ax=ax)
    ax.set_title(f"Top {n} termos no objeto")
    return _salvar_fig(fig, "03_top_termos.png")


def executar(caminho_parquet=None, mostrar_inline=True):
    """
    Lê o parquet de coleta, gera gráficos e relatorio.json.
    Retorna dict com paths salvos. Se mostrar_inline=True, renderiza no Colab.
    """
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")

    df = ler_parquet(caminho_parquet)
    print(f"[eda] {len(df):,} contratos")

    # Converte chaves para tipos JSON-safe (Int16/Int64 nullable não serializa
    # como chave de dict — converte para str ou int padrão).
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
    relatorio["graficos"]["distribuicao"] = str(_grafico_distribuicao(df))
    g_tmp = _grafico_temporal(df)
    if g_tmp:
        relatorio["graficos"]["temporal"] = str(g_tmp)
    g_tr = _grafico_termos(df)
    if g_tr:
        relatorio["graficos"]["termos"] = str(g_tr)

    # Detecta viés temporal: se há ano com >70% de "geral", recomenda filtro
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
                             "de critério de rotulação — considerar filtrar."),
            }

    saida = salvar_json(relatorio, config.caminho(config.SUB_EDA, "relatorio.json"))
    print(f"[eda] relatório em {saida}")
    liberar(df)
    if mostrar_inline:
        mostrar()
    return relatorio


def mostrar():
    """Renderiza inline os PNGs e o JSON resumo do EDA (use no Colab)."""
    try:
        from IPython.display import Image, display
    except ImportError:
        Image = display = None

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
    if display is None:
        print("   (IPython não disponível — gráficos ficam só em disco)")
        return
    for nome, p in (rel.get("graficos") or {}).items():
        if Path(p).exists():
            print(f"\n— {nome} —")
            display(Image(p))
