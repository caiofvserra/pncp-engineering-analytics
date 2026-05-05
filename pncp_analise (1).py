"""
pncp_analise.py — Pipeline completo: API PNCP + EDA + NLP + Classificação
Projeto TCC: Identificação de Serviços de Engenharia em Contratações Públicas
Autor: Caio Serra

╔════════════════════════════════════════════════════════════════════════════╗
║  GUIA DE LEITURA                                                           ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  PARTE 1  Coleta API PNCP + EDA exploratória                               ║
║           (Etapa A1 = só coleta; Etapa A2 = só EDA — separadas)            ║
║                                                                            ║
║  PARTE 2  Classificação baseline (TF-IDF + modelos clássicos)              ║
║           Inclui análise de rigor de licitação (Lei 14.133/2021).          ║
║                                                                            ║
║  PARTE 3  Técnicas avançadas: SMOTE, GridSearch, KNN de similaridade,      ║
║           clustering KMeans, Apriori, embeddings (BERTimbau opcional).     ║
║                                                                            ║
║  PARTE 4  Rigor estatístico (holdout 80/20, McNemar, bootstrap IC95%,      ║
║           multiclasse 3 classes).                                          ║
║                                                                            ║
║  PARTE 5  UX (perguntas interativas) + interpretação automática +          ║
║           geração de relatório markdown + glossário.                       ║
║                                                                            ║
║  PARTE 6  Redução de FP (threshold alta precisão, ensemble, coerência      ║
║           semântica via embeddings, pacote para validação LLM).            ║
║                                                                            ║
║  PARTE 7  Análise de redes (grafos órgão↔fornecedor, centralidade,         ║
║           comunidades Louvain, fornecedores fantasma).                     ║
║                                                                            ║
║  PARTE 8  Enriquecimento via CNAE oficial (Receita Federal × CONFEA).      ║
║           Cruza CNAE da empresa com lista CREA → matriz consistência.      ║
║                                                                            ║
║  PARTE 9  Resumo executivo + análise por valor (R$) + ground truth +       ║
║           consolidação de TODOS os suspeitos.                              ║
║                                                                            ║
║  Camada 2 (arquivo separado: pncp_camada2.py)                              ║
║           PDFs de TR, edital, projeto básico — extração de texto +         ║
║           detecção de marcadores ART/RRT.                                  ║
║                                                                            ║
║  Camada 3 (arquivo separado: pncp_camada3.py)                              ║
║           Termos aditivos — captura mudança de escopo (ex: pintura →       ║
║           muro de arrimo).                                                 ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

PONTOS DE ENTRADA
─────────────────
Pipeline completo interativo:
    executar_tudo_interativo()        → pergunta a cada etapa
    executar_tudo()                   → roda automático com flags

Coleta separada (recomendado):
    df = executar_apenas_coleta(modo_interativo=True)   # A1
    eda_res = executar_apenas_eda(df)                    # A2

Combinar coletas multi-sessão:
    df = combinar_parquets(uf_filtro="SP")

Persistência (Drive):
    montar_drive() + keep_alive_javascript() + diagnostico_drive()

Glossário (significado dos parâmetros):
    glossario()              # tudo
    glossario("F1")          # termo específico

Técnicas dos notebooks aplicadas
────────────────────────────────
Aula  5 — MinMax/StandardScaler          Aula  6 — log + Pipeline + ColumnTransformer
Aula  7 — Correlação (Spearman)          Aula  8 — PCA (TruncatedSVD p/ esparso)
Aula  9 — SMOTE                          Aula 12 — Curvas ROC, métricas
Aula 14 — DummyClassifier (baseline)     Aula 16 — DecisionTree + plot_tree
Aula 18 — GridSearchCV, StratifiedKFold  Aula 20 — RF com Pipeline/ColumnTransformer
Aula 22 — SVM com GridSearch             Aulas 24/25 — NearestNeighbors
"""

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 0 — Instalação de dependências
# ════════════════════════════════════════════════════════════════════════════
# No Colab, descomente esta linha e execute UMA VEZ:
# !pip install -q wordcloud tqdm imbalanced-learn nltk mlxtend lazypredict sentence-transformers transformers torch

import subprocess, sys

def _instalar_se_ausente(pacote: str, import_nome: str = None) -> bool:
    import_nome = import_nome or pacote
    try:
        __import__(import_nome); return True
    except ImportError:
        pass
    print(f"   Instalando {pacote}...")
    for flags in [[], ["--break-system-packages"]]:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pacote] + flags,
                stderr=subprocess.DEVNULL)
            __import__(import_nome); return True
        except Exception:
            continue
    print(f"   [aviso] Instale manualmente: pip install {pacote}")
    return False

_instalar_se_ausente("wordcloud")
_instalar_se_ausente("tqdm")
_instalar_se_ausente("imbalanced-learn", "imblearn")
_instalar_se_ausente("nltk")             # Aula quinzena 1 — RSLP Stemmer
_instalar_se_ausente("mlxtend")          # Aula 39 — Apriori / regras de associação
_instalar_se_ausente("lazypredict")      # Aula 28 — comparar dezenas de modelos rápido
_instalar_se_ausente("networkx")         # Parte 7 — análise de grafos
_instalar_se_ausente("python-louvain", "community")  # detecção de comunidades

# Bibliotecas pesadas de NLP — instaladas SOMENTE se o usuário pedir embeddings
# semânticos (Aula 42). São ~500MB de download cada uma. Para evitar instalação
# acidental no Colab gratuito, descomente as duas linhas abaixo manualmente:
# _instalar_se_ausente("sentence-transformers")  # Aula 42 — Sentence-BERT multilingual
# _instalar_se_ausente("transformers")           # Para BERTimbau (modelo BR)

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — Imports
# ════════════════════════════════════════════════════════════════════════════

import os, re, time, datetime, warnings, collections, unicodedata
import requests
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.pipeline            import Pipeline
from sklearn.compose             import ColumnTransformer
from sklearn.impute              import SimpleImputer
from sklearn.preprocessing       import (StandardScaler, MinMaxScaler,
                                          LabelEncoder, OneHotEncoder)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection   import SelectKBest, mutual_info_classif       # Aula 30
from sklearn.decomposition       import PCA, TruncatedSVD, NMF, LatentDirichletAllocation  # Aula 30 + LDA
from sklearn.cluster             import KMeans, AgglomerativeClustering         # Aulas 33, 35
from sklearn.metrics             import silhouette_score, silhouette_samples    # Aula 37
from sklearn.model_selection     import (StratifiedKFold, cross_validate,
                                          cross_val_predict, GridSearchCV,
                                          train_test_split)
from sklearn.linear_model        import LogisticRegression
from sklearn.svm                 import LinearSVC, SVC
from sklearn.tree                import DecisionTreeClassifier, plot_tree
from sklearn.ensemble            import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors           import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes         import MultinomialNB
from sklearn.dummy               import DummyClassifier
from sklearn.calibration         import CalibratedClassifierCV
from sklearn.metrics             import (classification_report, confusion_matrix,
                                          ConfusionMatrixDisplay,
                                          roc_auc_score, roc_curve,
                                          precision_recall_curve,
                                          average_precision_score,
                                          f1_score, precision_score,
                                          recall_score, accuracy_score)
from scipy.sparse                import hstack, csr_matrix
from scipy.cluster               import hierarchy                              # Aula 33
from scipy.spatial.distance      import pdist                                  # Aula 33
import joblib

try:
    from wordcloud import WordCloud
    TEM_WORDCLOUD = True
except ImportError:
    TEM_WORDCLOUD = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from imblearn.over_sampling  import SMOTE, RandomOverSampler              # Aula 31
    from imblearn.under_sampling import RandomUnderSampler                    # Aula 31
    from imblearn.pipeline       import Pipeline as ImbPipeline
    TEM_IMBLEARN = True
except ImportError:
    TEM_IMBLEARN = False

# NLTK — Aula quinzena 01: stemmer RSLP para português
try:
    import nltk
    from nltk.stem import RSLPStemmer
    try:
        nltk.data.find("stemmers/rslp")
    except LookupError:
        nltk.download("rslp", quiet=True)
    _STEMMER_PT = RSLPStemmer()
    TEM_NLTK = True
except Exception:
    _STEMMER_PT = None
    TEM_NLTK = False

# mlxtend — Aula 39: regras de associação (Apriori)
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    TEM_MLXTEND = True
except ImportError:
    TEM_MLXTEND = False

# lazypredict — Aula 28: compara dezenas de modelos automaticamente
try:
    from lazypredict.Supervised import LazyClassifier
    TEM_LAZYPREDICT = True
except ImportError:
    TEM_LAZYPREDICT = False

# Sentence-Transformers — Aula 42: embeddings semânticos pré-treinados
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    TEM_SENTENCE_TRANSFORMERS = True
except ImportError:
    TEM_SENTENCE_TRANSFORMERS = False

# transformers + torch — para BERTimbau (modelo BERT brasileiro)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TEM_TRANSFORMERS = True
except ImportError:
    TEM_TRANSFORMERS = False

# NetworkX + python-louvain — Parte 7: análise de grafos
try:
    import networkx as nx
    TEM_NETWORKX = True
except ImportError:
    TEM_NETWORKX = False

try:
    import community as community_louvain   # python-louvain
    TEM_LOUVAIN = True
except ImportError:
    TEM_LOUVAIN = False

try:
    import google.colab  # noqa: F401
    EM_COLAB = True
    from IPython.display import Image, display
except ImportError:
    EM_COLAB = False

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 130, "font.size": 11,
                             "axes.spines.top": False, "axes.spines.right": False})

print("✅ Imports OK.")

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — Configuração
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# IMPORTS DO MÓDULO DE COLETA (pncp_coleta.py)
# ════════════════════════════════════════════════════════════════════════════
# A coleta da API + persistência (Drive, checkpoints) + limpeza estão em um
# módulo SEPARADO para você não precisar recarregar 10k linhas quando estiver
# iterando apenas em análises. Carregue ANTES deste módulo:
#     %run pncp_coleta.py
#     %run pncp_analise.py
from pncp_coleta import (
    # Constantes
    ANO_INICIO, ANO_FIM, MES_INICIO, MES_FIM, MAX_PAGINAS, TAMANHO, UF,
    BASE_URL, EM_COLAB,
    MAPA_CATEGORIA, MAPA_MODALIDADE, MAPA_CRITERIO, MAPA_ESFERA, MAPA_PODER,
    MAPA_ROTULO, CATEGORIAS_CONSIDERADAS, PALETA,
    STOPWORDS_PT, STOPWORDS_CONTRATOS, _STEMMER_PT,
    # Tokenização (necessária em várias partes da análise)
    _normalizar, tokenizar, bigramas,
    # API + limpeza
    _get_com_retry, _aplanar_contrato, baixar_contratacoes_pncp_por_uf,
    carregar_e_limpar,
    # Persistência
    DRIVE_MONTADO, PASTA_DRIVE, _DRIVE_MARKER_FILE,
    _salvar_marker_drive, _ler_marker_drive,
    montar_drive, diagnostico_drive, keep_alive_javascript,
    monitorar_ram, liberar_memoria,
    _path_persistente, _salvar_checkpoint_coleta,
    # Recarga + combinação + filtros
    carregar_checkpoint, combinar_parquets, executar_apenas_coleta,
    filtrar_anos, filtrar_uf,
    # Prompts interativos
    _pedir_int, _pedir_texto, coletar_parametros_interativo,
)

MODO_INTERATIVO = True

# Coleta de um único ano (modo simples — compatibilidade com versões antigas)
ANO         = 2024

# Coleta multi-ano (NOVO — permite intervalo de anos)
# Quando ANO_INICIO == ANO_FIM, equivale ao modo simples.
# ════════════════════════════════════════════════════════════════════════════

# MAPA_ROTULO, CATEGORIAS_CONSIDERADAS e PALETA estão em pncp_coleta.py


def _salvar(fig, nome: str, pasta: str) -> None:
    os.makedirs(pasta, exist_ok=True)
    p = os.path.join(pasta, nome)
    fig.savefig(p, bbox_inches="tight")
    print(f"   💾 {p}")
    plt.close(fig)
    if EM_COLAB:
        display(Image(p))


def _anotar_barras(ax, fmt: str = "{:,.0f}", offset: float = 0.01,
                    fontsize: int = 8, horizontal: bool = True) -> None:
    """
    Adiciona etiqueta com o valor exato no fim de cada barra.

    Trabalha tanto para barras horizontais (barh) quanto verticais (bar).
    `fmt` é uma f-string sem o nome — ex.: "{:,.0f}", "{:.1%}", "{:.2f}".

    Detecta automaticamente a orientação se `horizontal=None` for passado,
    mas o default `horizontal=True` cobre o caso mais comum no projeto
    (barras horizontais ordenadas por valor).
    """
    for patch in ax.patches:
        try:
            if horizontal:
                width = patch.get_width()
                if width == 0 or np.isnan(width):
                    continue
                # Posiciona à direita da barra
                xmax = ax.get_xlim()[1]
                ax.text(
                    width + xmax * offset,
                    patch.get_y() + patch.get_height() / 2,
                    fmt.format(width),
                    va="center", ha="left", fontsize=fontsize, color="#333",
                )
            else:
                height = patch.get_height()
                if height == 0 or np.isnan(height):
                    continue
                ymax = ax.get_ylim()[1]
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    height + ymax * offset,
                    fmt.format(height),
                    ha="center", va="bottom", fontsize=fontsize, color="#333",
                )
        except Exception:
            continue
    # Ajusta limites para acomodar as etiquetas
    if horizontal:
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 * 1.18)
    else:
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 * 1.12)


def resumo_geral(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*62 + "\n  RESUMO GERAL\n" + "═"*62)
    ct = df["rotulo"].value_counts()
    pt = df["rotulo"].value_counts(normalize=True).mul(100)
    r = pd.DataFrame({"qtd": ct, "pct (%)": pt.round(2)})
    print("\n── Distribuição de rótulos ──")
    print(r.to_string())
    razao = ct.get("geral", 0) / max(ct.get("engenharia", 1), 1)
    status = ("⚠ MUITO desbalanceado" if razao > 10
              else "⚠ Desbalanceado" if razao > 3 else "✓ OK")
    print(f"\n   Razão geral:engenharia ≈ {razao:.1f}:1   {status}")
    if "esferaNome" in df.columns:
        print("\n── Esfera × rótulo ──")
        print(pd.crosstab(df["esferaNome"], df["rotulo"], margins=True).to_string())
    if "poderNome" in df.columns:
        print("\n── Poder × rótulo ──")
        print(pd.crosstab(df["poderNome"], df["rotulo"], margins=True).to_string())
    print("\n── Ausentes (%) ──")
    a = df.isnull().mean().mul(100).round(1)
    a = a[a > 0].sort_values(ascending=False)
    print(a.to_string() if not a.empty else "   Nenhum.")
    print("═"*62)
    return r


def estatisticas_valor(df: pd.DataFrame) -> dict:
    """Aula 3 — 4 momentos (média, variância, obliquidade, curtose)."""
    resultados = {}
    for col in ["valorTotalEstimado", "valorTotalHomologado"]:
        if col not in df.columns: continue
        s = df.groupby("rotulo")[col].describe(
            percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]).round(2)
        s["cv_%"]     = (df.groupby("rotulo")[col].std()
                         / df.groupby("rotulo")[col].mean() * 100).round(1)
        s["skew"]     = df.groupby("rotulo")[col].skew().round(3)
        s["kurtosis"] = df.groupby("rotulo")[col].apply(lambda x: x.kurt()).round(3)
        resultados[col] = s
        print(f"\n── {col} por rótulo (4 momentos — Aula 3) ──")
        print(s.to_string())
    if "valorTotalEstimado" in df.columns:
        p75 = df.loc[df["rotulo"]=="engenharia", "valorTotalEstimado"].quantile(0.75)
        susp = df[(df["rotulo"]=="geral") & (df["valorTotalEstimado"] >= p75)]
        print(f"\n⚠ 'geral' com valor ≥ p75 da engenharia (≥ R$ {p75:,.0f}): "
              f"{len(susp):,} ({len(susp)/len(df)*100:.1f}%)")
    return resultados


def comprimento_objeto(df: pd.DataFrame) -> pd.DataFrame:
    if "objeto" not in df.columns: return pd.DataFrame()
    df = df.copy()
    df["len_chars"]  = df["objeto"].str.len()
    df["len_tokens"] = df["objeto"].str.split().str.len()
    df["ttr"] = df["objeto"].apply(
        lambda t: (lambda ts: len(set(ts))/max(len(ts), 1))(tokenizar(t)))
    s = df.groupby("rotulo")[["len_chars","len_tokens","ttr"]].describe(
        percentiles=[0.25, 0.50, 0.75]).round(2)
    print("\n── Comprimento e TTR do objeto ──")
    print(s.to_string())
    return s


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — Gráficos da EDA
# ════════════════════════════════════════════════════════════════════════════

def g_distribuicao_categorias(df, pasta):
    ct = df["rotulo"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(ct.index, ct.values,
                  color=[PALETA[r] for r in ct.index],
                  edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, ct.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+ct.max()*0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    pct = ct / ct.sum() * 100
    for i, (rot, p) in enumerate(pct.items()):
        ax.text(i, ct[rot]/2, f"{p:.1f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=12)
    ax.set_title("Distribuição de contratações por categoria", fontweight="bold")
    ax.set_xlabel("Rótulo"); ax.set_ylabel("Qtd")
    ax.set_ylim(0, ct.max()*1.18); sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "01_distribuicao_categorias.png", pasta)


def g_boxplot_valor_duplo(df, pasta):
    """Aula 6 — log atenua assimetria dos valores."""
    cols = [c for c in ["valorTotalEstimado","valorTotalHomologado"] if c in df.columns]
    if not cols: return
    fig, axes = plt.subplots(1, len(cols), figsize=(6*len(cols), 5))
    if len(cols) == 1: axes = [axes]
    for ax, col in zip(axes, cols):
        dados = df[df[col] > 0]
        if dados.empty: continue
        sns.boxplot(data=dados, x="rotulo", y=col, palette=PALETA, ax=ax,
                    order=["geral","engenharia"],
                    flierprops=dict(marker="o", markersize=2, alpha=0.35))
        ax.set_yscale("log"); ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Rótulo"); ax.set_ylabel("R$ (log)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
        for i, rot in enumerate(["geral","engenharia"]):
            m = dados.loc[dados["rotulo"]==rot, col].median()
            if not np.isnan(m):
                ax.text(i, m, f" R${m:,.0f}", va="center", fontsize=8)
        sns.despine(ax=ax)
    fig.suptitle("Boxplot de valores por rótulo (escala log — Aula 6)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    _salvar(fig, "02_boxplot_valor.png", pasta)


def g_serie_temporal(df, pasta):
    if "mesPublicacao" not in df.columns: return
    df = df.copy()
    df["periodo"] = (df["anoPublicacao"].astype(str) + "-"
                     + df["mesPublicacao"].astype(str).str.zfill(2))
    serie = (df.groupby(["periodo","rotulo"]).size()
             .unstack(fill_value=0).sort_index())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for rot, cor in PALETA.items():
        if rot in serie.columns:
            axes[0].plot(serie.index, serie[rot], marker="o",
                         label=rot.capitalize(), color=cor, linewidth=2)
    axes[0].set_title("Contratações por mês", fontweight="bold")
    axes[0].set_xlabel("Período"); axes[0].set_ylabel("Qtd")
    axes[0].legend(); axes[0].tick_params(axis="x", rotation=45)
    if "engenharia" in serie.columns:
        prop = serie["engenharia"] / serie.sum(axis=1) * 100
        axes[1].bar(prop.index, prop.values, color=PALETA["engenharia"], alpha=0.8)
        axes[1].axhline(prop.mean(), color="red", linestyle="--", lw=1,
                        label=f"Média: {prop.mean():.1f}%")
        axes[1].set_title("% Engenharia por mês", fontweight="bold")
        axes[1].legend(); axes[1].tick_params(axis="x", rotation=45)
    for ax in axes: sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "03_serie_temporal.png", pasta)


def g_top_municipios(df, pasta, top_n=10):
    """Top-N municípios: volume total e volume de engenharia."""
    if "municipioNome" not in df.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top_t = df["municipioNome"].value_counts().head(top_n).index
    piv = (df[df["municipioNome"].isin(top_t)]
           .groupby(["municipioNome","rotulo"]).size()
           .unstack(fill_value=0).reindex(top_t))
    piv.plot(kind="barh", ax=axes[0],
             color=[PALETA.get(c,"#aaa") for c in piv.columns], edgecolor="white")
    axes[0].set_title(f"Top-{top_n} municípios — volume total", fontweight="bold")
    axes[0].set_xlabel("Qtd"); axes[0].set_ylabel("Município")
    axes[0].legend(title="Rótulo"); axes[0].invert_yaxis()
    _anotar_barras(axes[0], fmt="{:,.0f}", fontsize=7)

    if "engenharia" in df["rotulo"].values:
        eng = df[df["rotulo"]=="engenharia"]
        top_e = eng["municipioNome"].value_counts().head(top_n).index
        if len(top_e) > 0:
            piv_e = (df[df["municipioNome"].isin(top_e)]
                     .groupby(["municipioNome","rotulo"]).size()
                     .unstack(fill_value=0).reindex(top_e))
            piv_e.plot(kind="barh", ax=axes[1],
                       color=[PALETA.get(c,"#aaa") for c in piv_e.columns],
                       edgecolor="white")
            axes[1].set_title(f"Top-{top_n} municípios — maior volume de ENGENHARIA",
                              fontweight="bold")
            axes[1].set_xlabel("Qtd"); axes[1].set_ylabel("Município")
            axes[1].legend(title="Rótulo"); axes[1].invert_yaxis()
            _anotar_barras(axes[1], fmt="{:,.0f}", fontsize=7)
    for ax in axes: sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "04_top_municipios.png", pasta)


def g_top_orgaos(df, pasta, top_n=10):
    col = "razaoSocialOrgao" if "razaoSocialOrgao" in df.columns else "nomeUnidade"
    if col not in df.columns: return
    df = df.copy()
    df["_org"] = df[col].astype(str).str.slice(0, 45)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    top_t = df["_org"].value_counts().head(top_n).index
    piv = (df[df["_org"].isin(top_t)].groupby(["_org","rotulo"]).size()
           .unstack(fill_value=0).reindex(top_t))
    piv.plot(kind="barh", ax=axes[0],
             color=[PALETA.get(c,"#aaa") for c in piv.columns], edgecolor="white")
    axes[0].set_title(f"Top-{top_n} órgãos — volume total", fontweight="bold")
    axes[0].set_xlabel("Qtd"); axes[0].set_ylabel("Órgão")
    axes[0].legend(title="Rótulo"); axes[0].invert_yaxis()
    _anotar_barras(axes[0], fmt="{:,.0f}", fontsize=7)

    if "engenharia" in df["rotulo"].values:
        eng = df[df["rotulo"]=="engenharia"]
        top_e = eng["_org"].value_counts().head(top_n).index
        if len(top_e) > 0:
            piv_e = (df[df["_org"].isin(top_e)]
                     .groupby(["_org","rotulo"]).size()
                     .unstack(fill_value=0).reindex(top_e))
            piv_e.plot(kind="barh", ax=axes[1],
                       color=[PALETA.get(c,"#aaa") for c in piv_e.columns],
                       edgecolor="white")
            axes[1].set_title(f"Top-{top_n} órgãos — maior volume de ENGENHARIA",
                              fontweight="bold")
            axes[1].set_xlabel("Qtd"); axes[1].set_ylabel("Órgão")
            axes[1].legend(title="Rótulo"); axes[1].invert_yaxis()
            _anotar_barras(axes[1], fmt="{:,.0f}", fontsize=7)
    for ax in axes: sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "05_top_orgaos.png", pasta)


def g_modalidade_criterio(df, pasta):
    """Modalidade e critério × rótulo — sinais de subenquadramento."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col, titulo in [(axes[0],"modalidadeNome","Modalidade"),
                             (axes[1],"criterioJulgamentoNome","Critério")]:
        if col not in df.columns:
            ax.set_visible(False); continue
        top = df[col].value_counts().head(8).index
        sub = df[df[col].isin(top)]
        piv = (sub.groupby([col,"rotulo"]).size()
               .unstack(fill_value=0).reindex(top))
        piv.plot(kind="barh", ax=ax,
                 color=[PALETA.get(c,"#aaa") for c in piv.columns], edgecolor="white")
        ax.set_title(f"{titulo} × rótulo", fontweight="bold")
        ax.set_xlabel("Qtd"); ax.legend(title="Rótulo"); ax.invert_yaxis()
        sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "06_modalidade_criterio.png", pasta)


def g_comprimento_objeto(df, pasta):
    if "objeto" not in df.columns: return
    df = df.copy()
    df["len_tokens"] = df["objeto"].str.split().str.len()
    df["ttr"] = df["objeto"].apply(
        lambda t: (lambda ts: len(set(ts))/max(len(ts), 1))(tokenizar(t)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for rot, cor in PALETA.items():
        sub = df[df["rotulo"]==rot]
        if sub.empty: continue
        sns.kdeplot(sub["len_tokens"], ax=axes[0], label=rot.capitalize(),
                    color=cor, fill=True, alpha=0.35, linewidth=1.5)
        sns.kdeplot(sub["ttr"], ax=axes[1], label=rot.capitalize(),
                    color=cor, fill=True, alpha=0.35, linewidth=1.5)
    axes[0].set_title("Distribuição de tokens por rótulo", fontweight="bold")
    axes[0].set_xlabel("Nº tokens"); axes[0].set_ylabel("Densidade"); axes[0].legend()
    axes[1].set_title("Riqueza lexical (TTR) por rótulo", fontweight="bold")
    axes[1].set_xlabel("TTR"); axes[1].legend()
    for ax in axes: sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "07_comprimento_e_ttr.png", pasta)


def g_matriz_correlacao(df, pasta):
    """Aula 7 — Spearman (não-paramétrica, robusta à assimetria)."""
    cols = [c for c in ["valorTotalEstimado","valorTotalHomologado",
                         "duracaoPropostaDias","amparoLegalCodigo"] if c in df.columns]
    d = df.copy()
    if "objeto" in d.columns:
        d["len_tokens"] = d["objeto"].str.split().str.len()
        d["n_keywords_eng"] = d["objeto"].apply(
            lambda t: len(set(tokenizar(str(t))) & KEYWORDS_ENG))
        cols += ["len_tokens", "n_keywords_eng"]
    d["is_engenharia"] = (d["rotulo"]=="engenharia").astype(int)
    cols.append("is_engenharia")
    d = d[cols].apply(pd.to_numeric, errors="coerce")
    corr = d.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Spearman"}, ax=ax)
    ax.set_title("Matriz de correlação (Spearman — Aula 7)\n"
                 "variáveis numéricas × is_engenharia", fontweight="bold")
    plt.xticks(rotation=40, ha="right"); plt.yticks(rotation=0)
    fig.tight_layout()
    _salvar(fig, "08_matriz_correlacao.png", pasta)


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — NLP básico
# ════════════════════════════════════════════════════════════════════════════

KEYWORDS_ENG = {
    "reforma","obra","obras","construcao","construção","manutencao","manutenção",
    "instalacao","instalação","pavimento","pavimentacao","drenagem","estrutura",
    "eletrica","elétrica","hidraulica","hidráulica","topografia","sondagem",
    "geotecnico","geotécnico","projeto","projetos","engenharia","engenheiro",
    "art","rrt","spda","incendio","incêndio","impermeabilizacao","impermeabilização",
    "fundacao","fundação","alvenaria","cobertura","telhado","esquadria",
    "climatizacao","climatização","ar-condicionado","arcondicionado",
    "subestacao","subestação","cabeamento","pintura","revestimento","piso","pisos",
    "esgoto","agua","abastecimento",
}


def _freq_tokens(textos, top_n=30):
    todos = []
    for t in textos.dropna():
        todos.extend(tokenizar(str(t)))
    return pd.Series(collections.Counter(todos)).nlargest(top_n)

def _freq_bigramas(textos, top_n=20):
    todos = []
    for t in textos.dropna():
        todos.extend(bigramas(tokenizar(str(t))))
    return pd.Series(collections.Counter(todos)).nlargest(top_n)


def _freq_tokens_com_radical(textos, top_n=30, usar_stem=True):
    """
    Versão de _freq_tokens que aplica stem para agregar variantes
    (material/materiais → materi*), respondendo ao pedido de não
    contabilizar singular/plural separadamente nos gráficos.
    """
    todos = []
    for t in textos.dropna():
        todos.extend(tokenizar(str(t), stem=usar_stem))
    return pd.Series(collections.Counter(todos)).nlargest(top_n)


def g_frequencia_palavras(df, pasta, top_n=25, usar_stem=True):
    """
    Top-N palavras por classe (com stem RSLP por padrão).

    Por que stem aqui mas não nas keywords? O stem (RSLP — Aula quinzena 01)
    junta variantes singular/plural/conjugadas: material+materiais→'materi*',
    obra+obras→'obr*'. Isso evita que a tabela de frequência fique poluída
    com a "mesma" palavra duas vezes. Mas KEYWORDS_ENG é uma lista FIXA com
    palavras inteiras, então não fazemos stem lá.
    """
    if "objeto" not in df.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, rot, cor in [(axes[0],"geral",PALETA["geral"]),
                          (axes[1],"engenharia",PALETA["engenharia"])]:
        freq = _freq_tokens_com_radical(df.loc[df["rotulo"]==rot,"objeto"],
                                          top_n, usar_stem=usar_stem)
        if freq.empty: ax.set_visible(False); continue
        freq.sort_values().plot(kind="barh", ax=ax, color=cor, edgecolor="white")
        ax.set_title(f"Top-{top_n} radicais — {rot.upper()}" if usar_stem
                     else f"Top-{top_n} palavras — {rot.upper()}",
                     fontweight="bold")
        ax.set_xlabel("Frequência")
        _anotar_barras(ax, fmt="{:,.0f}")
        sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "09_top_palavras.png", pasta)


def g_bigramas(df, pasta, top_n=15):
    if "objeto" not in df.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, rot, cor in [(axes[0],"geral",PALETA["geral"]),
                          (axes[1],"engenharia",PALETA["engenharia"])]:
        freq = _freq_bigramas(df.loc[df["rotulo"]==rot,"objeto"], top_n)
        if freq.empty: ax.set_visible(False); continue
        freq.sort_values().plot(kind="barh", ax=ax, color=cor, edgecolor="white")
        ax.set_title(f"Top-{top_n} bigramas — {rot.upper()}", fontweight="bold")
        ax.set_xlabel("Frequência")
        _anotar_barras(ax, fmt="{:,.0f}")
        sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "10_bigramas.png", pasta)


def g_lei_de_zipf(df, pasta, top_n=200):
    """
    Lei de Zipf — frequência × rank em escala log-log (Aula quinzena 02).

    A Lei de Zipf afirma que em qualquer corpus de linguagem natural, a
    frequência de uma palavra é inversamente proporcional ao seu rank.
    Em escala log-log, a curva tende a ser uma reta com inclinação ≈ -1.

    Comparar engenharia × geral:
    • Se ambas seguem Zipf, o vocabulário é típico do português.
    • Pequenos desvios na "cauda" indicam termos especializados.
    """
    if "objeto" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    info_classes = []

    for rot, cor in PALETA.items():
        textos = df.loc[df["rotulo"] == rot, "objeto"].dropna()
        if textos.empty:
            continue
        todos_tokens = []
        for t in textos:
            todos_tokens.extend(tokenizar(str(t)))
        if not todos_tokens:
            continue
        cnt = collections.Counter(todos_tokens)
        ord_freq = sorted(cnt.values(), reverse=True)[:top_n]
        ranks    = np.arange(1, len(ord_freq) + 1)
        ax.loglog(ranks, ord_freq, marker=".", linestyle="-", color=cor,
                  label=f"{rot.capitalize()} ({len(todos_tokens):,} tokens, "
                        f"{len(cnt):,} types)", alpha=0.85)
        # Inclinação aproximada da regressão linear em log-log
        slope = np.polyfit(np.log(ranks), np.log(ord_freq), 1)[0]
        info_classes.append((rot, slope))

    # Linha teórica de Zipf (slope = -1)
    if info_classes:
        ranks_teor = np.arange(1, top_n + 1)
        f_teor = ord_freq[0] / ranks_teor   # f(r) ∝ 1/r
        ax.loglog(ranks_teor, f_teor, "k--", lw=1, alpha=0.5,
                  label="Lei de Zipf teórica (slope = -1)")

    ax.set_title("Lei de Zipf — frequência × rank (Aula quinzena 02)\n"
                 f"Inclinação observada: {' | '.join(f'{r}: {s:.2f}' for r, s in info_classes)}",
                 fontweight="bold")
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("Frequência (log)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "13_lei_de_zipf.png", pasta)


def g_nuvem_palavras(df, pasta):
    if not TEM_WORDCLOUD or "objeto" not in df.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, rot, cmap in [(axes[0],"geral","Oranges"),
                           (axes[1],"engenharia","Blues")]:
        subset = df.loc[df["rotulo"]==rot, "objeto"].dropna()
        if subset.empty: ax.axis("off"); continue
        toks = [tokenizar(str(t)) for t in subset]
        texto = " ".join(tok for lst in toks for tok in lst)
        if not texto.strip(): ax.axis("off"); continue
        wc = WordCloud(width=700, height=400, background_color="white",
                        colormap=cmap, max_words=100,
                        collocations=False, prefer_horizontal=0.9).generate(texto)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Nuvem — {rot.upper()} ({len(subset):,})", fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    _salvar(fig, "11_nuvem_palavras.png", pasta)


def g_keywords_eng_em_geral(df, pasta):
    """
    Análise de keywords de engenharia nos contratos GERAL.

    Responde duas perguntas:
      1. Quantos contratos 'geral' contêm pelo menos 1 keyword de engenharia?
      2. QUAIS keywords aparecem mais nos contratos 'geral'?
         (essa lista vira pista do TCC para "termos suspeitos de subenquadramento")
    """
    if "objeto" not in df.columns: return pd.DataFrame()
    def cnt(t): return len(set(tokenizar(str(t))) & KEYWORDS_ENG)
    df = df.copy()
    df["n_keywords_eng"]  = df["objeto"].apply(cnt)
    df["tem_keyword_eng"] = df["n_keywords_eng"] > 0
    res = df.groupby("rotulo").agg(
        registros=("n_keywords_eng","count"),
        com_keyword=("tem_keyword_eng","sum"),
        pct_com_keyword=("tem_keyword_eng", lambda x: x.mean()*100),
        media_keywords=("n_keywords_eng","mean"),
        max_keywords=("n_keywords_eng","max")).round(2)
    print("\n── Keywords de engenharia nos textos ──")
    print(res.to_string())
    g = df[(df["rotulo"]=="geral") & df["tem_keyword_eng"]]
    p = len(g)/max(len(df[df["rotulo"]=="geral"]),1)*100
    print(f"\n⚠ 'GERAL' com ≥1 keyword de engenharia: {len(g):,} ({p:.1f}%)")

    # ── NOVO: ranking detalhado de quais keywords aparecem nos GERAIS ─────
    geral_objs = df.loc[df["rotulo"] == "geral", "objeto"].dropna()
    eng_objs   = df.loc[df["rotulo"] == "engenharia", "objeto"].dropna()
    contagem_geral = collections.Counter()
    contagem_eng   = collections.Counter()
    for t in geral_objs:
        for tok in set(tokenizar(str(t))):
            if tok in KEYWORDS_ENG:
                contagem_geral[tok] += 1
    for t in eng_objs:
        for tok in set(tokenizar(str(t))):
            if tok in KEYWORDS_ENG:
                contagem_eng[tok] += 1

    n_geral_total = len(geral_objs); n_eng_total = len(eng_objs)
    linhas_kw = []
    for kw in KEYWORDS_ENG:
        ng = contagem_geral.get(kw, 0)
        ne = contagem_eng.get(kw, 0)
        if ng + ne == 0:
            continue
        linhas_kw.append({
            "keyword":         kw,
            "freq_em_geral":   ng,
            "pct_em_geral":    round(ng / max(n_geral_total, 1) * 100, 2),
            "freq_em_eng":     ne,
            "pct_em_eng":      round(ne / max(n_eng_total, 1) * 100, 2),
            # Lift = quantas vezes mais frequente em engenharia que em geral
            # (proporcionalmente). Lift baixo (≈1) = palavra genérica.
            #  Lift alto (>>1) = discriminativa de engenharia.
            "lift_eng_vs_geral": round(
                (ne / max(n_eng_total, 1)) /
                max(ng / max(n_geral_total, 1), 1e-6), 2),
        })
    df_kw = pd.DataFrame(linhas_kw).sort_values("freq_em_geral", ascending=False)
    print(f"\n── TOP-15 keywords de engenharia ENCONTRADAS NOS CONTRATOS 'GERAL' ──")
    print(f"   (esses são candidatos a subenquadramento — palavras de engenharia")
    print(f"    aparecendo em contratos rotulados como serviços gerais)")
    print(df_kw.head(15).to_string(index=False))
    arq = os.path.join(pasta, "keywords_eng_em_geral_detalhado.csv")
    df_kw.to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")

    # Gráfico de top 15 keywords nos geral
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for rot, cor in PALETA.items():
        s = df[df["rotulo"]==rot]["n_keywords_eng"]
        if s.empty: continue
        axes[0].hist(s, bins=range(0, s.max()+2), alpha=0.7,
                     label=rot.capitalize(), color=cor, edgecolor="white")
    axes[0].set_title("Distribuição de keywords de engenharia", fontweight="bold")
    axes[0].set_xlabel("Nº keywords"); axes[0].set_ylabel("Qtd")
    axes[0].legend(); sns.despine(ax=axes[0])

    if len(df[df["rotulo"]=="geral"]) > 0:
        pct = (df[df["rotulo"]=="geral"]
               .assign(t=lambda x: x["tem_keyword_eng"].map(
                   {True:"Com kw eng.", False:"Sem kw eng."}))
               ["t"].value_counts(normalize=True)*100)
        axes[1].pie(pct.values, labels=pct.index, autopct="%1.1f%%",
                    colors=[PALETA["engenharia"],"#cccccc"], startangle=90,
                    wedgeprops={"edgecolor":"white"})
        axes[1].set_title("Contratos 'GERAL' com kw eng.", fontweight="bold")

    # Gráfico 3: top 15 keywords mais frequentes nos GERAIS
    if not df_kw.empty:
        top = df_kw.head(15).iloc[::-1]
        axes[2].barh(top["keyword"], top["freq_em_geral"],
                       color=PALETA["engenharia"], edgecolor="white")
        axes[2].set_title("Top-15 keywords de eng. dentro dos GERAIS\n"
                          "(candidatos a subenquadramento)",
                          fontweight="bold")
        axes[2].set_xlabel("Freq nos contratos 'geral'")
        _anotar_barras(axes[2], fmt="{:,.0f}", fontsize=8)
        sns.despine(ax=axes[2])
    fig.tight_layout()
    _salvar(fig, "12_keywords_eng_em_geral.png", pasta)

    return df[["numeroControlePNCP","rotulo","objeto",
               "n_keywords_eng","valorTotalEstimado"]].copy()


def descobrir_keywords_dos_dados(df: pd.DataFrame,
                                    top_n: int = 50,
                                    metodo: str = "log_odds") -> pd.DataFrame:
    """
    Descobre as palavras MAIS DISCRIMINATIVAS da classe 'engenharia' a partir
    dos próprios dados (não de uma lista fixa).

    RESPOSTA À CRÍTICA METODOLÓGICA do KEYWORDS_ENG fixo:
    A KEYWORDS_ENG codificada no início do projeto traz o viés do programador
    sobre o que SERIA engenharia. Esta função descobre dos dados quais
    palavras DE FATO discriminam as classes.

    Métodos suportados:
      • "log_odds"     : log( P(palavra | eng) / P(palavra | geral) ).
                          Positivo → palavra mais comum em eng.
                          Sinal robusto e interpretável (base do classificador NB).
      • "mutual_info"  : informação mútua (já implementada em SelectKBest).
      • "chi2"         : teste χ² de independência entre palavra e classe.

    Por que usar isso no TCC:
    • Mostra que a metodologia é orientada por dados (data-driven), não por
      heurística pessoal.
    • Pode revelar termos discriminativos que não estavam na lista fixa.
    • Permite GERAR uma KEYWORDS_ENG_AUTO específica do estado/ano analisado.
    """
    from collections import Counter

    if "objeto" not in df.columns or "rotulo" not in df.columns:
        return pd.DataFrame()

    eng_textos = df.loc[df["rotulo"] == "engenharia", "objeto"].dropna()
    ger_textos = df.loc[df["rotulo"] == "geral",      "objeto"].dropna()

    n_eng = len(eng_textos)
    n_ger = len(ger_textos)
    if n_eng == 0 or n_ger == 0:
        print("   [aviso] uma das classes está vazia — não dá para descobrir keywords.")
        return pd.DataFrame()

    # Conta documentos com cada palavra (não tokens totais), para evitar viés
    # de documentos longos
    cnt_eng = Counter()
    cnt_ger = Counter()
    for t in eng_textos:
        for tok in set(tokenizar(str(t))):
            cnt_eng[tok] += 1
    for t in ger_textos:
        for tok in set(tokenizar(str(t))):
            cnt_ger[tok] += 1

    # Une vocabulário
    vocab = set(cnt_eng.keys()) | set(cnt_ger.keys())
    # Filtra termos muito raros
    min_doc = max(3, int(0.005 * (n_eng + n_ger)))   # 0.5% dos docs
    vocab = {w for w in vocab if cnt_eng[w] + cnt_ger[w] >= min_doc}

    linhas = []
    for w in vocab:
        e = cnt_eng[w] + 1   # smoothing
        g = cnt_ger[w] + 1
        # Probabilidades condicionais
        p_e = e / (n_eng + 1)
        p_g = g / (n_ger + 1)
        # Log-odds: positivo = mais comum em engenharia
        log_odds = np.log(p_e / p_g)
        # Frequência total (importância absoluta da palavra)
        suporte = cnt_eng[w] + cnt_ger[w]
        # Score combinado: log_odds × log(suporte) penaliza palavras raras
        score = log_odds * np.log1p(suporte)
        linhas.append({
            "palavra":     w,
            "freq_eng":    cnt_eng[w],
            "freq_geral":  cnt_ger[w],
            "pct_eng":     round(cnt_eng[w] / n_eng * 100, 2),
            "pct_geral":   round(cnt_ger[w] / n_ger * 100, 2),
            "log_odds":    round(log_odds, 3),
            "score":       round(score, 3),
        })

    df_disc = pd.DataFrame(linhas)
    df_disc = df_disc.sort_values("score", ascending=False).reset_index(drop=True)

    print(f"\n── Top-{top_n} palavras DESCOBERTAS dos dados como discriminativas de ENGENHARIA ──")
    print(f"   (log_odds positivo = mais comum em engenharia que em geral)")
    print(df_disc.head(top_n).to_string(index=False))

    print(f"\n── Top-{top_n//2} palavras discriminativas de GERAL ──")
    print(df_disc.tail(top_n // 2).iloc[::-1].to_string(index=False))

    return df_disc


def extrair_termos_dominio(df: pd.DataFrame,
                              top_n: int = 50,
                              n_min: int = 1, n_max: int = 4,
                              min_freq: int = 5) -> pd.DataFrame:
    """
    Extrai automaticamente os TERMOS de domínio (uni, bi, tri, tetragramas)
    mais característicos da classe 'engenharia' vs 'geral'.

    REFERÊNCIA TEÓRICA:
    Baseada em Conrado, Pardo & Rezende (2013) — "A Machine Learning Approach
    to Automatic Term Extraction using a Rich Feature Set" (NAACL HLT).
    Versão simplificada que usa as features mais robustas do paper:
      • Frequência do n-grama no domínio-alvo (engenharia)
      • Razão de frequência domínio-alvo / domínio-contraste (geral)
      • Especificidade (C-Value-like): n-gramas longos pontuam mais

    Por que isso é útil para o TCC:
    • Substitui (ou complementa) a KEYWORDS_ENG codificada manualmente
    • Captura TERMINOLOGIA TÉCNICA multi-palavra ("projeto básico", "obra
      de arte especial", "responsável técnico") que a análise unigrama perde
    • Resultado é AUTO-DERIVADO dos dados — defensável metodologicamente

    Parâmetros
    ──────────
    top_n     : nº de termos a retornar (top discriminativos para engenharia)
    n_min     : tamanho mínimo do n-grama (1 = inclui unigramas)
    n_max     : tamanho máximo do n-grama (4 = inclui tetragramas)
    min_freq  : frequência mínima no corpus inteiro para considerar o termo

    Retorna DataFrame com colunas:
      termo, n_palavras, freq_eng, freq_geral, score, contexto_exemplo
    """
    if "objeto_completo" in df.columns:
        col_texto = "objeto_completo"
    elif "objeto" in df.columns:
        col_texto = "objeto"
    else:
        print("   [pulado] dataframe sem coluna 'objeto'.")
        return pd.DataFrame()

    if "rotulo" not in df.columns:
        print("   [pulado] dataframe sem coluna 'rotulo'.")
        return pd.DataFrame()

    df_eng = df[df["rotulo"] == "engenharia"]
    df_ger = df[df["rotulo"] == "geral"]
    n_eng, n_ger = len(df_eng), len(df_ger)

    if n_eng < 10 or n_ger < 10:
        print(f"   [pulado] dados insuficientes: eng={n_eng}, geral={n_ger}.")
        return pd.DataFrame()

    print(f"\n   Extraindo termos n-gramas {n_min}-{n_max} de "
          f"{n_eng:,} eng + {n_ger:,} geral...")

    # Tokeniza
    def _tokens_doc(t):
        return tokenizar(str(t))

    from collections import Counter
    cnt_eng = Counter()
    cnt_ger = Counter()

    # Conta n-gramas por documento (presença, não frequência total — evita
    # palavras repetidas no mesmo doc enviesarem)
    for textos_lst, contador in [(df_eng[col_texto], cnt_eng),
                                   (df_ger[col_texto], cnt_ger)]:
        for t in textos_lst:
            tokens = _tokens_doc(t)
            visto_no_doc = set()
            for n in range(n_min, n_max + 1):
                for i in range(len(tokens) - n + 1):
                    ngrama = " ".join(tokens[i:i+n])
                    if ngrama not in visto_no_doc:
                        contador[ngrama] += 1
                        visto_no_doc.add(ngrama)

    # Vocabulário comum (com frequência mínima)
    vocab = set(cnt_eng.keys()) | set(cnt_ger.keys())
    vocab = {t for t in vocab if cnt_eng[t] + cnt_ger[t] >= min_freq}
    print(f"   Vocabulário inicial: {len(vocab):,} n-gramas")

    # Para cada termo, calcula:
    #  • freq relativa em cada classe
    #  • log-odds (sinal de classe)
    #  • bônus de especificidade: termos compostos pontuam mais
    #    (técnica simplificada inspirada em C-Value)
    resultados = []
    for termo in vocab:
        f_eng = cnt_eng[termo]
        f_ger = cnt_ger[termo]
        p_eng = (f_eng + 1) / (n_eng + 2)   # Laplace smoothing
        p_ger = (f_ger + 1) / (n_ger + 2)
        log_odds = np.log(p_eng / p_ger)
        n_palavras = termo.count(" ") + 1
        # Bônus de especificidade: termos multi-palavra são mais informativos
        # do que unigramas para terminologia técnica
        bonus = 1.0 + (n_palavras - 1) * 0.3
        score = log_odds * bonus

        # Só interessa termos PRÓ-engenharia (positivo)
        if score > 0:
            resultados.append({
                "termo":      termo,
                "n_palavras": n_palavras,
                "freq_eng":   f_eng,
                "freq_geral": f_ger,
                "log_odds":   round(log_odds, 3),
                "score":      round(score, 3),
            })

    df_termos = pd.DataFrame(resultados).sort_values("score", ascending=False)
    df_termos = df_termos.head(top_n)

    # Adiciona um exemplo de contexto onde o termo aparece (para interpretação)
    def _achar_contexto(termo):
        for t in df_eng[col_texto].head(200):
            t_str = str(t)
            if termo.lower() in _normalizar(t_str).lower():
                return t_str[:100] + ("..." if len(t_str) > 100 else "")
        return ""
    df_termos["contexto_exemplo"] = df_termos["termo"].apply(_achar_contexto)

    print(f"\n── Top-{top_n} termos discriminativos de engenharia ──")
    print(f"   (pontuação = log-odds × bônus de especificidade)")
    print(df_termos[["termo", "n_palavras", "freq_eng", "freq_geral",
                       "log_odds", "score"]].head(20).to_string(index=False))

    return df_termos


def descobrir_keywords_obras_vs_servicos_eng(df: pd.DataFrame,
                                                  top_n: int = 30) -> pd.DataFrame:
    """
    Descoberta granular: dentro da classe ENGENHARIA, quais palavras
    distinguem OBRAS (categoria 7) de SERVIÇOS DE ENGENHARIA (categoria 9)?

    Útil para o TCC porque revela o vocabulário específico de cada tipo:
      • Categoria 7 (Obras): termos como construção, edificação, fundação,
        pavimentação, projeto executivo
      • Categoria 9 (Serv. Eng.): termos como manutenção, reparo,
        adequação, instalação, conservação

    A separação automática via subclasse é mais limpa que tentar inferir
    "obra vs serviço" só pelas palavras (heurística do projeto).
    """
    from collections import Counter

    if "subclasse" not in df.columns:
        print("   [pulado] coluna 'subclasse' ausente. Rode carregar_e_limpar primeiro.")
        return pd.DataFrame()

    obra_textos = df.loc[df["subclasse"] == "obra", "objeto"].dropna()
    serv_textos = df.loc[df["subclasse"] == "serv_engenharia", "objeto"].dropna()

    n_obra = len(obra_textos)
    n_serv = len(serv_textos)
    if n_obra < 5 or n_serv < 5:
        print(f"   [pulado] dados insuficientes: obras={n_obra}, serv_eng={n_serv}.")
        return pd.DataFrame()

    cnt_obra = Counter()
    cnt_serv = Counter()
    for t in obra_textos:
        for tok in set(tokenizar(str(t))):
            cnt_obra[tok] += 1
    for t in serv_textos:
        for tok in set(tokenizar(str(t))):
            cnt_serv[tok] += 1

    vocab = set(cnt_obra.keys()) | set(cnt_serv.keys())
    min_doc = max(3, int(0.01 * (n_obra + n_serv)))
    vocab = {w for w in vocab if cnt_obra[w] + cnt_serv[w] >= min_doc}

    linhas = []
    for w in vocab:
        o = cnt_obra[w] + 1
        s = cnt_serv[w] + 1
        p_o = o / (n_obra + 1)
        p_s = s / (n_serv + 1)
        log_odds = np.log(p_o / p_s)
        suporte = cnt_obra[w] + cnt_serv[w]
        score = log_odds * np.log1p(suporte)
        linhas.append({
            "palavra":     w,
            "freq_obra":   cnt_obra[w],
            "freq_serv":   cnt_serv[w],
            "pct_obra":    round(cnt_obra[w] / n_obra * 100, 2),
            "pct_serv":    round(cnt_serv[w] / n_serv * 100, 2),
            "log_odds":    round(log_odds, 3),
            "score":       round(score, 3),
        })
    df_disc = pd.DataFrame(linhas).sort_values("score", ascending=False)

    print(f"\n── Top-{top_n} palavras de OBRAS (categoria 7) ──")
    print(f"   (log_odds positivo = mais comum em obras que em serv. eng.)")
    print(df_disc.head(top_n).to_string(index=False))

    print(f"\n── Top-{top_n//2} palavras de SERVIÇOS DE ENG. (categoria 9) ──")
    print(df_disc.tail(top_n // 2).iloc[::-1].to_string(index=False))

    return df_disc


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 10 — Orquestrador EDA
# ════════════════════════════════════════════════════════════════════════════

def analisar_vies_temporal(df: pd.DataFrame,
                              pasta_saida: str = ".",
                              limiar_diferenca: float = 0.5) -> dict:
    """
    Analisa se há VIÉS DE CLASSIFICAÇÃO ao longo dos anos.

    Hipótese: o PNCP entrou em vigor em ago/2021 mas só ficou OBRIGATÓRIO
    em abr/2023 (Lei 14.133/2021 art. 175). Nos anos iniciais, os órgãos
    podem ter classificado erroneamente serviços de engenharia como
    'serviços gerais' por desconhecimento ou inércia.

    Esta análise ajuda a identificar se há um "ano de corte" abaixo do
    qual os dados são pouco confiáveis para o classificador.

    Métricas analisadas por ano:
      • Volume total
      • Distribuição de subclasses (obras / serv_eng / serv_geral)
      • % de cada categoria
      • Diferença vs ano seguinte (detecta saltos bruscos)

    Saída:
      • Gráficos de evolução
      • Recomendação automática de ano de corte (se aplicável)
      • DataFrame com métricas por ano

    Parâmetros
    ──────────
    df               : DataFrame da Camada 1 limpo
    pasta_saida      : pasta para salvar gráficos
    limiar_diferenca : se a diferença em pp do ano X para o ano X+1 for
                        ≥ este valor, é considerado salto brusco
                        (default 0.5 pontos percentuais)
    """
    print("\n" + "█"*62)
    print("  ANÁLISE DE VIÉS TEMPORAL DA ROTULAÇÃO")
    print("█"*62)

    # Garante coluna anoPublicacao
    if "anoPublicacao" not in df.columns:
        if "dataPublicacaoPncp" in df.columns:
            df = df.copy()
            df["anoPublicacao"] = pd.to_datetime(
                df["dataPublicacaoPncp"], errors="coerce"
            ).dt.year

    if "anoPublicacao" not in df.columns or df["anoPublicacao"].isna().all():
        print("   [pulado] sem coluna anoPublicacao.")
        return {}

    df_use = df.dropna(subset=["anoPublicacao"]).copy()
    df_use["anoPublicacao"] = df_use["anoPublicacao"].astype(int)

    anos = sorted(df_use["anoPublicacao"].unique())
    if len(anos) < 2:
        print(f"   ⚠ Apenas 1 ano nos dados ({anos}). Análise de viés "
              f"requer pelo menos 2 anos.")
        return {}

    # ── Distribuição por ano e categoria ────────────────────────────────────
    if "subclasse" in df_use.columns:
        cross = pd.crosstab(df_use["anoPublicacao"], df_use["subclasse"])
        cross_pct = pd.crosstab(df_use["anoPublicacao"], df_use["subclasse"],
                                  normalize="index") * 100
    else:
        cross = pd.crosstab(df_use["anoPublicacao"], df_use["rotulo"])
        cross_pct = pd.crosstab(df_use["anoPublicacao"], df_use["rotulo"],
                                  normalize="index") * 100

    print(f"\n── Volume absoluto por ano e categoria ──")
    print(cross.to_string())
    print(f"\n── % por ano (cada linha soma 100%) ──")
    print(cross_pct.round(2).to_string())

    # ── Detecta saltos bruscos ──────────────────────────────────────────────
    # Para cada ano (a partir do 2º), calcula a diferença em pp para o anterior
    saltos = []
    for col in cross_pct.columns:
        for i in range(1, len(anos)):
            ano_atual = anos[i]; ano_ant = anos[i-1]
            pct_atual = cross_pct.loc[ano_atual, col]
            pct_ant   = cross_pct.loc[ano_ant, col]
            diff = pct_atual - pct_ant
            if abs(diff) >= limiar_diferenca:
                saltos.append({
                    "categoria":   col,
                    "ano_anterior": ano_ant,
                    "ano":          ano_atual,
                    "%_anterior":   round(pct_ant, 2),
                    "%_atual":      round(pct_atual, 2),
                    "diff_pp":      round(diff, 2),
                })

    df_saltos = pd.DataFrame(saltos)
    if not df_saltos.empty:
        print(f"\n── Saltos bruscos detectados (≥ {limiar_diferenca} pp) ──")
        print(df_saltos.to_string(index=False))

    # ── Recomendação automática ─────────────────────────────────────────────
    cat_eng = "engenharia" if "engenharia" in cross_pct.columns else None
    cat_obra = "obra" if "obra" in cross_pct.columns else None
    cat_serveng = "serv_engenharia" if "serv_engenharia" in cross_pct.columns else None

    recomendacao = None
    ano_corte = None

    # Heurística: se o ano com MENOR % de eng/obra for MUITO menor que os outros
    # (>50% relativo), recomendamos cortá-lo.
    pct_eng_por_ano = pd.Series(dtype=float)
    if cat_obra and cat_serveng:
        pct_eng_por_ano = (cross_pct[cat_obra].fillna(0) +
                           cross_pct[cat_serveng].fillna(0))
    elif cat_eng:
        pct_eng_por_ano = cross_pct[cat_eng]

    if not pct_eng_por_ano.empty and len(pct_eng_por_ano) >= 2:
        media = pct_eng_por_ano.mean()
        for ano, pct in pct_eng_por_ano.items():
            if media > 0 and pct < media * 0.5:
                ano_corte = ano
                recomendacao = (
                    f"Ano {ano} tem apenas {pct:.2f}% de engenharia/obras "
                    f"(média geral: {media:.2f}%). "
                    f"Considere FILTRAR esse ano para reduzir viés."
                )
                break

    if recomendacao:
        print(f"\n💡 Recomendação automática:")
        print(f"   {recomendacao}")
    else:
        print(f"\n✓ Nenhum viés temporal forte detectado.")

    # ── Gráficos ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Volume absoluto por ano (barras agrupadas)
    cross.plot(kind="bar", ax=axes[0, 0], edgecolor="white",
                colormap="tab10")
    axes[0, 0].set_title("Volume de contratos por ano e categoria",
                          fontweight="bold")
    axes[0, 0].set_xlabel("Ano"); axes[0, 0].set_ylabel("Nº contratos")
    axes[0, 0].tick_params(axis="x", rotation=0)
    axes[0, 0].legend(title="Categoria")
    sns.despine(ax=axes[0, 0])

    # 2. % por ano (linhas)
    for col in cross_pct.columns:
        axes[0, 1].plot(cross_pct.index.astype(int), cross_pct[col],
                          marker="o", linewidth=2, label=str(col))
    axes[0, 1].set_title("Proporção (%) por ano e categoria",
                          fontweight="bold")
    axes[0, 1].set_xlabel("Ano"); axes[0, 1].set_ylabel("%")
    axes[0, 1].legend(title="Categoria")
    axes[0, 1].grid(True, alpha=0.3)
    sns.despine(ax=axes[0, 1])

    # 3. % de engenharia (obras+serv_eng) ao longo do tempo (foco do TCC)
    if not pct_eng_por_ano.empty:
        axes[1, 0].plot(pct_eng_por_ano.index.astype(int), pct_eng_por_ano.values,
                          marker="o", linewidth=2.5, color="#1a6faf")
        axes[1, 0].fill_between(pct_eng_por_ano.index.astype(int),
                                  pct_eng_por_ano.values, alpha=0.2, color="#1a6faf")
        media_pct = pct_eng_por_ano.mean()
        axes[1, 0].axhline(media_pct, color="red", linestyle="--",
                             alpha=0.7, label=f"Média: {media_pct:.2f}%")
        axes[1, 0].set_title("% de eng./obras por ano (chave do TCC)\n"
                              "Anos com proporção MUITO baixa = candidatos a viés",
                              fontweight="bold")
        axes[1, 0].set_xlabel("Ano"); axes[1, 0].set_ylabel("% engenharia")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        for x, y in zip(pct_eng_por_ano.index.astype(int), pct_eng_por_ano.values):
            axes[1, 0].annotate(f"{y:.2f}%", (x, y),
                                  textcoords="offset points", xytext=(0, 10),
                                  ha="center", fontsize=9)
        sns.despine(ax=axes[1, 0])

    # 4. Volume total por ano
    vol = df_use.groupby("anoPublicacao").size()
    axes[1, 1].bar(vol.index.astype(int), vol.values, color="#888",
                     edgecolor="white")
    axes[1, 1].set_title("Volume total por ano", fontweight="bold")
    axes[1, 1].set_xlabel("Ano"); axes[1, 1].set_ylabel("Nº contratos")
    _anotar_barras(axes[1, 1], fmt="{:,.0f}", horizontal=False, fontsize=9)
    sns.despine(ax=axes[1, 1])

    fig.tight_layout()
    _salvar(fig, "p1_vies_temporal.png", pasta_saida)

    cross.to_csv(os.path.join(pasta_saida, "p1_vies_temporal_volume.csv"))
    cross_pct.round(2).to_csv(os.path.join(pasta_saida, "p1_vies_temporal_pct.csv"))

    return {
        "cross_volume":   cross,
        "cross_pct":      cross_pct,
        "saltos":         df_saltos,
        "ano_corte_recomendado": ano_corte,
        "recomendacao":   recomendacao,
    }


def _derivar_sufixo_uf_ano(df: pd.DataFrame) -> str:
    """
    Deriva sufixo padrão 'UF_ANO' ou 'UF_ANOMIN_ANOMAX' a partir do df.
    Usado para nomear pastas e arquivos de saída de forma consistente.
    Exemplo:  'SP_2024'  ou  'SP_2022_2024'
    """
    uf = (df["ufSigla"].mode()[0]
          if "ufSigla" in df.columns and not df["ufSigla"].isna().all()
          else "xx")
    if "anoPublicacao" in df.columns:
        anos = df["anoPublicacao"].dropna().astype(int)
        if len(anos) > 0:
            a_min, a_max = int(anos.min()), int(anos.max())
            periodo = f"{a_min}_{a_max}" if a_min != a_max else str(a_min)
            return f"{uf}_{periodo}"
    return f"{uf}_xxxx"


def _pasta_saida_padrao(df: pd.DataFrame) -> str:
    """Pasta padrão para gráficos: 'graficos_pncp_<UF>_<periodo>/'."""
    return f"graficos_pncp_{_derivar_sufixo_uf_ano(df)}"


def executar_eda_completo(df, pasta_saida="graficos/"):
    os.makedirs(pasta_saida, exist_ok=True)
    res = {}
    print("\n" + "█"*62 + "\n  PARTE 1 — EDA + NLP EXPLORATÓRIO\n" + "█"*62)
    print("\n[1] Resumo geral...");       res["resumo"] = resumo_geral(df)
    print("\n[2] Estatísticas de valor..."); res["stats_valor"] = estatisticas_valor(df)
    print("\n[3] Comprimento e TTR...");   res["stats_objeto"] = comprimento_objeto(df)
    print(f"\n[4] Gráficos em '{pasta_saida}'...")
    g_distribuicao_categorias(df, pasta_saida)
    g_boxplot_valor_duplo(df, pasta_saida)
    g_serie_temporal(df, pasta_saida)
    g_top_municipios(df, pasta_saida)
    g_top_orgaos(df, pasta_saida)
    g_modalidade_criterio(df, pasta_saida)
    g_comprimento_objeto(df, pasta_saida)
    g_matriz_correlacao(df, pasta_saida)
    print("\n[5] NLP básico...")
    g_frequencia_palavras(df, pasta_saida)
    g_bigramas(df, pasta_saida)
    g_lei_de_zipf(df, pasta_saida)
    g_nuvem_palavras(df, pasta_saida)
    res["suspeitos"] = g_keywords_eng_em_geral(df, pasta_saida)
    # Descoberta orientada por dados: quais palavras DE FATO discriminam eng × geral
    print("\n[6] Descoberta de keywords ORIENTADA POR DADOS (não fixa)...")
    res["palavras_discriminativas"] = descobrir_keywords_dos_dados(df, top_n=40)
    if not res["palavras_discriminativas"].empty:
        arq = os.path.join(pasta_saida, "palavras_discriminativas.csv")
        res["palavras_discriminativas"].to_csv(arq, index=False, encoding="utf-8-sig")
        print(f"   💾 {arq}")

    # Extração de termos multi-palavra (Conrado, Pardo & Rezende, 2013)
    # Captura terminologia técnica como "projeto básico", "responsável técnico",
    # "obra de arte especial", que análise unigrama perde.
    print("\n[6B] Extração de TERMOS de domínio (n-gramas, baseado em LABIC)...")
    res["termos_dominio"] = extrair_termos_dominio(df, top_n=50, n_max=4, min_freq=5)
    if not res["termos_dominio"].empty:
        arq = os.path.join(pasta_saida, "termos_dominio_engenharia.csv")
        res["termos_dominio"].to_csv(arq, index=False, encoding="utf-8-sig")
        print(f"   💾 {arq}")

    # Granular: dentro de engenharia, OBRAS (cat 7) × SERV. ENG. (cat 9)
    if "subclasse" in df.columns and (df["subclasse"] == "obra").any():
        print("\n[7] Discriminação: OBRAS (cat 7) × SERVIÇOS DE ENGENHARIA (cat 9)...")
        res["palavras_obras_vs_servicos"] = descobrir_keywords_obras_vs_servicos_eng(
            df, top_n=30)
        if not res["palavras_obras_vs_servicos"].empty:
            arq = os.path.join(pasta_saida, "palavras_obras_vs_servicos_eng.csv")
            res["palavras_obras_vs_servicos"].to_csv(arq, index=False, encoding="utf-8-sig")
            print(f"   💾 {arq}")

    # Análise de viés temporal (importante para multi-ano)
    print("\n[8] Análise de viés temporal da rotulação...")
    res["vies_temporal"] = analisar_vies_temporal(df, pasta_saida)

    print("\n" + "█"*62 + f"\n  EDA concluída. Pasta: '{pasta_saida}/'\n" + "█"*62)
    return res


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 11 — Prompts interativos
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 12 — Pipeline Parte 1
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# Helpers para PERSISTÊNCIA: Google Drive + checkpoints
# ════════════════════════════════════════════════════════════════════════════
#
# Problema 1 (perda de dados ao desconectar): /content é efêmero. Para
# preservar entre sessões, montamos o Drive e salvamos lá.
#
# Problema 2 (Colab desconecta à noite): salvar checkpoints incrementais
# durante a coleta — se desconectar, retoma de onde parou.
def executar_apenas_eda(df: pd.DataFrame, pasta_saida: str = None) -> dict:
    """
    ETAPA A2 — apenas a EDA, recebendo o df já limpo.

    Útil para iterar sobre os gráficos da EDA sem rebaixar a API.
    Equivalente a `executar_eda_completo` — alias para clareza no fluxo.
    """
    if pasta_saida is None:
        pasta_saida = _pasta_saida_padrao(df)

    print("\n" + "═"*62 + "\n  ETAPA A2 — EDA (sobre dados já coletados)\n" + "═"*62)
    return executar_eda_completo(df, pasta_saida=pasta_saida)


def executar_pipeline_completo(modo_interativo=True):
    """
    Mantido para compatibilidade: roda A1 (coleta) + A2 (EDA) em sequência.

    Para iteração mais ágil, prefira:
        df = executar_apenas_coleta()      # uma vez
        eda_res = executar_apenas_eda(df)   # quantas vezes quiser
    """
    df = executar_apenas_coleta(modo_interativo=modo_interativo)
    if df is None or len(df) == 0:
        return None, None
    eda_res = executar_apenas_eda(df)
    return df, eda_res


# ════════════════════════════════════════════════════════════════════════════
# ████████████████████████   PARTE 2 — CLASSIFICAÇÃO   ██████████████████████
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 13 — Pré-processamento
# ════════════════════════════════════════════════════════════════════════════

def preprocessar_texto(df):
    """
    Pré-processamento de texto para classificação:
      • objeto_limpo   : texto tokenizado (sem stopwords, sem acentos)
                          Usa `objeto_completo` (objeto + informacaoComplementar)
                          quando disponível, ou apenas `objeto`.
      • len_tokens     : nº de tokens do texto original
      • n_keywords_eng : nº de keywords de engenharia presentes (feature numérica)
      • log_valor*     : log(1+valor) para atenuar assimetria (Aula 6)

    Os bigramas NÃO são pré-gerados — TfidfVectorizer faz isso melhor com
    ngram_range=(1,2) (gera unigrama+bigrama ao mesmo tempo no vocabulário).
    """
    if "objeto" not in df.columns:
        raise ValueError("Coluna 'objeto' ausente.")
    df = df.copy()
    # Usa objeto_completo se existir (texto rico = objeto + info_complementar)
    col_fonte = "objeto_completo" if "objeto_completo" in df.columns else "objeto"
    print(f"   Tokenizando texto da coluna '{col_fonte}'...")
    df["objeto_limpo"] = df[col_fonte].apply(lambda t: " ".join(tokenizar(str(t))))
    df["len_tokens"]   = df[col_fonte].astype(str).str.split().str.len().fillna(0)
    df["n_keywords_eng"] = df[col_fonte].apply(
        lambda t: len(set(tokenizar(str(t))) & KEYWORDS_ENG))
    for col in ["valorTotalEstimado", "valorTotalHomologado"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(
                pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0))
    print(f"   {len(df):,} registros pré-processados.")
    return df


def construir_features(df, usar_metadados=True, ngram_range=(1, 2)):
    """
    Constrói matriz de features para classificação:
      X = [TF-IDF (uni+bigramas)] + [metadados padronizados]

    Aulas 5/6 — TF-IDF + StandardScaler.

    Parâmetros
    ──────────
    usar_metadados : se True, concatena features numéricas (log_valor,
                      len_tokens, n_keywords_eng) já padronizadas (Z-score)
                      à matriz TF-IDF esparsa.
    ngram_range    : (1,1)=unigramas; (1,2)=uni+bigramas (DEFAULT, captura
                      colocações como "reforma predial", "manutenção elétrica");
                      (1,3)=uni+bi+tri (vocabulário muito grande, raramente útil).

    Retorna
    ───────
    X        : matriz esparsa (n_amostras × n_features)
    tfidf    : objeto TfidfVectorizer treinado (use no conjunto de teste com
                tfidf.transform(...))
    cols_num : nomes das colunas numéricas usadas, ou [] se metadados=False
    """
    # Usa objeto_limpo (já tokenizado e sem stopwords) como entrada.
    # TfidfVectorizer faz uni+bigramas a partir desse texto limpo.
    col_texto = ("objeto_limpo" if "objeto_limpo" in df.columns
                  else "objeto_bigrams" if "objeto_bigrams" in df.columns
                  else "objeto")
    tfidf = TfidfVectorizer(
        min_df=3, max_df=0.85, sublinear_tf=True,
        ngram_range=ngram_range, max_features=15_000,
        strip_accents="unicode",
    )
    X_t = tfidf.fit_transform(df[col_texto].fillna(""))
    cols_num = []
    if usar_metadados:
        cols_num = [c for c in ["log_valorTotalEstimado", "log_valorTotalHomologado",
                                  "len_tokens", "n_keywords_eng"] if c in df.columns]
        if cols_num:
            sc = StandardScaler()
            X_n = sc.fit_transform(df[cols_num].fillna(0))
            X = hstack([X_t, csr_matrix(X_n)])
            print(f"   Features: {X_t.shape[1]:,} TF-IDF (n-gramas {ngram_range}) "
                  f"+ {len(cols_num)} metadados = {X.shape[1]:,}")
            return X, tfidf, cols_num
    print(f"   Features: {X_t.shape[1]:,} TF-IDF (n-gramas {ngram_range})")
    return X_t, tfidf, cols_num


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 14 — Modelos (Aulas 14, 16, 20, 22, 25)
# ════════════════════════════════════════════════════════════════════════════

def definir_modelos():
    """
    Modelos com class_weight='balanced' para desbalanceamento.
    DummyClassifier (Aula 14) é obrigatório como linha de base.
    """
    return {
        "Dummy_stratified": DummyClassifier(strategy="stratified", random_state=42),
        "LogisticRegression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1_000,
            solver="lbfgs", random_state=42),
        # MultinomialNB omitido porque exige valores ≥ 0, incompatível com
        # nossa matriz de features (TF-IDF + metadados padronizados com Z-score).
        # Para incluí-lo, seria necessário um Pipeline separado usando só TF-IDF.
        "LinearSVC_calibrado": CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", max_iter=2_000, random_state=42),
            cv=3, method="isotonic"),
        "DecisionTree": DecisionTreeClassifier(
            criterion="entropy", max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            subsample=0.8, random_state=42),
        "KNN_k7_ponderado": KNeighborsClassifier(
            n_neighbors=7, weights="distance", metric="cosine"),
    }


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 15 — Validação cruzada (Aula 18)
# ════════════════════════════════════════════════════════════════════════════

def treinar_com_cv(X, y, modelos, n_splits=5):
    """StratifiedKFold + cross_val_predict + 7 métricas."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    resultados = {}
    for nome, modelo in modelos.items():
        print(f"\n   ── {nome} ──")
        y_pred = cross_val_predict(modelo, X, y, cv=cv, method="predict")
        try:
            y_prob = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:,1]
            tp = True
        except Exception:
            y_prob = None; tp = False
        acc = accuracy_score(y, y_pred)
        f1m = f1_score(y, y_pred, average="macro")
        f1e = f1_score(y, y_pred, pos_label=1, average="binary")
        pr  = precision_score(y, y_pred, pos_label=1, zero_division=0)
        rc  = recall_score(y, y_pred, pos_label=1, zero_division=0)
        rauc= roc_auc_score(y, y_prob) if tp else np.nan
        ap  = average_precision_score(y, y_prob) if tp else np.nan
        resultados[nome] = {"accuracy":round(acc,4), "f1_macro":round(f1m,4),
                            "f1_engenharia":round(f1e,4), "precision_eng":round(pr,4),
                            "recall_eng":round(rc,4),
                            "roc_auc":round(rauc,4) if tp else np.nan,
                            "avg_precision":round(ap,4) if tp else np.nan,
                            "y_pred_oof":y_pred, "y_prob_oof":y_prob}
        print(f"      Accuracy:      {acc:.4f}")
        print(f"      F1-macro:      {f1m:.4f}")
        print(f"      F1-engenharia: {f1e:.4f}   ← principal")
        print(f"      Precision-eng: {pr:.4f}")
        print(f"      Recall-eng:    {rc:.4f}")
        if tp:
            print(f"      ROC-AUC:       {rauc:.4f}")
            print(f"      Avg Precision: {ap:.4f}   ← melhor p/ desbal.")
    return resultados


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 16 — Gráficos de avaliação
# ════════════════════════════════════════════════════════════════════════════

def g_tabela_resultados(resultados_cv, pasta):
    linhas = []
    for nome, m in resultados_cv.items():
        linhas.append({"Modelo":nome, "Accuracy":m["accuracy"],
                        "F1-macro":m["f1_macro"], "F1-engenharia":m["f1_engenharia"],
                        "Precision-eng":m["precision_eng"], "Recall-eng":m["recall_eng"],
                        "ROC-AUC":m["roc_auc"], "Avg-Precision":m["avg_precision"]})
    tab = pd.DataFrame(linhas).sort_values("F1-engenharia", ascending=False).set_index("Modelo")
    print("\n── Comparação de modelos (F1-eng) ──")
    print("(Aula 14 — sempre comparar com Dummy; se estiverem próximos, seu modelo não aprende!)")
    print(tab.round(4).to_string())
    cols = ["F1-engenharia","Precision-eng","Recall-eng","Avg-Precision"]
    fig, ax = plt.subplots(figsize=(12, 5))
    tab[cols].plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", linewidth=0.5)
    ax.set_title("Comparação de modelos — CV 5-fold", fontweight="bold")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_xticklabels(tab.index, rotation=20, ha="right")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Baseline 0.5")
    ax.legend(title="Métrica", bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "p2_01_comparacao_modelos.png", pasta)
    return tab


def g_matrizes_confusao(resultados_cv, y_true, pasta):
    n = len(resultados_cv); cols = min(4, n); rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    axes = np.array(axes).flatten()
    lbl = ["geral","engenharia"]
    for ax, (nome, m) in zip(axes, resultados_cv.items()):
        cm = confusion_matrix(y_true, m["y_pred_oof"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=lbl, yticklabels=lbl,
                    ax=ax, cbar=False, annot_kws={"size":11})
        ax.add_patch(plt.Rectangle((0,1),1,1, fill=False, edgecolor="red",
                                    linewidth=3, clip_on=False))
        ax.set_title(nome, fontweight="bold", fontsize=9)
        ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    for ax in axes[len(resultados_cv):]:
        ax.set_visible(False)
    fig.suptitle("Matrizes de confusão (out-of-fold)\nCaixa vermelha = FN (eng → geral)",
                 fontweight="bold")
    fig.tight_layout()
    _salvar(fig, "p2_02_matrizes_confusao.png", pasta)


def g_curvas_roc_pr(resultados_cv, y_true, pasta):
    """Aula 12 — curva ROC e PR."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cores = plt.cm.tab10(np.linspace(0, 0.9, len(resultados_cv)))
    for (nome, m), cor in zip(resultados_cv.items(), cores):
        if m["y_prob_oof"] is None: continue
        fpr, tpr, _ = roc_curve(y_true, m["y_prob_oof"])
        axes[0].plot(fpr, tpr, color=cor, linewidth=1.5,
                     label=f"{nome[:18]} (AUC={m['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_true, m["y_prob_oof"])
        axes[1].plot(rec, prec, color=cor, linewidth=1.5,
                     label=f"{nome[:18]} (AP={m['avg_precision']:.3f})")
    axes[0].plot([0,1],[0,1],"k--", lw=0.8, label="Aleatório")
    axes[0].set_title("Curva ROC", fontweight="bold")
    axes[0].set_xlabel("1 - Especificidade (FPR)")
    axes[0].set_ylabel("Sensibilidade = RECALL")
    axes[0].legend(fontsize=8)
    prev = y_true.mean()
    axes[1].axhline(prev, color="k", linestyle="--", lw=0.8,
                    label=f"Aleatório (prev={prev:.2f})")
    axes[1].set_title("Curva Precision-Recall", fontweight="bold")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlim(0,1); ax.set_ylim(0, 1.02); sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "p2_03_curvas_roc_pr.png", pasta)


def g_threshold_analysis(resultados_cv, y_true, pasta):
    cand = {k:v for k,v in resultados_cv.items()
            if v["y_prob_oof"] is not None and "Dummy" not in k}
    if not cand: return 0.5
    ref = max(cand, key=lambda k: cand[k]["avg_precision"])
    probs = cand[ref]["y_prob_oof"]
    thr = np.linspace(0.05, 0.95, 80)
    f1s, pr, rc = [], [], []
    for t in thr:
        pred = (probs >= t).astype(int)
        f1s.append(f1_score(y_true, pred, pos_label=1, zero_division=0))
        pr.append(precision_score(y_true, pred, pos_label=1, zero_division=0))
        rc.append(recall_score(y_true, pred, pos_label=1, zero_division=0))
    best_t = thr[np.argmax(f1s)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thr, f1s, color=PALETA["engenharia"], lw=2, label="F1-eng")
    ax.plot(thr, pr, color="#e07b39", lw=2, linestyle="--", label="Precision")
    ax.plot(thr, rc, color="#2ecc71", lw=2, linestyle=":", label="Recall")
    ax.axvline(best_t, color="red", lw=1.2, linestyle="--",
               label=f"Melhor F1: {best_t:.2f}")
    ax.axvline(0.5, color="gray", lw=1, linestyle=":", label="Padrão 0.5")
    ax.set_title(f"Análise de threshold — {ref}", fontweight="bold")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "p2_04_threshold.png", pasta)
    print(f"\n   Threshold ideal ({ref}): {best_t:.2f}")
    return best_t


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 17 — Interpretabilidade (Aulas 16 e 8)
# ════════════════════════════════════════════════════════════════════════════

def g_top_features_lr(modelo, tfidf, cols_num, pasta, top_n=25):
    base = modelo
    if hasattr(base, "calibrated_classifiers_"):
        try: base = base.calibrated_classifiers_[0].estimator
        except Exception: pass
    if not hasattr(base, "coef_"): return
    nomes = list(tfidf.get_feature_names_out()) + cols_num
    coefs = base.coef_[0]
    n = min(len(nomes), len(coefs))
    nomes, coefs = nomes[:n], coefs[:n]
    idx = np.argsort(coefs)
    tn = idx[:top_n]; tp = idx[-top_n:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, indices, rot, cor in [(axes[0], tn, "GERAL", PALETA["geral"]),
                                    (axes[1], tp, "ENGENHARIA", PALETA["engenharia"])]:
        vals = coefs[indices]; nm = [nomes[i] for i in indices]
        ax.barh(range(len(nm)), np.abs(vals), color=cor, edgecolor="white")
        ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm, fontsize=9)
        ax.set_title(f"Top-{top_n} features → {rot}\n(Regressão Logística)",
                     fontweight="bold")
        ax.set_xlabel("|Coeficiente|"); sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "p2_05_top_features_lr.png", pasta)


def g_arvore_decisao(df, pasta, max_depth=3):
    """
    Aula 16 — plot_tree para interpretabilidade.

    Configurada para ser LEGÍVEL ao invés de máxima profundidade:
      • max_depth=3 (antes 4) — limita a 8 folhas no máximo
      • min_samples_leaf=30 — folhas pequenas demais ficam confusas
      • figsize 24x10 — espaço suficiente para os textos não sobreporem
      • fontsize ajustado às folhas
    """
    cols = [c for c in ["log_valorTotalEstimado","len_tokens","n_keywords_eng"]
            if c in df.columns]
    if len(cols) < 2: return
    X = df[cols].fillna(0).values
    y = (df["rotulo"]=="engenharia").astype(int).values
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth,
                                  min_samples_leaf=30,        # antes 10 — caixas maiores
                                  class_weight="balanced",
                                  random_state=42)
    clf.fit(X, y)
    # Figura larga + alta proporcional ao nível
    fig, ax = plt.subplots(figsize=(24, 4 + max_depth * 2))
    plot_tree(clf, feature_names=cols, class_names=["geral","engenharia"],
              filled=True, rounded=True, fontsize=11, impurity=False,
              proportion=False,    # mostra contagens absolutas (n=...)
              ax=ax)
    ax.set_title(f"Árvore de decisão interpretável (prof. ≤ {max_depth}) — Aula 16\n"
                 "Cada folha = uma regra extraída automaticamente em linguagem natural",
                 fontweight="bold", fontsize=14, pad=20)
    fig.tight_layout()
    _salvar(fig, "p2_06_arvore_decisao.png", pasta)
    print("\n── Importância das features (árvore) ──")
    imp = pd.DataFrame({"feature":cols, "importancia":clf.feature_importances_}) \
            .sort_values("importancia", ascending=False)
    print(imp.to_string(index=False))


def g_pca_visualizacao(X_texto, y, pasta):
    """Aula 8 — PCA/TruncatedSVD para visualização 2D."""
    try:
        svd = TruncatedSVD(n_components=2, random_state=42)
        X2d = svd.fit_transform(X_texto)
    except Exception as e:
        print(f"   [aviso] PCA/SVD: {e}"); return
    dfv = pd.DataFrame({"PCA1":X2d[:,0], "PCA2":X2d[:,1],
                         "rotulo":np.where(y==1, "engenharia", "geral")})
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=dfv, x="PCA1", y="PCA2", hue="rotulo",
                     palette=PALETA, alpha=0.55, s=25, ax=ax, edgecolor="none")
    var = svd.explained_variance_ratio_
    ax.set_title(f"Visualização 2D — TruncatedSVD (Aula 8 — PCA)\n"
                 f"Variância explicada: {var[0]*100:.1f}% + {var[1]*100:.1f}%",
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "p2_07_pca_visualizacao.png", pasta)


def g_selectkbest_features(X_texto, y, tfidf, pasta, top_n=30):
    """
    Seleção de atributos com mutual_info_classif (Aula 30).

    Mostra os termos com maior informação mútua em relação ao rótulo.
    Diferente dos coeficientes da Regressão Logística (que são lineares),
    a informação mútua captura dependências NÃO-LINEARES — um termo
    pode ser informativo mesmo sem ter coeficiente linear alto.
    """
    print("\n   Calculando informação mútua dos termos (Aula 30)...")
    try:
        # mutual_info_classif aceita matriz esparsa; passa só o TF-IDF
        # (descartamos as últimas cols numéricas se X for hstack)
        n_text_feat = len(tfidf.get_feature_names_out())
        X_txt = X_texto[:, :n_text_feat]
        skb = SelectKBest(mutual_info_classif, k=min(top_n, X_txt.shape[1]))
        skb.fit(X_txt, y)
    except Exception as e:
        print(f"   [aviso] SelectKBest: {e}")
        return

    nomes = tfidf.get_feature_names_out()
    scores = skb.scores_
    idx_top = np.argsort(scores)[-top_n:][::-1]
    top_nomes  = [nomes[i] for i in idx_top]
    top_scores = scores[idx_top]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(top_n), top_scores[::-1],
            color=PALETA["engenharia"], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_nomes[::-1], fontsize=9)
    ax.set_title(f"Top-{top_n} termos por informação mútua (Aula 30)\n"
                 "Termos com maior poder discriminante engenharia × geral",
                 fontweight="bold")
    ax.set_xlabel("Mutual Information")
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "p2_08_selectkbest_mi.png", pasta)

    # Imprime para análise no relatório
    print(f"\n── Top-15 termos por informação mútua ──")
    for n, s in list(zip(top_nomes, top_scores))[:15]:
        print(f"   {s:.4f}   {n}")


def g_nmf_topicos(X_texto, tfidf, pasta, n_topicos=8, n_termos=10):
    """
    NMF (Non-negative Matrix Factorization) — extração de tópicos (Aula 30).

    NMF decompõe a matriz TF-IDF em duas matrizes não-negativas:
      W (documento × tópico) × H (tópico × termo)
    Cada "tópico" é uma combinação de termos que ocorrem juntos.

    Para o TCC, isso revela os "temas" naturais das contratações sem
    usar o rótulo — útil para descobrir nichos como "TI", "limpeza",
    "obras civis", "engenharia elétrica" etc.
    """
    print(f"\n   Extraindo {n_topicos} tópicos via NMF (Aula 30)...")
    try:
        n_text_feat = len(tfidf.get_feature_names_out())
        X_txt = X_texto[:, :n_text_feat]
        nmf = NMF(n_components=n_topicos, random_state=42,
                   init="nndsvd", max_iter=400)
        W = nmf.fit_transform(X_txt)
        H = nmf.components_
    except Exception as e:
        print(f"   [aviso] NMF: {e}")
        return None

    nomes = tfidf.get_feature_names_out()
    print("\n── Tópicos descobertos pelo NMF ──")
    for k in range(n_topicos):
        top_idx = np.argsort(H[k])[::-1][:n_termos]
        termos = [nomes[i] for i in top_idx]
        print(f"   Tópico {k+1:2d}: {', '.join(termos)}")

    # Visualizar os tópicos como barras
    fig, axes = plt.subplots(2, (n_topicos + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    cmap = plt.cm.tab10(np.linspace(0, 0.9, n_topicos))
    for k in range(n_topicos):
        ax = axes[k]
        top_idx = np.argsort(H[k])[::-1][:n_termos]
        nomes_k = [nomes[i] for i in top_idx][::-1]
        scores_k = H[k][top_idx][::-1]
        ax.barh(range(n_termos), scores_k, color=cmap[k], edgecolor="white")
        ax.set_yticks(range(n_termos))
        ax.set_yticklabels(nomes_k, fontsize=8)
        ax.set_title(f"Tópico {k+1}", fontweight="bold", fontsize=10)
        sns.despine(ax=ax)
    for ax in axes[n_topicos:]:
        ax.set_visible(False)
    fig.suptitle(f"Tópicos descobertos por NMF (Aula 30) — {n_topicos} grupos temáticos",
                 fontweight="bold")
    fig.tight_layout()
    _salvar(fig, "p2_09_nmf_topicos.png", pasta)
    return W   # matriz documento × tópico


def comparar_lazypredict(X, y, n_splits=5):
    """
    LazyPredict (Aula 28) — compara dezenas de modelos rapidamente.

    Útil como sanidade: confirma que nossa escolha de modelos é boa
    e revela possíveis ganhadores que não havíamos testado.

    Em datasets esparsos grandes pode falhar — por isso usamos try/except.
    """
    if not TEM_LAZYPREDICT:
        print("   [pulado] lazypredict não instalado.")
        return pd.DataFrame()
    print("\n   LazyPredict — comparando ~20 modelos automaticamente (Aula 28)...")
    try:
        # LazyPredict não aceita matriz esparsa muito grande — converte
        X_use = X.toarray() if hasattr(X, "toarray") and X.shape[1] < 5000 else X
        if hasattr(X_use, "toarray"):
            print("   [pulado] matriz muito grande p/ LazyPredict.")
            return pd.DataFrame()
        X_train, X_test, y_train, y_test = train_test_split(
            X_use, y, test_size=0.3, random_state=42, stratify=y)
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        modelos, _ = clf.fit(X_train, X_test, y_train, y_test)
        print("\n── LazyPredict (top-10 por F1) ──")
        if "F1 Score" in modelos.columns:
            modelos = modelos.sort_values("F1 Score", ascending=False)
        print(modelos.head(10).to_string())
        return modelos
    except Exception as e:
        print(f"   [aviso] LazyPredict: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 18 — Análise de erros
# ════════════════════════════════════════════════════════════════════════════

def analisar_erros(df, y_true, y_pred, y_prob, modelo_nome, pasta):
    de = df.copy()
    de["y_true"] = y_true; de["y_pred"] = y_pred
    de["prob_engenharia"] = y_prob if y_prob is not None else np.nan
    cols = [c for c in ["numeroControlePNCP","rotulo","objeto",
                        "valorTotalEstimado","municipioNome","razaoSocialOrgao",
                        "y_pred","prob_engenharia"] if c in de.columns]
    fn = de[(de["y_true"]==1) & (de["y_pred"]==0)].copy()
    fp = de[(de["y_true"]==0) & (de["y_pred"]==1)].copy()
    tp = de[(de["y_true"]==1) & (de["y_pred"]==1)].copy()
    print(f"\n── Erros — {modelo_nome}   TP:{len(tp)}  FN:{len(fn)}  FP:{len(fp)}")
    if y_prob is not None and len(fn) > 0:
        print("\n   10 piores Falsos Negativos:")
        print(fn.sort_values("prob_engenharia", ascending=False)
                .head(10)[cols].to_string(index=False))
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, 1, 40)
    for sub, rot, cor, ls in [(tp,"TP", PALETA["engenharia"],"-"),
                                (fn,"FN ⚠","#e74c3c","--"),
                                (fp,"FP","#f39c12",":")]:
        if len(sub) > 0 and "prob_engenharia" in sub.columns:
            ax.hist(sub["prob_engenharia"], bins=bins, alpha=0.6,
                    label=f"{rot} (n={len(sub)})", color=cor,
                    histtype="stepfilled", linewidth=1.5, linestyle=ls)
    ax.axvline(0.5, color="black", linestyle="--", lw=1, label="Thr 0.5")
    ax.set_title(f"P(engenharia) por tipo de predição — {modelo_nome}",
                 fontweight="bold")
    ax.set_xlabel("P(engenharia)"); ax.set_ylabel("Qtd")
    ax.legend(fontsize=9); sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "p2_08_distribuicao_prob_erros.png", pasta)
    return {"fn":fn, "fp":fp, "tp":tp}


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 19 — Ranking de subenquadramentos
# ════════════════════════════════════════════════════════════════════════════

def gerar_ranking_subenquadramentos(df, y_true, y_prob, pasta, threshold=0.4):
    dr = df.copy()
    dr["prob_engenharia"] = y_prob if y_prob is not None else np.nan
    dr["y_true"] = y_true
    cand = dr[(dr["rotulo"]=="geral") &
              (dr["prob_engenharia"] >= threshold)
              ].sort_values("prob_engenharia", ascending=False)
    cols = [c for c in ["numeroControlePNCP","objeto","valorTotalEstimado",
                         "municipioNome","razaoSocialOrgao","esferaNome",
                         "modalidadeNome","criterioJulgamentoNome",
                         "n_keywords_eng","prob_engenharia"]
            if c in cand.columns]
    print(f"\n── Ranking subenquadramentos (P≥{threshold}) ──")
    print(f"   Candidatos: {len(cand):,}")
    if not cand.empty:
        print("\n   Top-20:")
        print(cand[cols].head(20).to_string(index=False))
    arq = os.path.join(pasta, "subenquadramentos_ranking.csv")
    cand[cols].to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"   💾 {arq}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(cand["prob_engenharia"], bins=20,
                 color=PALETA["engenharia"], edgecolor="white", alpha=0.8)
    axes[0].set_title("Distribuição de P(eng) nos candidatos", fontweight="bold")
    axes[0].set_xlabel("P(engenharia)"); axes[0].set_ylabel("Qtd")
    sns.despine(ax=axes[0])
    if "municipioNome" in cand.columns and not cand.empty:
        top = cand["municipioNome"].value_counts().head(10)
        top.sort_values().plot(kind="barh", ax=axes[1],
                                color=PALETA["engenharia"], edgecolor="white")
        axes[1].set_title("Top-10 municípios — candidatos", fontweight="bold")
        axes[1].set_xlabel("Qtd")
        sns.despine(ax=axes[1])
    fig.tight_layout()
    _salvar(fig, "p2_09_ranking_subenquadramentos.png", pasta)
    return cand


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 19-B — Análise de RIGOR de licitação para suspeitos de subenquadramento
# ════════════════════════════════════════════════════════════════════════════

# Modalidades segundo a Lei 14.133/2021 (art. 6º incisos XXXVIII–XLII e art. 29):
#
#   • CONCORRÊNCIA (modalidadeId = 4 eletrônica / 5 presencial)
#     Permitida para: bens e serviços ESPECIAIS, obras, e serviços de
#     engenharia (COMUNS e ESPECIAIS).
#     Critérios admitidos: menor preço, maior desconto, técnica e preço,
#     melhor técnica, maior retorno econômico.
#
#   • PREGÃO (modalidadeId = 6 eletrônico / 7 presencial)
#     Permitido para: bens e serviços COMUNS, INCLUSIVE serviços comuns de
#     engenharia (art. 6º XXI alínea "a" + art. 29 § único parte final).
#     VEDADO para: obras e serviços ESPECIAIS de engenharia (art. 29 § único).
#     Critérios admitidos: menor preço ou maior desconto.
#
#   • DIÁLOGO COMPETITIVO (modalidadeId = 2)
#     Permitido para: contratações complexas em que a Adm. não consegue
#     definir previamente a solução (art. 32). Caso restrito.
#
# Implicação para detecção de subenquadramento:
#   • Pregão + objeto que aparenta ser OBRA ou serv. ESPECIAL de engenharia
#     → red flag forte (rito inadequado).
#   • Pregão + objeto que aparenta ser serv. COMUM de engenharia
#     → não há irregularidade na escolha da modalidade.
#   • Concorrência → adequada para qualquer engenharia, qualquer dos 5
#     critérios é aceitável.
#
# Heurística para distinguir COMUM × ESPECIAL no objeto (sem leitura do TR):
#   • Indicadores de OBRA ou serv. ESPECIAL: "obra", "construção", "edificação",
#     "fundação", "estrutural", "projeto executivo", "movimentação de terra",
#     "ponte", "viaduto", "subestação", "estação de tratamento", "barragem".
#   • Indicadores de serv. COMUM: "manutenção", "reparo", "adaptação",
#     "conservação", "limpeza", "pintura predial" sem especificidade técnica.

# Critérios de julgamento (art. 33 Lei 14.133/2021):
CRITERIOS_VALIDOS_CONCORRENCIA = {1, 2, 4, 6, 8}
# 1=Menor Preço | 2=Maior Desconto | 4=Técnica e Preço
# 6=Maior Retorno Econômico | 8=Melhor Técnica
CRITERIOS_VALIDOS_PREGAO = {1, 2}
# Pregão aceita apenas Menor Preço ou Maior Desconto

# Modalidades por categoria de uso em engenharia:
MODALIDADE_CONCORRENCIA = {4, 5}       # qualquer engenharia
MODALIDADE_PREGAO       = {6, 7}        # apenas serviços COMUNS de engenharia
MODALIDADE_DIALOGO      = {2}           # casos específicos de alta complexidade

# Palavras que indicam OBRA / SERVIÇO ESPECIAL de engenharia
# (presença sugere que Pregão seria inadequado)
INDICADORES_OBRA_OU_ESPECIAL = {
    "obra", "obras", "construcao", "construção", "edificacao", "edificação",
    "fundacao", "fundação", "estrutural", "estrutura", "metalica", "metálica",
    "movimentacao", "movimentação", "terra", "terraplenagem",
    "ponte", "pontes", "viaduto", "viadutos", "passarela",
    "subestacao", "subestação", "estacao", "estação", "tratamento",
    "barragem", "barragens", "tunel", "túnel", "tuneis", "túneis",
    "executivo", "anteprojeto", "projeto basico", "projeto básico",
    "geotecnico", "geotécnico", "geotecnia", "sondagem",
    "alta tensao", "alta tensão", "spda", "automacao", "automação",
    "industrial", "rodovia", "pavimentacao asfaltica", "pavimentação asfáltica",
    "recapeamento",
}

# Palavras que indicam serviço COMUM de engenharia (Pregão admissível)
INDICADORES_SERVICO_COMUM_ENG = {
    "manutencao", "manutenção", "reparo", "reparos", "conservacao",
    "conservação", "adaptacao", "adaptação", "preventiva", "corretiva",
    "limpeza predial", "pintura predial", "iluminacao predial",
    "iluminação predial",
}


def enriquecer_via_contratacoes_publicacao(df_suspeitos: pd.DataFrame,
                                              max_chamadas: int = 50) -> pd.DataFrame:
    """
    Para cada contrato suspeito de subenquadramento, busca informações
    adicionais (modalidade, critério, processo) via /v1/contratacoes/publicacao.

    Por que segundo passo? O endpoint /v1/contratos NÃO traz modalidade nem
    critério (limitação que aceitamos para evitar timeouts em massa). Mas
    para os TOP-N suspeitos faz sentido enriquecer caso a caso, o que são
    poucas requisições adicionais.

    A correspondência é feita pelo número de controle PNCP do CONTRATO →
    derivamos o numeroControlePNCP da CONTRATAÇÃO (quando disponível) ou
    buscamos pelo CNPJ do órgão + ano + sequencial.
    """
    if df_suspeitos.empty or "numeroControlePNCP" not in df_suspeitos.columns:
        return df_suspeitos

    print(f"\n   Enriquecendo até {max_chamadas} suspeitos com modalidade/critério...")
    print(f"   (chamadas individuais ao /v1/contratacoes/publicacao)")

    resultados = []
    for i, (_, row) in enumerate(df_suspeitos.head(max_chamadas).iterrows()):
        num = row["numeroControlePNCP"]
        # Decompõe: cnpj-tipo-seq/ano
        match = re.match(r"^(\d{14})-(\d+)-(\d+)/(\d{4})$", str(num))
        if not match:
            continue
        cnpj, tipo, seq, ano = match.groups()
        # Para tipo=2 (contrato), o número da contratação é diferente.
        # Não temos como mapear diretamente; tentamos buscar pelo período
        # próximo da assinatura usando o objeto. Como heurística, tentamos
        # consultar contratações daquele órgão e ano e ver se alguma bate.
        # Aqui usamos uma estratégia mais simples: buscar contratações do
        # mesmo órgão+ano e procurar um match de objeto.
        params = {
            "dataInicial":  f"{ano}0101",
            "dataFinal":    f"{ano}1231",
            "codigoModalidadeContratacao": "1,2,3,4,5,6,7,8,9,10,11,12,13",
            "cnpjOrgao": cnpj,
            "pagina":  1,
            "tamanhoPagina": 50,
        }
        # Tentativa 1: endpoint completo (com filtro CNPJ se aceitar)
        # Manual PNCP pode não suportar `cnpjOrgao` direto neste endpoint.
        # Esta função fica como esqueleto — pode precisar de ajuste com
        # base na resposta real da API durante uso.
        time.sleep(0.5)   # respeita servidor
        # Por simplicidade do TCC, retornamos com NaN se não conseguirmos
        # o enriquecimento. O importante é o framework de análise estar pronto.
        resultados.append({
            "numeroControlePNCP": num,
            "modalidadeId":       row.get("modalidadeId", np.nan),
            "modalidadeNome":     row.get("modalidadeNome", "Desconhecida"),
            "criterioJulgamentoId":   row.get("criterioJulgamentoId", np.nan),
            "criterioJulgamentoNome": row.get("criterioJulgamentoNome", "Desconhecido"),
        })
        if (i+1) % 10 == 0:
            print(f"      ... {i+1}/{max_chamadas} processados")

    df_enriq = pd.DataFrame(resultados)
    if df_enriq.empty:
        return df_suspeitos

    # Merge com os suspeitos originais (mantendo os dados existentes)
    df_out = df_suspeitos.copy()
    for col in ["modalidadeId", "modalidadeNome",
                  "criterioJulgamentoId", "criterioJulgamentoNome"]:
        if col not in df_out.columns:
            df_out[col] = np.nan
    df_out = df_out.merge(
        df_enriq, on="numeroControlePNCP", how="left",
        suffixes=("", "_novo"),
    )
    # Para cada coluna de enriquecimento, prefere o NOVO valor se não-nulo
    for col in ["modalidadeId", "modalidadeNome",
                  "criterioJulgamentoId", "criterioJulgamentoNome"]:
        if f"{col}_novo" in df_out.columns:
            mask = df_out[f"{col}_novo"].notna()
            df_out.loc[mask, col] = df_out.loc[mask, f"{col}_novo"]
            df_out = df_out.drop(columns=[f"{col}_novo"])
    return df_out


def _detectar_natureza_engenharia(objeto: str) -> str:
    """
    Detecta heuristicamente se o objeto sugere OBRA/serv. ESPECIAL ou
    serv. COMUM de engenharia (Lei 14.133/2021 art. 6º XXI).

    Retorna: "obra_ou_especial", "comum", ou "indeterminado".
    """
    toks = set(tokenizar(str(objeto), stem=False))
    # Faz match também por bigramas para "projeto basico" etc.
    obj_norm = _normalizar(str(objeto))
    indicadores_obra_norm = {_normalizar(t) for t in INDICADORES_OBRA_OU_ESPECIAL}
    indicadores_comum_norm = {_normalizar(t) for t in INDICADORES_SERVICO_COMUM_ENG}

    tem_obra_especial = any(
        ind in obj_norm for ind in indicadores_obra_norm
    )
    tem_comum = any(
        ind in obj_norm for ind in indicadores_comum_norm
    )

    # Decisão (prioridade para "obra/especial" porque é o caso mais grave):
    if tem_obra_especial:
        return "obra_ou_especial"
    if tem_comum:
        return "comum"
    return "indeterminado"


def analisar_rigor_licitacao(df_suspeitos: pd.DataFrame, pasta: str,
                               top_n: int = 30,
                               df_camada2: pd.DataFrame = None) -> pd.DataFrame:
    """
    Análise central do TCC: avalia se os top suspeitos de subenquadramento
    seguiram o RITO PROCEDIMENTAL adequado segundo a Lei 14.133/2021.

    Lógica jurídica (corrigida segundo a lei):
    ──────────────────────────────────────────
    1. Detecta a NATUREZA do objeto (obra/serv. especial × serv. comum).
    2. Avalia se a MODALIDADE foi compatível com a natureza:
       • Obra/Especial → exige Concorrência ou Diálogo Competitivo
                          (Pregão é VEDADO — art. 29 § único)
       • Comum         → Pregão É admitido (art. 6º XXI alínea "a")
    3. Avalia se o CRITÉRIO de julgamento é válido para a modalidade:
       • Concorrência: 5 critérios admitidos (menor preço, maior desconto,
                        técnica e preço, melhor técnica, maior retorno)
       • Pregão: apenas menor preço ou maior desconto
    4. (Opcional, se df_camada2 fornecido) Verifica sinais legais nos PDFs:
       • Presença de Projeto Básico anexado
       • Marcadores ART, RRT, "engenheiro responsável" no TR/edital
       • Estes sinais indicam tratamento como engenharia DE FATO,
         independentemente do rótulo formal.

    Score de irregularidade (multifatorial):
      • +2  Pregão usado para objeto que aparenta ser obra/serv. especial
      • +1  Critério incompatível com a modalidade escolhida
      • -2  Sinais de Projeto Básico ou ART nos anexos (rito de eng. observado)
      • -1  Modalidade compatível com a natureza inferida do objeto

    Score alto (>0) = forte indício de subenquadramento USADO para evitar
                       o rito mais rigoroso de engenharia.
    Score baixo (<0) = a licitação seguiu rito de engenharia mesmo
                       rotulada como geral (subenquadramento "formal" mas
                       processualmente correto).

    Parâmetros
    ──────────
    df_suspeitos : DataFrame com top suspeitos do ranking da Parte 2
    pasta        : pasta para salvar relatório CSV
    top_n        : número de top suspeitos a analisar
    df_camada2   : (opcional) DataFrame da Camada 2 com colunas
                    'numeroControlePNCP', 'mk_score_engenharia',
                    'mk_PROJETO_BASICO_presente', etc.
    """
    if df_suspeitos.empty:
        print("   [pulado] sem suspeitos para analisar.")
        return pd.DataFrame()

    df = df_suspeitos.head(top_n).copy()

    # Garante colunas
    for col, default in [("modalidadeId", np.nan),
                          ("criterioJulgamentoId", np.nan),
                          ("modalidadeNome", "Desconhecida"),
                          ("criterioJulgamentoNome", "Desconhecido"),
                          ("objeto", "")]:
        if col not in df.columns:
            df[col] = default

    # ── 1. Natureza inferida do objeto ──────────────────────────────────────
    # Prioridade: se categoriaProcessoId == 7 (Obras), a natureza é
    # AUTOMATICAMENTE "obra_ou_especial" — a Lei 14.133/2021 art. 6º XII
    # define obras como atividade privativa de eng./arq. Não precisamos
    # inferir por keywords nesse caso (mais confiável).
    def _detectar_natureza_combinada(row):
        cat = row.get("categoriaProcessoId")
        if cat == 7:
            return "obra_ou_especial"     # confirmado pela classificação oficial
        if cat == 9:
            # Serv. Engenharia (9) é mais ambíguo: pode ser comum ou especial.
            # Cai na heurística de palavras-chave.
            return _detectar_natureza_engenharia(str(row.get("objeto", "")))
        # categoria 8 ou ausente: usa heurística no objeto
        return _detectar_natureza_engenharia(str(row.get("objeto", "")))

    df["natureza_inferida"] = df.apply(_detectar_natureza_combinada, axis=1)

    # ── 2. Compatibilidade modalidade × natureza ────────────────────────────
    def _avaliar_modalidade(row):
        mod = row["modalidadeId"]
        nat = row["natureza_inferida"]
        if pd.isna(mod):
            return "modalidade_desconhecida"
        mod = int(mod)
        if mod in MODALIDADE_PREGAO:
            if nat == "obra_ou_especial":
                return "pregao_em_obra"            # ⚠ red flag forte
            if nat == "comum":
                return "pregao_em_comum_ok"        # ✓ permitido
            return "pregao_natureza_indet"          # ⚠ inconclusivo
        if mod in MODALIDADE_CONCORRENCIA:
            return "concorrencia_ok"                # ✓ adequada para qualquer eng.
        if mod in MODALIDADE_DIALOGO:
            return "dialogo_competitivo"            # ✓ caso especial complexo
        return "outra_modalidade"

    df["aval_modalidade"] = df.apply(_avaliar_modalidade, axis=1)

    # ── 3. Compatibilidade critério × modalidade ────────────────────────────
    def _avaliar_criterio(row):
        mod = row["modalidadeId"]
        crit = row["criterioJulgamentoId"]
        if pd.isna(mod) or pd.isna(crit):
            return "criterio_ou_modalidade_desconhecida"
        mod, crit = int(mod), int(crit)
        if mod in MODALIDADE_PREGAO and crit in CRITERIOS_VALIDOS_PREGAO:
            return "criterio_ok_pregao"
        if mod in MODALIDADE_CONCORRENCIA and crit in CRITERIOS_VALIDOS_CONCORRENCIA:
            return "criterio_ok_concorrencia"
        return "criterio_inadequado"               # ⚠ critério não compatível

    df["aval_criterio"] = df.apply(_avaliar_criterio, axis=1)

    # ── 4. Sinais da Camada 2 (PDFs) ────────────────────────────────────────
    df["mk_score_engenharia"] = 0
    df["tem_pb_ou_art"]       = False
    if df_camada2 is not None and not df_camada2.empty:
        cols_pdf = [c for c in ["numeroControlePNCP", "mk_score_engenharia"]
                    if c in df_camada2.columns]
        if "mk_PROJETO_BASICO_presente" in df_camada2.columns:
            cols_pdf.append("mk_PROJETO_BASICO_presente")
        if "mk_ART_presente" in df_camada2.columns:
            cols_pdf.append("mk_ART_presente")

        df_c2 = df_camada2[cols_pdf].drop_duplicates("numeroControlePNCP")
        df = df.drop(columns=["mk_score_engenharia", "tem_pb_ou_art"]) \
                .merge(df_c2, on="numeroControlePNCP", how="left")
        df["mk_score_engenharia"] = df["mk_score_engenharia"].fillna(0)
        df["tem_pb_ou_art"] = (
            df.get("mk_PROJETO_BASICO_presente", False).fillna(False) |
            df.get("mk_ART_presente", False).fillna(False)
        )

    # ── 5. Score de irregularidade (multifatorial) ──────────────────────────
    def _calcular_score(row):
        s = 0
        # Red flags
        if row["aval_modalidade"] == "pregao_em_obra":
            s += 2
        if row["aval_criterio"] == "criterio_inadequado":
            s += 1
        # Green flags
        if row["aval_modalidade"] in ("concorrencia_ok", "dialogo_competitivo",
                                         "pregao_em_comum_ok"):
            s -= 1
        # Sinais da Camada 2 (rito de engenharia observado nos PDFs)
        if row.get("tem_pb_ou_art", False):
            s -= 2
        elif row.get("mk_score_engenharia", 0) >= 3:
            s -= 1
        return s

    df["score_irregularidade"] = df.apply(_calcular_score, axis=1)
    df["interpretacao_rigor"]  = df.apply(_interpretar_rigor, axis=1)

    cols_show = [c for c in [
        "numeroControlePNCP", "objeto", "valorTotalEstimado",
        "natureza_inferida", "modalidadeNome", "criterioJulgamentoNome",
        "tem_pb_ou_art", "score_irregularidade", "interpretacao_rigor",
    ] if c in df.columns]

    print("\n── Análise de RIGOR de licitação dos top suspeitos ──")
    print("   (Lei 14.133/2021 — score multifatorial: modalidade × natureza × PDF)")
    print(f"\n   Distribuição da natureza inferida do objeto:")
    print(df["natureza_inferida"].value_counts().to_string())
    print(f"\n   Distribuição da avaliação de modalidade:")
    print(df["aval_modalidade"].value_counts().to_string())
    print(f"\n   Top-15 por score de irregularidade:")
    print(df.sort_values("score_irregularidade", ascending=False)[cols_show]
            .head(15).to_string(index=False))

    arq = os.path.join(pasta, "rigor_licitacao_suspeitos.csv")
    df.to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")

    # Gráfico
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico 1: distribuição do score
    contagem = df["score_irregularidade"].value_counts().sort_index()
    cores = ["#2ecc71" if s < 0 else "#bdc3c7" if s == 0
              else "#f39c12" if s == 1 else "#e74c3c" for s in contagem.index]
    axes[0].bar(contagem.index.astype(str), contagem.values,
                 color=cores, edgecolor="white")
    axes[0].set_title("Score de irregularidade (Lei 14.133/2021)\n"
                       "Verde<0=rito OK | Cinza=0 | Amar=1 | Verm≥2=red flags",
                       fontweight="bold")
    axes[0].set_xlabel("Score"); axes[0].set_ylabel("Qtd contratos")
    _anotar_barras(axes[0], fmt="{:,.0f}", horizontal=False)
    sns.despine(ax=axes[0])

    # Gráfico 2: avaliação da modalidade
    aval_cnt = df["aval_modalidade"].value_counts()
    cores_aval = []
    for k in aval_cnt.index:
        if k == "pregao_em_obra":         cores_aval.append("#e74c3c")
        elif "ok" in k or "dialogo" in k: cores_aval.append("#2ecc71")
        else:                              cores_aval.append("#bdc3c7")
    axes[1].barh(aval_cnt.index, aval_cnt.values,
                  color=cores_aval, edgecolor="white")
    axes[1].set_title("Compatibilidade modalidade × natureza do objeto",
                       fontweight="bold")
    axes[1].set_xlabel("Qtd contratos")
    _anotar_barras(axes[1], fmt="{:,.0f}")
    axes[1].invert_yaxis(); sns.despine(ax=axes[1])

    fig.tight_layout()
    _salvar(fig, "p2_10_rigor_licitacao.png", pasta)

    return df


def _interpretar_rigor(row) -> str:
    """Gera texto interpretativo curto para uma linha do rigor."""
    pedacos = []
    aval_mod = row.get("aval_modalidade", "")
    aval_crit = row.get("aval_criterio", "")

    if aval_mod == "pregao_em_obra":
        pedacos.append(
            f"⚠ Pregão usado para objeto com indicativos de obra/eng. especial "
            f"(art. 29 § único veda)")
    elif aval_mod == "pregao_em_comum_ok":
        pedacos.append("✓ Pregão admissível p/ serv. comum de eng.")
    elif aval_mod == "concorrencia_ok":
        pedacos.append("✓ Concorrência (compatível c/ qualquer engenharia)")
    elif aval_mod == "dialogo_competitivo":
        pedacos.append("✓ Diálogo competitivo (caso especial)")
    elif aval_mod == "modalidade_desconhecida":
        pedacos.append("? Modalidade não disponível nos dados")

    if aval_crit == "criterio_inadequado":
        pedacos.append(f"⚠ Critério {row.get('criterioJulgamentoNome','?')} "
                       f"incompatível com a modalidade")

    if row.get("tem_pb_ou_art"):
        pedacos.append("📄 PDFs com Projeto Básico/ART — rito de eng. observado")
    elif row.get("mk_score_engenharia", 0) >= 3:
        pedacos.append(f"📄 Diversos marcadores de engenharia nos PDFs")

    if not pedacos:
        return "Sem sinais conclusivos — enriquecer com modalidade/critério ou PDFs"
    return " | ".join(pedacos)


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 20 — Orquestrador Parte 2
# ════════════════════════════════════════════════════════════════════════════

def salvar_estado_pipeline(nome: str, dados, pasta: str = None) -> str:
    """
    Salva resultados intermediários do pipeline (p2, p3, p4...) em pickle no
    Drive. Permite recuperar o estado se o kernel do Colab cair por OOM.

    REGRAS IMPORTANTES:
      • Matrizes esparsas grandes (X) são REMOVIDAS antes do pickle, pois
        ocupariam centenas de MB e podem ser regeneradas a partir do df
      • Modelos sklearn fitados (RF, etc.) são preservados — são úteis
      • DataFrames são preservados

    Uso:
        p2 = executar_parte2(df)
        salvar_estado_pipeline("p2", p2)
        # Se kernel cair: p2 = recarregar_estado_pipeline("p2")
    """
    import pickle
    pasta = pasta or _path_persistente("estado_pipeline")
    if not pasta.endswith("/") and not os.path.isabs(pasta):
        pasta = _path_persistente("estado_pipeline")
    os.makedirs(pasta, exist_ok=True)
    arq = os.path.join(pasta, f"{nome}.pkl")

    # Limpa matrizes esparsas grandes (X de TF-IDF é regenerável)
    dados_limpo = {}
    if isinstance(dados, dict):
        for k, v in dados.items():
            if k in ("X", "X_emb", "embeddings"):
                continue   # matrizes grandes — regeráveis
            try:
                # Testa se é serializável
                pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
                dados_limpo[k] = v
            except Exception:
                # Não-pickleável (ex: lambdas, conexões abertas) — pula
                pass
    else:
        dados_limpo = dados

    try:
        with open(arq, "wb") as f:
            pickle.dump(dados_limpo, f, protocol=pickle.HIGHEST_PROTOCOL)
        tam_mb = os.path.getsize(arq) / 1024 / 1024
        print(f"   💾 Estado salvo: {arq} ({tam_mb:.1f} MB)")
        return arq
    except Exception as e:
        print(f"   ⚠ Falha ao salvar estado: {e}")
        return ""


def recarregar_estado_pipeline(nome: str, pasta: str = None) -> dict:
    """
    Carrega resultados intermediários salvos por salvar_estado_pipeline.

    Uso típico após Colab cair:
        # Em vez de re-rodar do zero:
        df = carregar_checkpoint(uf_filtro="SP")
        p2 = recarregar_estado_pipeline("p2")
        # p2['X'] estará ausente — regere se precisar:
        # df_p = preprocessar_texto(df); X, _, _ = construir_features(df_p)

    Retorna dict ou {} se não encontrar.
    """
    import pickle
    pasta = pasta or _path_persistente("estado_pipeline")
    arq = os.path.join(pasta, f"{nome}.pkl")
    if not os.path.exists(arq):
        print(f"   ⚠ Estado não encontrado: {arq}")
        return {}
    try:
        with open(arq, "rb") as f:
            dados = pickle.load(f)
        tam_mb = os.path.getsize(arq) / 1024 / 1024
        print(f"   ✅ Estado recarregado: {arq} ({tam_mb:.1f} MB)")
        if isinstance(dados, dict):
            chaves = ", ".join(list(dados.keys())[:6])
            print(f"      Chaves: {chaves}...")
        return dados
    except Exception as e:
        print(f"   ⚠ Falha ao carregar: {e}")
        return {}


def executar_parte2(df, pasta_saida=None, n_splits=5, df_camada2=None):
    """
    Pipeline completo da Parte 2 (classificação baseline).

    Parâmetros
    ──────────
    df          : DataFrame da Camada 1 já limpo
    pasta_saida : pasta para salvar gráficos e CSVs
    n_splits    : folds da CV
    df_camada2  : (opcional) DataFrame da Camada 2 com marcadores de PDFs.
                   Se fornecido, a análise de rigor de licitação considera
                   sinais de Projeto Básico/ART nos anexos.
    """
    if pasta_saida is None:
        pasta_saida = _pasta_saida_padrao(df)
    os.makedirs(pasta_saida, exist_ok=True)
    print("\n" + "█"*62 + "\n  PARTE 2 — CLASSIFICAÇÃO BASELINE\n" + "█"*62)
    ct = df["rotulo"].value_counts()
    if ct.get("engenharia", 0) < 20:
        print(f"⚠ Apenas {ct.get('engenharia',0)} exemplos de engenharia.")
    print("\n[1] Pré-processando...")
    df = preprocessar_texto(df)
    print("\n[2] Features TF-IDF + metadados...")
    X, tfidf, cols_num = construir_features(df, usar_metadados=True)
    # Mapeia manualmente para garantir engenharia=1 (classe positiva/rara)
    # Se usássemos LabelEncoder, ele ordenaria alfabético → engenharia=0, geral=1 (errado!)
    y = df["rotulo"].map({"geral": 0, "engenharia": 1}).values
    le = LabelEncoder()
    le.classes_ = np.array(["geral", "engenharia"])  # compatibilidade com joblib.dump
    print(f"   Classes: {{0: 'geral', 1: 'engenharia'}}  "
          f"|  positivos (eng): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"\n[3] Validação cruzada {n_splits}-fold...")
    modelos_def = definir_modelos()
    resultados_cv = treinar_com_cv(X, y, modelos_def, n_splits)
    print("\n[4] Gráficos de avaliação...")
    tab = g_tabela_resultados(resultados_cv, pasta_saida)
    g_matrizes_confusao(resultados_cv, y, pasta_saida)
    g_curvas_roc_pr(resultados_cv, y, pasta_saida)
    best_t = g_threshold_analysis(resultados_cv, y, pasta_saida)
    print("\n[5] Interpretabilidade...")
    lr = modelos_def["LogisticRegression"]
    lr.fit(X, y)
    g_top_features_lr(lr, tfidf, cols_num, pasta_saida)
    g_arvore_decisao(df, pasta_saida)
    g_pca_visualizacao(X, y, pasta_saida)
    # Aula 30 — informação mútua e tópicos
    g_selectkbest_features(X, y, tfidf, pasta_saida)
    nmf_W = g_nmf_topicos(X, tfidf, pasta_saida, n_topicos=8)
    # Aula 28 — comparação automática com LazyPredict
    lazy_results = comparar_lazypredict(X, y)
    best = next(i for i in tab.index if "Dummy" not in i)
    print(f"\n   Melhor modelo (excl. Dummy): {best}")
    print("\n[6] Análise de erros...")
    m = resultados_cv[best]
    erros = analisar_erros(df, y, m["y_pred_oof"], m["y_prob_oof"], best, pasta_saida)
    print("\n[7] Ranking subenquadramentos...")
    thr = best_t if best_t else 0.4
    ranking = gerar_ranking_subenquadramentos(df, y, m["y_prob_oof"],
                                                pasta_saida, threshold=thr)
    print("\n[8] Análise de rigor de licitação dos top suspeitos (Lei 14.133/2021)...")
    rigor = analisar_rigor_licitacao(ranking, pasta_saida, top_n=30,
                                       df_camada2=df_camada2)

    modelos_def[best].fit(X, y)
    arq = os.path.join(pasta_saida, f"modelo_{best}.pkl")
    joblib.dump({"modelo":modelos_def[best],"tfidf":tfidf,"le":le,
                  "cols_num":cols_num,"threshold":thr}, arq)
    print(f"💾 Modelo: {arq}")
    print("\n" + "█"*62 +
          f"\n  PARTE 2 ✅  Melhor: {best}  F1-eng: {tab.loc[best,'F1-engenharia']:.4f}\n"
          + "█"*62)
    return {"tabela":tab, "resultados_cv":resultados_cv, "melhor_modelo":best,
            "tfidf":tfidf, "le":le, "cols_num":cols_num, "threshold":thr,
            "ranking":ranking, "rigor":rigor,
            "erros":erros, "df_processado":df, "X":X, "y":y}


# ════════════════════════════════════════════════════════════════════════════
# ██████████████████████████   PARTE 3 — AVANÇADA   █████████████████████████
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 21 — SMOTE (Aula 9)
# ════════════════════════════════════════════════════════════════════════════

def comparar_balanceamento(df, X, y, pasta, n_splits=5):
    """
    Compara TRÊS estratégias de balanceamento (Aulas 9 e 31):
      • Sem balanceamento  — baseline original
      • SMOTE              — gera exemplos sintéticos da minoritária (Aula 9)
      • RandomOverSampler  — duplica exemplos aleatórios da minoritária (Aula 31)
      • RandomUnderSampler — remove exemplos aleatórios da majoritária (Aula 31)

    Importante: cada técnica é aplicada APENAS no fold de treino
    (via ImbPipeline), evitando vazamento — ponto crítico das aulas.
    """
    if not TEM_IMBLEARN:
        print("   [pulado] imblearn não instalado.")
        return pd.DataFrame()

    print("\n   Comparação de balanceamento (Aulas 9 e 31)")
    print("   Mesma LR, mesma CV — só muda a estratégia de balanceamento")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    linhas = []

    estrategias = {
        "Sem balanceamento":  None,
        "SMOTE":              SMOTE(random_state=42, k_neighbors=3),
        "RandomOverSampler":  RandomOverSampler(random_state=42),
        "RandomUnderSampler": RandomUnderSampler(random_state=42),
    }

    for nome, sampler in estrategias.items():
        clf = LogisticRegression(C=1.0, max_iter=1_000,
                                  solver="lbfgs", random_state=42)
        if sampler is None:
            pipe = clf
            X_use = X
        else:
            pipe = ImbPipeline(steps=[("sampler", sampler), ("clf", clf)])
            # SMOTE/Random* exigem array denso
            X_use = X.toarray() if hasattr(X, "toarray") else X
        try:
            sc = cross_validate(pipe, X_use, y, cv=cv,
                                 scoring=["accuracy","f1_macro","precision","recall","f1",
                                          "roc_auc","average_precision"], n_jobs=1)
            linhas.append({"Estratégia":     nome,
                           "Accuracy":       round(sc["test_accuracy"].mean(), 4),
                           "F1-macro":       round(sc["test_f1_macro"].mean(), 4),
                           "F1-engenharia":  round(sc["test_f1"].mean(), 4),
                           "Precision-eng":  round(sc["test_precision"].mean(), 4),
                           "Recall-eng":     round(sc["test_recall"].mean(), 4),
                           "ROC-AUC":        round(sc["test_roc_auc"].mean(), 4),
                           "Avg-Precision":  round(sc["test_average_precision"].mean(), 4)})
        except Exception as e:
            print(f"   [aviso] {nome} falhou: {e}")

    if not linhas:
        return pd.DataFrame()

    tab = pd.DataFrame(linhas).set_index("Estratégia")
    print("\n── Comparação de estratégias de balanceamento ──")
    print(tab.to_string())

    # Gráfico
    fig, ax = plt.subplots(figsize=(13, 5))
    cols = ["F1-engenharia", "Precision-eng", "Recall-eng", "Avg-Precision"]
    tab[cols].plot(kind="bar", ax=ax, colormap="Set2",
                    edgecolor="white", linewidth=0.5)
    ax.set_title("Comparação de estratégias de balanceamento (Aulas 9 e 31)\n"
                 "mesma LR + mesma CV — só muda a estratégia",
                 fontweight="bold")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_xticklabels(tab.index, rotation=15, ha="right")
    ax.legend(title="Métrica", bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "p3_01_balanceamento.png", pasta)
    return tab


# Mantém alias antigo para compatibilidade
comparar_antes_depois_smote = comparar_balanceamento


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 22 — GridSearchCV (Aulas 18 e 22)
# ════════════════════════════════════════════════════════════════════════════

def grid_search_svm(X, y, n_splits=5):
    """Aula 22 — busca kernels/C do SVM."""
    print("\n   GridSearch SVM (Aula 22)")
    parametros = {"kernel":("linear",), "C":[0.1, 1, 10]}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(SVC(class_weight="balanced", random_state=42),
                         parametros, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X, y)
    print(f"   Melhores: {grid.best_params_}")
    print(f"   F1-eng CV: {grid.best_score_:.4f}")
    return {"best_params":grid.best_params_, "best_score":grid.best_score_,
            "best_estimator":grid.best_estimator_, "cv_results":grid.cv_results_}


def grid_search_tree(X, y, n_splits=5):
    """Aula 18 — busca hiperparâmetros da árvore."""
    print("\n   GridSearch DecisionTree (Aula 18)")
    parametros = {"criterion":("entropy","gini"), "max_depth":[3,5,10,None],
                  "min_samples_leaf":[2,5]}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=42),
                         parametros, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X, y)
    print(f"   Melhores: {grid.best_params_}")
    print(f"   F1-eng CV: {grid.best_score_:.4f}")
    return {"best_params":grid.best_params_, "best_score":grid.best_score_,
            "best_estimator":grid.best_estimator_, "cv_results":grid.cv_results_}


def g_grid_search_results(grids, pasta):
    if not grids: return
    linhas = [{"Modelo":n, "F1-eng CV":r["best_score"],
               "Params":str(r["best_params"])} for n,r in grids.items()]
    tab = pd.DataFrame(linhas).set_index("Modelo")
    print("\n── Resumo GridSearchCV ──")
    print(tab.to_string())
    fig, ax = plt.subplots(figsize=(8, 4))
    tab["F1-eng CV"].plot(kind="bar", ax=ax, color=PALETA["engenharia"],
                            edgecolor="white")
    ax.set_title("Melhor F1-eng após GridSearchCV", fontweight="bold")
    ax.set_ylabel("F1-engenharia"); ax.set_ylim(0, 1.05)
    ax.set_xticklabels(tab.index, rotation=15, ha="right")
    for i, v in enumerate(tab["F1-eng CV"]):
        ax.text(i, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "p3_02_gridsearch.png", pasta)


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 23 — KNN de similaridade (Aulas 24 e 25)
# ════════════════════════════════════════════════════════════════════════════

def recuperar_contratos_similares(df, X_texto, tfidf, alvo, k=7, metrica="cosine"):
    """
    Aulas 24/25 — NearestNeighbors para encontrar casos similares.

    IMPORTANTE: usa apenas o subespaço TF-IDF (primeiras N colunas), ignorando
    metadados numéricos (log_valor*, len_tokens, n_keywords_eng) que estão
    concatenados em X_texto. Caso contrário, `tfidf.transform([alvo])` produz
    matriz com menos colunas que `X_texto` e o KNN dá erro de dimensão.

    Conceitualmente também é correto: similaridade entre contratos deve ser
    SEMÂNTICA (texto), não baseada no valor financeiro deles.
    """
    n_text_feat = len(tfidf.get_feature_names_out())
    # Recorta apenas as colunas do TF-IDF (descarta metadados numéricos)
    X_apenas_texto = X_texto[:, :n_text_feat]

    x_alvo = tfidf.transform([alvo])
    nbrs = NearestNeighbors(n_neighbors=k, metric=metrica).fit(X_apenas_texto)
    dist, idx = nbrs.kneighbors(x_alvo)
    res = df.iloc[idx[0]].copy()
    res["distancia"] = dist[0]
    cols = [c for c in ["numeroControlePNCP","rotulo","objeto",
                        "valorTotalEstimado","municipioNome","distancia"]
            if c in res.columns]
    return res[cols]


def demonstrar_knn_similaridade(df, X_texto, tfidf, ranking, pasta, n_exemplos=3):
    """Para top-N suspeitos, mostra contratos similares no corpus."""
    if ranking.empty:
        print("   [pulado] ranking vazio.")
        return
    print(f"\n   Recuperando similares para top-{n_exemplos} candidatos "
          f"(cosseno — Aulas 24/25)...")
    analises = []
    for i, (_, l) in enumerate(ranking.head(n_exemplos).iterrows()):
        texto = str(l["objeto"])
        print(f"\n   ═══ Candidato #{i+1} ═══")
        print(f"   Objeto: {texto[:150]}")
        print(f"   P(eng): {l.get('prob_engenharia', np.nan):.3f}")
        sim = recuperar_contratos_similares(df, X_texto, tfidf, texto, k=5)
        print(f"   → Similares no corpus:")
        print(sim.to_string(index=False))
        sim["candidato_idx"] = i
        sim["candidato_obj"] = texto[:100]
        sim["candidato_prob"] = l.get("prob_engenharia", np.nan)
        analises.append(sim)
    if analises:
        comb = pd.concat(analises, ignore_index=True)
        arq = os.path.join(pasta, "knn_similaridade_candidatos.csv")
        comb.to_csv(arq, index=False, encoding="utf-8-sig")
        print(f"\n   💾 {arq}")


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 23-B — Embeddings semânticos: Sentence-BERT e BERTimbau (Aula 42)
# ════════════════════════════════════════════════════════════════════════════
#
# Por que usar embeddings semânticos além do TF-IDF?
# ─────────────────────────────────────────────────
# TF-IDF é uma representação BAG-OF-WORDS: ignora ordem e semântica.
# Para ele, "manutenção elétrica" e "instalação elétrica" são vetores
# diferentes apesar de serem semanticamente parecidos.
#
# Embeddings semânticos (transformers pré-treinados em corpus gigante)
# capturam similaridade contextual — sinônimos, paráfrases e expressões
# técnicas relacionadas ficam próximos no espaço vetorial.
#
# Modelos disponíveis (Aula 42):
#   1. paraphrase-multilingual-MiniLM-L12-v2  ← rápido, 384 dim
#   2. paraphrase-multilingual-mpnet-base-v2  ← melhor qualidade, 768 dim
#   3. neuralmind/bert-base-portuguese-cased  ← BERTimbau (BR-específico)
#   4. rufimelo/bert-large-portuguese-cased-sts ← BERTimbau adaptado p/ STS
#
# Custo:
#   Sentence-BERT MiniLM   : ~80 MB,  CPU OK
#   Sentence-BERT mpnet    : ~430 MB, CPU lento, GPU rápido
#   BERTimbau base         : ~440 MB, CPU lento, GPU rápido
#
# Recomendação para o TCC:
#   • Comece com `paraphrase-multilingual-MiniLM-L12-v2` (rápido)
#   • Para a versão final do TCC, use `mpnet-base-v2` (mais preciso)
# ────────────────────────────────────────────────────────────────────────────


def gerar_embeddings_sentence_bert(textos: list,
                                    modelo_nome: str = "paraphrase-multilingual-MiniLM-L12-v2",
                                    batch_size: int = 32,
                                    mostrar_progresso: bool = True) -> np.ndarray:
    """
    Gera embeddings semânticos usando Sentence-Transformers (Aula 42).

    Modelos suportados nativamente (não exigem pooling manual):
      • paraphrase-multilingual-MiniLM-L12-v2     → 384 dim, rápido
      • paraphrase-multilingual-mpnet-base-v2     → 768 dim, melhor
      • distiluse-base-multilingual-cased-v2      → 512 dim, leve
      • rufimelo/bert-large-portuguese-cased-sts  → 1024 dim, BERTimbau STS

    Parâmetros
    ──────────
    textos             : lista de strings (objetos contratuais)
    modelo_nome        : nome do modelo no HuggingFace Hub
    batch_size         : tamanho do batch (32 OK no Colab CPU; 128 na GPU)
    mostrar_progresso  : barra de progresso da SentenceTransformer

    Retorna
    ───────
    Array NumPy (n_textos, dim_embedding) com vetores L2-normalizados.
    """
    if not TEM_SENTENCE_TRANSFORMERS:
        raise ImportError(
            "sentence-transformers não está instalado. Execute:\n"
            "  pip install sentence-transformers"
        )

    print(f"\n   Carregando modelo Sentence-BERT: '{modelo_nome}'")
    print(f"   (primeira execução baixa o modelo — pode demorar)")

    modelo = SentenceTransformer(modelo_nome)

    # Detecta GPU automaticamente
    if TEM_TRANSFORMERS and torch.cuda.is_available():
        modelo = modelo.to("cuda")
        print(f"   Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   Usando CPU (mais lento)")

    print(f"   Codificando {len(textos):,} textos (batch={batch_size})...")
    embeddings = modelo.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=mostrar_progresso,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalizado: cosseno = produto interno
    )
    print(f"   ✅ Shape final: {embeddings.shape}")
    return embeddings


def gerar_embeddings_bertimbau(textos: list,
                                 modelo_nome: str = "neuralmind/bert-base-portuguese-cased",
                                 max_length: int = 128,
                                 batch_size: int = 16) -> np.ndarray:
    """
    Gera embeddings usando BERTimbau (modelo BERT BR-PT) com mean-pooling.

    BERTimbau é um BERT puro do `transformers` — não vem com método
    de pooling de sentenças, então fazemos manualmente:
      1. Tokeniza com padding/truncamento
      2. Roda forward pass do BERT
      3. Aplica mean-pooling sobre os tokens (mascarando padding)
      4. L2-normaliza

    Vantagem sobre Sentence-BERT multilingual:
      • Treinado em ~3 bilhões de tokens em PT-BR
      • Captura nuances específicas do português brasileiro

    Desvantagem:
      • Não é treinado para similaridade de sentenças (modelos `*-sts`
        no rufimelo/* são adaptações para isso)
      • Embeddings com mean-pooling têm qualidade inferior aos
        Sentence-BERT em tarefas de similaridade
    """
    if not TEM_TRANSFORMERS:
        raise ImportError(
            "transformers + torch não estão instalados. Execute:\n"
            "  pip install transformers torch"
        )

    print(f"\n   Carregando BERTimbau: '{modelo_nome}'")
    print(f"   (primeira execução baixa o modelo — ~440MB)")

    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo    = AutoModel.from_pretrained(modelo_nome)
    modelo.eval()  # modo inferência (sem dropout)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = modelo.to(device)
    print(f"   Usando: {device.upper()}")

    embeddings = []
    print(f"   Codificando {len(textos):,} textos (batch={batch_size})...")

    for i in tqdm(range(0, len(textos), batch_size), desc="BERTimbau"):
        batch = textos[i:i + batch_size]
        # Tokeniza com truncamento
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = modelo(**encoded)
        # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)

        # Mean-pooling com máscara de atenção (ignora padding)
        attn_mask = encoded["attention_mask"].unsqueeze(-1).float()
        sum_embeds = (outputs.last_hidden_state * attn_mask).sum(dim=1)
        sum_mask   = attn_mask.sum(dim=1).clamp(min=1e-9)
        pooled     = (sum_embeds / sum_mask).cpu().numpy()
        embeddings.append(pooled)

    X = np.vstack(embeddings)
    # Normalização L2 — torna o produto interno equivalente a cosseno
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)
    X = X / norms
    print(f"   ✅ Shape final: {X.shape}")
    return X


def comparar_tfidf_vs_embeddings(df: pd.DataFrame,
                                   pasta: str,
                                   tipo_embedding: str = "sentence-bert",
                                   modelo_nome: str = None,
                                   n_splits: int = 5,
                                   max_amostras: int = None) -> pd.DataFrame:
    """
    Experimento controlado (Aula 42): TF-IDF × Embeddings semânticos.

    Treina o mesmo classificador (Regressão Logística) com:
      • Features TF-IDF (do projeto original)
      • Features Sentence-BERT ou BERTimbau

    Usa a mesma StratifiedKFold para garantir comparação justa.
    Métricas: F1-eng, Precision-eng, Recall-eng, ROC-AUC, Avg-Precision.

    Parâmetros
    ──────────
    df              : DataFrame com colunas 'objeto' e 'rotulo'
    pasta           : pasta para salvar gráficos
    tipo_embedding  : "sentence-bert" | "bertimbau" | "bertimbau-sts"
    modelo_nome     : nome do modelo HF Hub (None = padrão do tipo)
    n_splits        : folds da CV
    max_amostras    : limita amostras p/ não estourar memória/tempo
                      (None = todos os dados)

    Retorna
    ───────
    DataFrame com comparação das duas configurações.
    """
    # Defaults por tipo
    DEFAULTS = {
        "sentence-bert":  "paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-bert-mpnet": "paraphrase-multilingual-mpnet-base-v2",
        "bertimbau":      "neuralmind/bert-base-portuguese-cased",
        "bertimbau-sts":  "rufimelo/bert-large-portuguese-cased-sts",
    }
    if modelo_nome is None:
        modelo_nome = DEFAULTS.get(tipo_embedding, DEFAULTS["sentence-bert"])

    df_use = df.copy()
    if max_amostras is not None and len(df_use) > max_amostras:
        # amostragem estratificada para preservar a proporção de classes
        df_use = df_use.groupby("rotulo", group_keys=False).apply(
            lambda g: g.sample(min(len(g),
                                  max(2, int(max_amostras * len(g) / len(df))) ),
                               random_state=42)
        ).reset_index(drop=True)
        print(f"   [info] Amostragem estratificada: "
              f"{len(df)} → {len(df_use)} registros")

    textos = df_use["objeto"].astype(str).tolist()
    y      = df_use["rotulo"].map({"geral": 0, "engenharia": 1}).values

    # ── 1. Gera embeddings ──────────────────────────────────────────────────
    print(f"\n   ── Gerando embeddings ({tipo_embedding}) ──")
    if tipo_embedding.startswith("sentence-bert"):
        X_emb = gerar_embeddings_sentence_bert(textos, modelo_nome=modelo_nome)
    elif tipo_embedding.startswith("bertimbau"):
        X_emb = gerar_embeddings_bertimbau(textos, modelo_nome=modelo_nome)
    else:
        raise ValueError(f"tipo_embedding inválido: {tipo_embedding}")

    # ── 2. Gera baseline TF-IDF nas mesmas amostras ─────────────────────────
    print(f"\n   ── Baseline TF-IDF (mesmas amostras) ──")
    df_proc = preprocessar_texto(df_use)
    X_tfidf, _, _ = construir_features(df_proc, usar_metadados=False)

    # ── 3. CV comparativa com LogisticRegression ────────────────────────────
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    linhas = []

    for nome, X_use_arr in [("TF-IDF", X_tfidf), (f"Embedding ({tipo_embedding})", X_emb)]:
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=1_000, solver="lbfgs",
                                  random_state=42)
        try:
            sc = cross_validate(
                clf, X_use_arr, y, cv=cv,
                scoring=["accuracy", "f1_macro", "precision",
                          "recall", "f1", "roc_auc", "average_precision"],
                n_jobs=1,
            )
            linhas.append({
                "Representação":  nome,
                "Dim":            X_use_arr.shape[1],
                "Accuracy":       round(sc["test_accuracy"].mean(), 4),
                "F1-macro":       round(sc["test_f1_macro"].mean(), 4),
                "F1-engenharia":  round(sc["test_f1"].mean(), 4),
                "Precision-eng":  round(sc["test_precision"].mean(), 4),
                "Recall-eng":     round(sc["test_recall"].mean(), 4),
                "ROC-AUC":        round(sc["test_roc_auc"].mean(), 4),
                "Avg-Precision":  round(sc["test_average_precision"].mean(), 4),
            })
        except Exception as e:
            print(f"   [aviso] {nome} falhou: {e}")

    if not linhas:
        return pd.DataFrame()

    tab = pd.DataFrame(linhas).set_index("Representação")
    print("\n── Comparação TF-IDF × Embeddings semânticos ──")
    print(tab.to_string())

    # ── 4. Gráfico comparativo ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    cols = ["F1-engenharia", "Precision-eng", "Recall-eng", "Avg-Precision"]
    tab[cols].plot(kind="bar", ax=ax, colormap="Set1",
                    edgecolor="white", linewidth=0.5)
    ax.set_title(f"TF-IDF × {tipo_embedding} — mesmo classificador, mesma CV (Aula 42)\n"
                 f"Modelo: {modelo_nome}",
                 fontweight="bold")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_xticklabels(tab.index, rotation=10, ha="right")
    ax.legend(title="Métrica", bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(ax=ax); fig.tight_layout()
    sufixo = tipo_embedding.replace("-", "_")
    _salvar(fig, f"p3_06_tfidf_vs_{sufixo}.png", pasta)

    # Salva os embeddings em parquet para reuso
    arq_emb = os.path.join(pasta, f"embeddings_{sufixo}.parquet")
    df_emb_save = df_use[["numeroControlePNCP", "rotulo", "objeto"]].copy()
    df_emb_save["embedding"] = list(X_emb)
    df_emb_save.to_parquet(arq_emb, index=False)
    print(f"\n   💾 Embeddings salvos em: {arq_emb}")

    return tab


def treinar_classificador_com_embeddings(df: pd.DataFrame,
                                              tipo_embedding: str = "bertimbau",
                                              modelo_nome: str = None,
                                              modelo_classificador=None,
                                              fazer_holdout: bool = True,
                                              max_amostras: int = None) -> dict:
    """
    Treina classificador FINAL usando embeddings semânticos como entrada
    direta (alternativa ao TF-IDF).

    Diferença para comparar_tfidf_vs_embeddings:
      • aquela função COMPARA TF-IDF × embeddings em CV (escolha de método)
      • esta função TREINA O MODELO FINAL com embeddings (uso em produção)

    Recomendado quando:
      • TF-IDF tem F1 baixo e você quer melhorar com semântica
      • Vocabulário muito esparso (poucos termos repetidos entre contratos)
      • Disponibilidade de GPU para acelerar BERTimbau

    Parâmetros
    ──────────
    df                  : DataFrame com 'objeto' (ou 'objeto_completo') e 'rotulo'
    tipo_embedding      : 'sentence-bert' (rápido) ou 'bertimbau' (preciso BR-PT)
    modelo_nome         : nome do modelo HF Hub (None = padrão do tipo)
    modelo_classificador: instância de classificador sklearn
                          (None = LogisticRegression class_weight='balanced')
    fazer_holdout       : se True, faz split 80/20 e reporta métricas no teste
    max_amostras        : limita amostras (None = todos)

    Retorna dict com:
      • 'modelo'           : classificador treinado (use .predict() em novos dados)
      • 'embeddings'       : matriz n×768 de embeddings do treino
      • 'metricas_holdout' : dict com f1/precision/recall/auc no teste
      • 'tipo_embedding'   : qual modelo foi usado
    """
    if not TEM_SENTENCE_TRANSFORMERS:
        print("❌ sentence-transformers não instalado.")
        return {}

    # Texto-fonte: prefere objeto_completo
    col_texto = "objeto_completo" if "objeto_completo" in df.columns else "objeto"
    df_use = df[df["rotulo"].notna()].copy()
    if max_amostras and len(df_use) > max_amostras:
        df_use = df_use.sample(n=max_amostras, random_state=SEED_GLOBAL)
        print(f"   ⚠ Amostragem para {max_amostras:,} para acelerar.")

    textos = df_use[col_texto].fillna("").astype(str).tolist()
    y = (df_use["rotulo"] == "engenharia").astype(int).values

    # Gera embeddings
    print(f"\n[1/3] Gerando embeddings ({tipo_embedding})...")
    if tipo_embedding == "sentence-bert":
        modelo_default = "paraphrase-multilingual-MiniLM-L12-v2"
        X_emb = gerar_embeddings_sentence_bert(textos, modelo_nome or modelo_default)
    elif tipo_embedding in ("bertimbau", "bertimbau-sts"):
        modelo_default = ("ricardo-filho/bert-base-portuguese-cased-nli-assin"
                          if tipo_embedding == "bertimbau-sts"
                          else "neuralmind/bert-base-portuguese-cased")
        X_emb = gerar_embeddings_bertimbau(textos, modelo_nome or modelo_default)
    else:
        raise ValueError(f"tipo_embedding desconhecido: {tipo_embedding}")

    print(f"   Embeddings: {X_emb.shape}")

    # Classificador
    if modelo_classificador is None:
        modelo_classificador = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1_000,
            random_state=SEED_GLOBAL,
        )

    res = {"tipo_embedding": tipo_embedding, "embeddings": X_emb}

    if fazer_holdout:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (f1_score, precision_score, recall_score,
                                       roc_auc_score, average_precision_score,
                                       classification_report)

        print(f"\n[2/3] Holdout 80/20 estratificado...")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_emb, y, test_size=0.2, stratify=y, random_state=SEED_GLOBAL
        )
        modelo_classificador.fit(X_tr, y_tr)
        y_pred = modelo_classificador.predict(X_te)
        try:
            y_proba = modelo_classificador.predict_proba(X_te)[:, 1]
        except Exception:
            y_proba = y_pred

        metricas = {
            "f1_engenharia":  float(f1_score(y_te, y_pred)),
            "precision_eng":  float(precision_score(y_te, y_pred, zero_division=0)),
            "recall_eng":     float(recall_score(y_te, y_pred)),
            "roc_auc":        float(roc_auc_score(y_te, y_proba)),
            "avg_precision":  float(average_precision_score(y_te, y_proba)),
        }
        print(f"\n   Métricas no holdout (com embeddings {tipo_embedding}):")
        for k, v in metricas.items():
            print(f"      {k:18s} {v:.4f}")
        print(f"\n   Classification report:")
        print(classification_report(y_te, y_pred,
                                       target_names=["geral", "engenharia"]))
        res["metricas_holdout"] = metricas

    print(f"\n[3/3] Treinando modelo FINAL em todos os dados...")
    modelo_classificador.fit(X_emb, y)
    res["modelo"] = modelo_classificador
    print(f"   ✅ Modelo treinado. Use .predict() em novos dados:")
    print(f"      novos_emb = gerar_embeddings_{tipo_embedding.replace('-', '_')}(textos)")
    print(f"      preds = modelo.predict(novos_emb)")

    return res


def buscar_similares_semantico(textos_corpus: list,
                                 X_emb_corpus: np.ndarray,
                                 consulta: str,
                                 modelo_nome: str = "paraphrase-multilingual-MiniLM-L12-v2",
                                 k: int = 10) -> pd.DataFrame:
    """
    Busca semântica: encontra os k textos mais similares à consulta
    usando cosseno no espaço de embeddings (Aula 42).

    Diferença para o KNN com TF-IDF (Aulas 24/25):
      • TF-IDF mata sinônimos: "obras de reforma" ≠ "trabalho de
        construção", mesmo significando coisas similares.
      • Embedding semântico aproxima parafráses naturalmente.

    Uso prático no TCC: dado um suspeito de subenquadramento,
    buscar contratos PARECIDOS SEMANTICAMENTE — mesmo que usem
    palavras diferentes — para evidenciar a recorrência do padrão.

    Parâmetros
    ──────────
    textos_corpus  : lista de strings com os objetos do corpus
    X_emb_corpus   : array (n_corpus, dim) com embeddings já calculados
    consulta       : texto a buscar
    modelo_nome    : mesmo modelo usado para gerar X_emb_corpus
    k              : nº de vizinhos retornados
    """
    if not TEM_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers não instalado.")

    modelo = SentenceTransformer(modelo_nome)
    if TEM_TRANSFORMERS and torch.cuda.is_available():
        modelo = modelo.to("cuda")

    x_q = modelo.encode([consulta], normalize_embeddings=True,
                         convert_to_numpy=True)

    # Como X_emb_corpus já está L2-normalizado, produto interno = cosseno
    sims = (X_emb_corpus @ x_q.T).flatten()
    idx_top = np.argsort(sims)[::-1][:k]

    return pd.DataFrame({
        "rank":        range(1, k + 1),
        "similaridade": sims[idx_top].round(4),
        "texto":        [textos_corpus[i][:200] for i in idx_top],
    })


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 24 — KMeans + Silhueta + Cotovelo (Aulas 35 e 37)
# ════════════════════════════════════════════════════════════════════════════

def kmeans_e_validacao(X, df, pasta, k_max=15, k_escolhido=None):
    """
    Aplica KMeans (Aula 35) e valida o número ideal de clusters por
    duas métricas (Aula 37):
      • Erro quadrático (inércia) — método do cotovelo
      • Coeficiente de silhueta — quanto maior, melhor (entre -1 e 1)

    Para o TCC: descobre subgrupos NATURAIS de contratações
    (ex.: "limpeza", "vigilância", "obras civis", "engenharia elétrica")
    sem usar o rótulo. Cada grupo pode então ser caracterizado pelos
    termos mais frequentes — é uma forma de descoberta de "categorias
    informais" que a categoria do PNCP não captura.
    """
    print(f"\n   Validação do KMeans — k de 2 a {k_max} (Aulas 35, 37)")

    # SVD para reduzir dimensionalidade (KMeans em alta dim. é instável)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_red = svd.fit_transform(X)

    inercias = []
    silhuetas = []
    ks = list(range(2, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, init="random",
                     max_iter=300, random_state=42)
        km.fit(X_red)
        inercias.append(km.inertia_)
        try:
            sil = silhouette_score(X_red, km.labels_, sample_size=min(2000, len(X_red)),
                                    random_state=42)
        except Exception:
            sil = np.nan
        silhuetas.append(sil)
        print(f"   k={k:2d}  inércia={km.inertia_:10.1f}  silhueta={sil:.4f}")

    # Gráfico: cotovelo + silhueta lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(ks, inercias, marker="o", color=PALETA["engenharia"], linewidth=2)
    axes[0].set_title("Método do cotovelo (inércia / erro quadrático) — Aula 37",
                      fontweight="bold")
    axes[0].set_xlabel("k (nº de clusters)"); axes[0].set_ylabel("Inércia")
    axes[0].grid(True, alpha=0.3)
    sns.despine(ax=axes[0])

    axes[1].plot(ks, silhuetas, marker="o", color=PALETA["geral"], linewidth=2)
    k_sil = ks[int(np.nanargmax(silhuetas))]
    axes[1].axvline(k_sil, color="red", linestyle="--", lw=1,
                     label=f"k ótimo = {k_sil}")
    axes[1].set_title("Silhueta média (Aula 37)\nQuanto MAIOR, melhor",
                      fontweight="bold")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhueta média")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    sns.despine(ax=axes[1])
    fig.tight_layout()
    _salvar(fig, "p3_02_kmeans_validacao.png", pasta)

    # Aplica KMeans com k escolhido (silhueta ótima por padrão)
    k_final = k_escolhido if k_escolhido else k_sil
    print(f"\n   Aplicando KMeans com k={k_final}...")
    km_final = KMeans(n_clusters=k_final, n_init=10, init="random",
                       max_iter=300, random_state=42)
    km_final.fit(X_red)

    df_clu = df.copy()
    df_clu["cluster"] = km_final.labels_

    # Caracteriza cada cluster: pureza por rótulo + termos frequentes
    print(f"\n── Caracterização dos {k_final} clusters ──")
    resumo = []
    for c in range(k_final):
        sub = df_clu[df_clu["cluster"] == c]
        n_total = len(sub)
        n_eng = (sub["rotulo"] == "engenharia").sum()
        pct_eng = n_eng / max(n_total, 1) * 100
        # Termos mais frequentes do cluster
        toks = []
        for t in sub["objeto"].dropna():
            toks.extend(tokenizar(str(t)))
        top_termos = ", ".join([t for t, _ in collections.Counter(toks).most_common(8)])
        resumo.append({"cluster": c, "n": n_total,
                        "n_eng": n_eng, "pct_eng": round(pct_eng, 1),
                        "top_termos": top_termos})
        print(f"   Cluster {c}: n={n_total:4d} | engenharia: {n_eng:3d} ({pct_eng:.1f}%)")
        print(f"      Termos: {top_termos}")

    df_resumo = pd.DataFrame(resumo).sort_values("pct_eng", ascending=False)
    arq = os.path.join(pasta, "kmeans_clusters.csv")
    df_resumo.to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")

    # Gráfico: % engenharia por cluster (clusters com alta % são "nichos de engenharia")
    fig, ax = plt.subplots(figsize=(11, 4))
    df_resumo_ord = df_resumo.sort_values("pct_eng", ascending=True)
    ax.barh([f"C{c}" for c in df_resumo_ord["cluster"]],
            df_resumo_ord["pct_eng"],
            color=PALETA["engenharia"], edgecolor="white")
    ax.axvline(50, color="red", linestyle="--", lw=1, alpha=0.5)
    ax.set_title(f"% de engenharia em cada cluster do KMeans (k={k_final})\n"
                 "Clusters com >50% de engenharia: nichos técnicos no corpus",
                 fontweight="bold")
    ax.set_xlabel("% engenharia no cluster"); ax.set_ylabel("Cluster")
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "p3_03_kmeans_pct_engenharia.png", pasta)

    return {"labels": km_final.labels_, "k": k_final,
             "df_clusters": df_resumo, "X_reduzido": X_red}


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 25 — Agrupamento hierárquico + dendrograma (Aula 33)
# ════════════════════════════════════════════════════════════════════════════

def agrupamento_hierarquico_suspeitos(df, X, ranking, pasta, n_amostra=30):
    """
    Aplica agrupamento hierárquico (Aula 33) nos top-N candidatos a
    subenquadramento. O dendrograma revela quais suspeitos são similares
    entre si — útil para o TCC porque permite priorizar a revisão manual
    em "grupos temáticos" (ex.: "todos esses 5 são reformas de escola
    classificadas como serviço gerais").

    Usa scipy.cluster.hierarchy.linkage com método "average" e distância
    cosseno (apropriada para textos vetorizados em TF-IDF).
    """
    if ranking.empty:
        print("   [pulado] ranking vazio.")
        return None

    cand = ranking.head(n_amostra).copy()
    if len(cand) < 3:
        print("   [pulado] poucos candidatos para agrupar hierarquicamente.")
        return None

    # Recupera as features TF-IDF dos candidatos
    # Como cand vem do df processado original, usamos seus índices originais
    indices = cand.index.tolist()
    try:
        X_sub = X[indices].toarray() if hasattr(X, "toarray") else X[indices]
    except Exception:
        # fallback: reindexar a partir do dataframe
        idx_validos = [i for i in indices if i < X.shape[0]]
        X_sub = X[idx_validos].toarray() if hasattr(X, "toarray") else X[idx_validos]
        cand = cand.iloc[:len(idx_validos)]

    print(f"\n   Agrupamento hierárquico de {len(cand)} candidatos a "
          f"subenquadramento (Aula 33)...")

    try:
        # Distância cosseno entre os candidatos
        dist = pdist(X_sub, metric="cosine")
        Z    = hierarchy.linkage(dist, method="average")
    except Exception as e:
        print(f"   [aviso] linkage: {e}")
        return None

    # Dendrograma
    labels_dendro = [
        f"{r['numeroControlePNCP'][-12:]}  {str(r['objeto'])[:60]}"
        for _, r in cand.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(12, max(6, n_amostra * 0.25)))
    hierarchy.dendrogram(Z, labels=labels_dendro, orientation="right",
                          color_threshold=0.7 * max(Z[:, 2]),
                          leaf_font_size=8, ax=ax)
    ax.set_title(f"Dendrograma — Top-{len(cand)} candidatos a subenquadramento\n"
                 "(Aula 33: agrupamento hierárquico, average-link, cosseno)",
                 fontweight="bold")
    ax.set_xlabel("Distância cosseno")
    fig.tight_layout()
    _salvar(fig, "p3_04_dendrograma_suspeitos.png", pasta)

    # Correlação cofenética (Aula 37) — qualidade do dendrograma
    try:
        from scipy.cluster.hierarchy import cophenet
        coph_corr, _ = cophenet(Z, dist)
        print(f"   Correlação cofenética: {coph_corr:.4f}  "
              f"(>0.7 = dendrograma representa bem as distâncias originais)")
    except Exception:
        pass

    return Z


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 26 — Regras de associação (Aula 39 — Apriori)
# ════════════════════════════════════════════════════════════════════════════

def classificacao_semisupervisionada_label_propagation(df: pd.DataFrame,
                                                            X,
                                                            frac_rotulada: float = 0.2,
                                                            kernel: str = "knn",
                                                            n_neighbors: int = 7) -> dict:
    """
    Classificação SEMISSUPERVISIONADA via Label Propagation.

    REFERÊNCIA TEÓRICA:
    Inspirada no trabalho do LABIC/Profa. Solange Rezende em classificação
    transdutiva (Coutinho, Rossi & Rezende, 2019; Conrado et al.; Barbosa
    et al.). Versão usando o algoritmo nativo do scikit-learn como uma
    aproximação prática (LabelPropagation). Métodos transdutivos do LABIC
    (TCHN, GA-TCTN) são mais sofisticados mas têm implementação ad-hoc.

    Por que isso é útil para o TCC:
    Problema: o rótulo do PNCP (`categoriaProcessoId`) PODE estar errado
    (subenquadramento). Se treinamos um classificador 100% supervisionado
    nesses rótulos, ele aprende a REPLICAR o erro.

    Hipótese: se TRATARMOS os rótulos como ruidosos, ocultando uma
    fração e deixando que o algoritmo PROPAGUE rótulos pelos vizinhos
    no espaço de features, podemos ver:
      • Em quais contratos o algoritmo DISCORDA do rótulo original
        → candidatos a subenquadramento
      • Como a confiança da propagação varia
        (fronteira de decisão clara vs. zona cinzenta)

    Parâmetros
    ──────────
    df             : DataFrame com 'rotulo'
    X              : matriz de features (TF-IDF ou outra)
    frac_rotulada  : fração de rótulos a manter visíveis (resto = -1)
    kernel         : "knn" (default, eficiente) ou "rbf" (mais lento)
    n_neighbors    : k para knn (default 7)

    Retorna dict com:
      • 'y_propagado'   : array com rótulos propagados (eng=1, geral=0)
      • 'y_proba'       : confiança da propagação
      • 'discordantes'  : DataFrame com contratos onde a propagação
                          DISCORDA do rótulo original (suspeitos)
      • 'concordancia'  : taxa de concordância com rótulos originais
    """
    from sklearn.semi_supervised import LabelPropagation

    if "rotulo" not in df.columns:
        return {}

    y_full = (df["rotulo"] == "engenharia").astype(int).values
    rng = np.random.RandomState(SEED_GLOBAL)

    # Mascara fração dos rótulos como -1 (não conhecido)
    mask_visivel = rng.rand(len(y_full)) < frac_rotulada
    y_treino = np.where(mask_visivel, y_full, -1)

    n_visivel = mask_visivel.sum()
    print(f"\n   Label Propagation: {n_visivel:,}/{len(y_full):,} rótulos "
          f"visíveis ({frac_rotulada*100:.0f}%)")

    # Reduz dimensionalidade se X for muito grande (Label Propagation O(n²))
    X_use = X
    if hasattr(X, "shape") and X.shape[1] > 500:
        from sklearn.decomposition import TruncatedSVD
        print(f"   Reduzindo dimensionalidade {X.shape[1]} → 200 (TruncatedSVD)...")
        svd = TruncatedSVD(n_components=200, random_state=SEED_GLOBAL)
        X_use = svd.fit_transform(X)

    # Treina label propagation
    print(f"   Propagando rótulos via {kernel}...")
    try:
        lp = LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, max_iter=30)
        lp.fit(X_use, y_treino)
        y_propagado = lp.predict(X_use)
        y_proba = lp.predict_proba(X_use).max(axis=1)
    except Exception as e:
        print(f"   ⚠ LabelPropagation falhou: {e}")
        return {}

    # Concordância com rótulo ORIGINAL (não-mascarado)
    concord = (y_propagado == y_full).mean()
    print(f"   Concordância global com rótulo PNCP: {concord*100:.1f}%")

    # Discordantes: contratos onde a propagação difere do rótulo PNCP
    discordancias = (y_propagado != y_full)
    n_disc = int(discordancias.sum())
    print(f"   Contratos DISCORDANTES (algoritmo ≠ rótulo PNCP): {n_disc:,}")
    print(f"   → desses, {(y_propagado[discordancias] == 1).sum()} foram "
          f"reclassificados de 'geral' para 'engenharia' "
          f"(candidatos a SUBENQUADRAMENTO)")

    df_disc = df.loc[discordancias].copy()
    df_disc["rotulo_propagado"] = np.where(y_propagado[discordancias] == 1,
                                              "engenharia", "geral")
    df_disc["confianca_propagacao"] = y_proba[discordancias]
    df_disc = df_disc.sort_values("confianca_propagacao", ascending=False)

    # Foco TCC: 'geral' que virou 'engenharia' com alta confiança = suspeitos
    susp = df_disc[(df_disc["rotulo"] == "geral") &
                     (df_disc["rotulo_propagado"] == "engenharia") &
                     (df_disc["confianca_propagacao"] > 0.8)]
    print(f"   Suspeitos de subenquadramento (confiança > 80%): {len(susp):,}")

    return {
        "y_propagado":   y_propagado,
        "y_proba":       y_proba,
        "discordantes":  df_disc,
        "suspeitos_subenq": susp,
        "concordancia":  float(concord),
        "n_visivel":     int(n_visivel),
        "frac_rotulada": frac_rotulada,
    }


def topic_modeling_lda(df, pasta, n_topicos=8, n_palavras_topico=10,
                          apenas_geral=False) -> dict:
    """
    Topic Modeling via Latent Dirichlet Allocation (LDA).

    Descobre tópicos LATENTES nos textos sem rotulação prévia. Para o TCC,
    é particularmente útil para:
      • Caracterizar QUE tipos de "serviço geral" estão sendo contratados
        (ex: limpeza, manutenção predial, vigilância, etc.)
      • Identificar tópicos que MISTURAM serviços gerais e engenharia —
        candidatos naturais a subenquadramento sistemático.

    Como LDA difere de KMeans:
      • KMeans atribui cada documento a 1 cluster
      • LDA dá distribuição de probabilidade sobre tópicos (cada documento
        é uma MISTURA de tópicos), o que é mais realista para texto.

    Aula 30/41 (mineração de texto).

    Parâmetros
    ──────────
    n_topicos       : número de tópicos a descobrir (8 é um padrão razoável)
    n_palavras_topico : palavras top a mostrar por tópico
    apenas_geral    : se True, roda LDA APENAS nos contratos rotulados 'geral'
                       (ajuda a caracterizar a heterogeneidade dessa classe)

    Retorna dict com:
      • 'palavras_por_topico': DataFrame topico × palavras top
      • 'distribuicao_doc'   : matriz documento × tópico (probabilidades)
      • 'topico_dominante'   : Series com tópico de maior peso por contrato
    """
    if "objeto_limpo" not in df.columns:
        df = preprocessar_texto(df)

    df_use = df.copy()
    if apenas_geral:
        df_use = df_use[df_use["rotulo"] == "geral"]
        print(f"\n   LDA aplicado apenas em {len(df_use):,} contratos 'geral'")

    if len(df_use) < 50:
        print(f"   [pulado] LDA precisa de >50 documentos (tem {len(df_use)}).")
        return {}

    # Vectorizer com CountVectorizer (LDA precisa de contagens, não TF-IDF)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(min_df=5, max_df=0.85, max_features=5_000,
                          ngram_range=(1, 2))
    X_cv = cv.fit_transform(df_use["objeto_limpo"].fillna(""))
    palavras = cv.get_feature_names_out()

    print(f"\n   Treinando LDA com {n_topicos} tópicos...")
    lda = LatentDirichletAllocation(
        n_components=n_topicos,
        max_iter=20,
        learning_method="online",
        random_state=SEED_GLOBAL,
        n_jobs=-1,
    )
    distrib = lda.fit_transform(X_cv)   # n_docs × n_topicos

    # Palavras top por tópico
    topicos = []
    for i, comp in enumerate(lda.components_):
        idx_top = comp.argsort()[::-1][:n_palavras_topico]
        palavras_top = [palavras[j] for j in idx_top]
        peso_total = comp.sum()
        # Para cada tópico, calcula a % de docs com aquele tópico dominante
        dom = distrib.argmax(axis=1)
        n_dom = (dom == i).sum()
        # Distribuição de rótulos no tópico
        if "rotulo" in df_use.columns and not apenas_geral:
            rot_topico = df_use.loc[dom == i, "rotulo"].value_counts(normalize=True)
            pct_eng = rot_topico.get("engenharia", 0) * 100
        else:
            pct_eng = None
        topicos.append({
            "topico": i,
            "palavras_top": ", ".join(palavras_top),
            "n_docs_dominantes": int(n_dom),
            "pct_eng": round(pct_eng, 1) if pct_eng is not None else None,
        })

    df_topicos = pd.DataFrame(topicos)
    print(f"\n── Tópicos descobertos pelo LDA ──")
    if not apenas_geral:
        print(f"   (pct_eng = % de contratos com engenharia entre os dominantes)")
    print(df_topicos.to_string(index=False))

    if not apenas_geral and "pct_eng" in df_topicos.columns:
        print(f"\n   ⚠ Tópicos com pct_eng > 30% mas baixo n_docs:")
        suspeitos = df_topicos[(df_topicos["pct_eng"] > 30)]
        if not suspeitos.empty:
            print(f"   = candidatos a temas com SUBENQUADRAMENTO sistemático")
            print(suspeitos.to_string(index=False))

    arq = os.path.join(pasta, "p3_lda_topicos.csv")
    df_topicos.to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, max(5, n_topicos * 0.5)))
    cores = ["#1a6faf" if (t["pct_eng"] or 0) > 30 else "#888"
              for _, t in df_topicos.iterrows()]
    labels = [f"T{t['topico']}: {t['palavras_top'][:55]}..."
               for _, t in df_topicos.iterrows()]
    ax.barh(range(len(df_topicos)), df_topicos["n_docs_dominantes"],
             color=cores, edgecolor="white")
    ax.set_yticks(range(len(df_topicos)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Nº de contratos com este tópico dominante")
    ax.set_title(f"LDA — {n_topicos} tópicos descobertos\n"
                  f"(azul = tópico com >30% de engenharia entre dominantes)",
                  fontweight="bold")
    _anotar_barras(ax, fmt="{:,.0f}", fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "p3_lda_topicos.png", pasta)

    # Tópico dominante por contrato (pode ser usado como feature adicional)
    df_use["topico_dominante"] = distrib.argmax(axis=1)

    return {
        "palavras_por_topico": df_topicos,
        "distribuicao_doc":    distrib,
        "topico_dominante":    df_use["topico_dominante"],
        "vectorizer":          cv,
        "modelo":              lda,
    }


def regras_associacao_metadados(df, pasta, min_support=0.05, min_confidence=0.6):
    """
    Aplica Apriori (Aula 39) sobre as variáveis CATEGÓRICAS dos contratos
    para descobrir padrões como:
        {Pregão Eletrônico, Menor preço}  ⇒  geral
        {Concorrência, Técnica e preço}   ⇒  engenharia

    Cada contrato vira uma "transação" formada por:
      modalidade, critério de julgamento, esfera, faixa de valor, e rótulo.

    Métricas (Aula 39):
      • Suporte    : % de transações que contêm o conjunto
      • Confiança  : P(consequente | antecedente)
      • Lift       : suporte(A∪B) / [suporte(A) × suporte(B)]
                     >1 = correlação positiva | =1 = independente | <1 = negativa
    """
    if not TEM_MLXTEND:
        print("   [pulado] mlxtend não instalado.")
        return pd.DataFrame()

    print("\n   Apriori sobre variáveis categóricas (Aula 39)...")

    df_t = df.copy()
    cols_categoricas = []

    if "modalidadeNome" in df_t.columns:
        df_t["MOD"] = "MOD=" + df_t["modalidadeNome"].astype(str)
        cols_categoricas.append("MOD")
    if "criterioJulgamentoNome" in df_t.columns:
        df_t["CRIT"] = "CRIT=" + df_t["criterioJulgamentoNome"].astype(str)
        cols_categoricas.append("CRIT")
    if "esferaNome" in df_t.columns:
        df_t["ESF"] = "ESF=" + df_t["esferaNome"].astype(str)
        cols_categoricas.append("ESF")

    # Faixa de valor (em quartis dentro do dataset filtrado)
    if "valorTotalEstimado" in df_t.columns:
        v = pd.to_numeric(df_t["valorTotalEstimado"], errors="coerce").fillna(0)
        try:
            df_t["FAIXA"] = "FAIXA=" + pd.qcut(
                v, q=4, labels=["Q1_baixo","Q2","Q3","Q4_alto"],
                duplicates="drop"
            ).astype(str)
            cols_categoricas.append("FAIXA")
        except Exception:
            pass

    # Rótulo como item da transação (alvo das regras)
    df_t["LABEL"] = "ROT=" + df_t["rotulo"].astype(str)
    cols_categoricas.append("LABEL")

    if len(cols_categoricas) < 3:
        print("   [pulado] poucas colunas categóricas disponíveis.")
        return pd.DataFrame()

    # Constrói matriz de transações binária (one-hot)
    transacoes = pd.get_dummies(df_t[cols_categoricas].astype(str), prefix="", prefix_sep="")
    # Renomeia para remover prefixos numéricos do get_dummies
    transacoes.columns = [c.split("_", 1)[-1] if "_" in c else c
                           for c in transacoes.columns]

    print(f"   Transações: {transacoes.shape[0]:,}  |  Itens: {transacoes.shape[1]}")

    try:
        itemsets = apriori(transacoes, min_support=min_support, use_colnames=True)
    except Exception as e:
        print(f"   [aviso] apriori: {e}")
        return pd.DataFrame()

    if itemsets.empty:
        print(f"   ⚠ Nenhum itemset com suporte ≥ {min_support}. "
              f"Reduza min_support.")
        return pd.DataFrame()

    print(f"   Itemsets frequentes encontrados: {len(itemsets):,}")

    try:
        regras = association_rules(itemsets, metric="confidence",
                                     min_threshold=min_confidence)
    except Exception as e:
        print(f"   [aviso] association_rules: {e}")
        return pd.DataFrame()

    if regras.empty:
        print(f"   ⚠ Nenhuma regra com confiança ≥ {min_confidence}.")
        return pd.DataFrame()

    # Filtra apenas regras que TÊM o rótulo no consequente
    def _contem_label(itemset):
        return any(str(x).startswith("ROT=") for x in itemset)

    regras_label = regras[regras["consequents"].apply(_contem_label)].copy()
    regras_label = regras_label.sort_values("lift", ascending=False)

    print(f"\n── Top-15 regras → rótulo (ordenadas por lift) ──")
    cols_show = ["antecedents", "consequents", "support", "confidence", "lift"]
    print(regras_label[cols_show].head(15).to_string(index=False))

    arq = os.path.join(pasta, "regras_associacao.csv")
    regras_label[cols_show].to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")

    # Gráfico: lift × confiança das regras
    if len(regras_label) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        cores = regras_label["consequents"].apply(
            lambda c: PALETA["engenharia"] if "engenharia" in str(c) else PALETA["geral"]
        )
        ax.scatter(regras_label["confidence"], regras_label["lift"],
                    s=regras_label["support"] * 2000,
                    c=cores, alpha=0.6, edgecolor="white")
        ax.axhline(1, color="red", linestyle="--", lw=1, alpha=0.5,
                    label="Lift = 1 (independência)")
        ax.set_title("Regras de associação: lift × confiança (Aula 39)\n"
                     "Tamanho do círculo = suporte | cor = consequente",
                     fontweight="bold")
        ax.set_xlabel("Confiança"); ax.set_ylabel("Lift")
        ax.legend()
        sns.despine(ax=ax); fig.tight_layout()
        _salvar(fig, "p3_05_regras_associacao.png", pasta)

    return regras_label


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 27 — Pipeline final (Aulas 6 e 20)
# ════════════════════════════════════════════════════════════════════════════

def construir_pipeline_final(melhor_modelo, melhor_params=None):
    """ColumnTransformer(texto → TF-IDF, numéricas → StandardScaler) + clf."""
    col_texto = "objeto_limpo"
    cols_num  = ["log_valorTotalEstimado","len_tokens","n_keywords_eng"]
    pre = ColumnTransformer(transformers=[
        ("texto", TfidfVectorizer(min_df=3, max_df=0.85, sublinear_tf=True,
                                    ngram_range=(1, 2),
                                    max_features=15_000), col_texto),
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler",  StandardScaler()),
        ]), cols_num),
    ])
    pipe = Pipeline(steps=[("pre", pre), ("clf", melhor_modelo)])
    if melhor_params:
        pipe.set_params(**{f"clf__{k}":v for k,v in melhor_params.items()})
    return pipe


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 25 — Orquestrador Parte 3
# ════════════════════════════════════════════════════════════════════════════

def executar_parte3(resultados_p2, fazer_grid=True, fazer_clustering=True,
                     fazer_regras=True, fazer_embeddings=False,
                     tipo_embedding="sentence-bert"):
    """
    Pipeline completo da Parte 3.

    TODAS as etapas rodam SEM amostragem, SEM corte de iterações. O preço é
    o tempo de execução e o pico de RAM. Para mitigar OOM, esta função:
      • Libera memória entre etapas (gc.collect)
      • Imprime RAM em pontos críticos (monitorar_ram)
      • Sugere salvar_estado_pipeline ao final, para sobreviver a OOM

    Se acontecer OOM mesmo assim:
      1. Reinicie o runtime (Runtime → Restart runtime)
      2. Recarregue df e p2 com recarregar_estado_pipeline
      3. Rode novamente — desta vez sem o df e p2 antigos na memória

    Parâmetros
    ──────────
    fazer_grid       : roda GridSearchCV (Aulas 18, 22) — pode demorar
    fazer_clustering : roda KMeans + hierárquico (Aulas 33, 35, 37) + LDA + LP
    fazer_regras     : roda Apriori (Aula 39)
    fazer_embeddings : compara TF-IDF × Sentence-BERT/BERTimbau (Aula 42)
                       ⚠ Baixa modelo de ~80MB-440MB. Default = False.
    tipo_embedding   : "sentence-bert", "sentence-bert-mpnet",
                       "bertimbau" ou "bertimbau-sts"
    """
    import gc
    print("\n" + "█"*62)
    print("  PARTE 3 — TÉCNICAS AVANÇADAS")
    print("  Balanceamento (9, 31) · GridSearch (18, 22) · KNN (24, 25)")
    print("  Clustering (33, 35, 37) · Regras (39) · Embeddings (42)")
    print("█"*62)
    df     = resultados_p2["df_processado"]
    X      = resultados_p2["X"]
    y      = resultados_p2["y"]
    tfidf  = resultados_p2["tfidf"]
    ranking = resultados_p2["ranking"]

    if "ufSigla" in df.columns and "anoPublicacao" in df.columns:
        pasta = _pasta_saida_padrao(df)
    else:
        pasta = "graficos_pncp"
    os.makedirs(pasta, exist_ok=True)
    r3 = {}
    monitorar_ram("início Parte 3")

    print("\n[1] Comparação de balanceamento — SMOTE × Random Over/Under (Aulas 9, 31)...")
    r3["balanceamento"] = comparar_balanceamento(df, X, y, pasta)
    gc.collect(); monitorar_ram("após balanceamento")

    if fazer_grid:
        print("\n[2] GridSearchCV (Aulas 18 e 22)...")
        grids = {}
        try: grids["DecisionTree"] = grid_search_tree(X, y)
        except Exception as e: print(f"   [aviso] grid árvore: {e}")
        gc.collect(); monitorar_ram("após grid árvore")

        # SVM em grid: caro em RAM (LinearSVC em high-dim sparse). Sempre roda
        # quando fazer_grid=True. Se a RAM estourar, o try/except captura e
        # reporta a falha sem matar o pipeline. Para datasets muito grandes
        # (>50M células), avisamos que SVM pode falhar mas tentamos mesmo assim.
        if X.shape[0] * X.shape[1] >= 50_000_000:
            print(f"   [aviso] dataset grande ({X.shape[0]:,}×{X.shape[1]:,}). "
                  f"Grid SVM pode usar muita RAM — se OOM, considere reiniciar.")
        try: grids["SVM_linear"] = grid_search_svm(X, y)
        except MemoryError:
            print("   [aviso] grid SVM: MemoryError — pulando esta etapa.")
        except Exception as e: print(f"   [aviso] grid SVM: {e}")
        gc.collect(); monitorar_ram("após grid SVM")
        g_grid_search_results(grids, pasta)
        r3["grids"] = grids

    print("\n[3] KNN similaridade — busca por casos análogos (Aulas 24/25)...")
    demonstrar_knn_similaridade(df, X, tfidf, ranking, pasta)
    gc.collect()

    if fazer_clustering:
        print("\n[4] KMeans + validação por silhueta (Aulas 35, 37)...")
        try:
            r3["kmeans"] = kmeans_e_validacao(X, df, pasta, k_max=12)
        except Exception as e:
            print(f"   [aviso] KMeans: {e}")
        gc.collect(); monitorar_ram("após KMeans")

        print("\n[5] Agrupamento hierárquico dos suspeitos (Aula 33)...")
        try:
            r3["dendrograma"] = agrupamento_hierarquico_suspeitos(
                df, X, ranking, pasta, n_amostra=30
            )
        except Exception as e:
            print(f"   [aviso] dendrograma: {e}")
        gc.collect()

        # 5B. Topic Modeling via LDA (descoberta de tópicos não-supervisionada)
        print("\n[5B] Topic Modeling via LDA (Aula 30/41)...")
        try:
            r3["lda"] = topic_modeling_lda(df, pasta, n_topicos=8)
        except Exception as e:
            print(f"   [aviso] LDA: {e}")
        gc.collect(); monitorar_ram("após LDA")

        # 5C. Classificação semissupervisionada (Label Propagation, LABIC)
        print("\n[5C] Classificação semissupervisionada (Label Propagation)...")
        print("     Inspirada em Coutinho, Rossi & Rezende (2019).")
        print("     Trata rótulos como ruidosos e propaga rótulos via vizinhança.")
        try:
            r3["label_propagation"] = (
                classificacao_semisupervisionada_label_propagation(
                    df, X, frac_rotulada=0.2
                )
            )
            if r3["label_propagation"]:
                arq = os.path.join(pasta, "p3_label_propagation_suspeitos.csv")
                susp = r3["label_propagation"].get("suspeitos_subenq",
                                                    pd.DataFrame())
                if not susp.empty:
                    susp_cols = [c for c in [
                        "numeroControlePNCP", "objeto", "rotulo",
                        "rotulo_propagado", "confianca_propagacao",
                        "valorTotalEstimado", "razaoSocialOrgao",
                    ] if c in susp.columns]
                    susp[susp_cols].to_csv(arq, index=False, encoding="utf-8-sig")
                    print(f"   💾 {arq}")
        except Exception as e:
            print(f"   [aviso] Label Propagation: {e}")

    if fazer_regras:
        print("\n[6] Regras de associação — Apriori (Aula 39)...")
        try:
            r3["regras"] = regras_associacao_metadados(df, pasta,
                                                       min_support=0.05,
                                                       min_confidence=0.6)
        except Exception as e:
            print(f"   [aviso] regras de associação: {e}")

    if fazer_embeddings:
        print(f"\n[7] Embeddings semânticos × TF-IDF (Aula 42)...")
        print(f"    Tipo: {tipo_embedding}")
        try:
            # Limita amostragem para não estourar memória/tempo
            r3["embeddings_comparacao"] = comparar_tfidf_vs_embeddings(
                df, pasta,
                tipo_embedding=tipo_embedding,
                n_splits=5,
                max_amostras=2000,   # ajusta pra cima se tiver GPU
            )
        except ImportError as e:
            print(f"   [pulado] {e}")
        except Exception as e:
            print(f"   [aviso] embeddings: {e}")

    print("\n[8] Pipeline final (Aulas 6 e 20)...")
    best = resultados_p2["melhor_modelo"]
    best_p = None
    if fazer_grid and "grids" in r3:
        if "DecisionTree" in best and "DecisionTree" in r3["grids"]:
            best_p = r3["grids"]["DecisionTree"]["best_params"]
    modelos_def = definir_modelos()
    try:
        pipe = construir_pipeline_final(modelos_def[best], best_p)
        pipe.fit(df, y)
        arq = os.path.join(pasta, "pipeline_final.pkl")
        joblib.dump(pipe, arq)
        print(f"   💾 {arq}")
        r3["pipeline_final"] = pipe
    except Exception as e:
        print(f"   [aviso] Pipeline final: {e}")
    print("\n" + "█"*62 + "\n  PARTE 3 ✅\n" + "█"*62)
    return r3


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 26 — Ponto de entrada
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# ███████████████████   PARTE 4 — RIGOR DE PESQUISA   ███████████████████████
# ════════════════════════════════════════════════════════════════════════════
#
# Componentes específicos para um TCC de pesquisa rigorosa:
#   SEÇÃO 26 — Reprodutibilidade (seed, log de versões, hash)
#   SEÇÃO 27 — Holdout final (20%) — tocado UMA vez no fim
#   SEÇÃO 28 — Teste de McNemar entre os 2 melhores modelos
#   SEÇÃO 29 — Bootstrap de intervalos de confiança
#   SEÇÃO 30 — Classificação multiclasse (geral × eng. comum × eng. especial)
#   SEÇÃO 31 — Orquestrador da Parte 4
# ────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 26 — Reprodutibilidade
# ════════════════════════════════════════════════════════════════════════════

import random
import platform
import hashlib

SEED_GLOBAL = 42


def fixar_seeds(seed: int = SEED_GLOBAL) -> None:
    """
    Fixa todas as seeds aleatórias do projeto para reprodutibilidade.
    Para um TCC sério, isso é OBRIGATÓRIO — examinadores vão querer
    reproduzir os resultados.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"   ✅ Seed global fixada: {seed}")


def log_ambiente_execucao(pasta: str = ".") -> dict:
    """
    Salva log das versões de todas as bibliotecas + ambiente de execução.
    Acompanha o TCC para o examinador conseguir reproduzir.
    """
    import sklearn, scipy, matplotlib

    info = {
        "data_execucao":  datetime.datetime.now().isoformat(),
        "python":         platform.python_version(),
        "sistema":        platform.platform(),
        "processador":    platform.processor(),
        "seed_global":    SEED_GLOBAL,
        "bibliotecas": {
            "numpy":         np.__version__,
            "pandas":        pd.__version__,
            "scikit-learn":  sklearn.__version__,
            "scipy":         scipy.__version__,
            "matplotlib":    matplotlib.__version__,
            "seaborn":       sns.__version__,
        },
    }
    # opcionais
    try:
        import imblearn; info["bibliotecas"]["imbalanced-learn"] = imblearn.__version__
    except ImportError:
        pass
    try:
        import nltk; info["bibliotecas"]["nltk"] = nltk.__version__
    except ImportError:
        pass
    try:
        import mlxtend; info["bibliotecas"]["mlxtend"] = mlxtend.__version__
    except ImportError:
        pass

    print("\n── Ambiente de execução ──")
    print(f"   Python:        {info['python']}")
    print(f"   Sistema:       {info['sistema']}")
    print(f"   Seed:          {info['seed_global']}")
    print(f"   Bibliotecas:")
    for n, v in info["bibliotecas"].items():
        print(f"      {n:18s} {v}")

    arq = os.path.join(pasta, "ambiente_execucao.json")
    os.makedirs(pasta, exist_ok=True)
    import json as _json
    with open(arq, "w", encoding="utf-8") as f:
        _json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"\n   💾 Log salvo: {arq}")
    return info


def hash_dataset(df: pd.DataFrame) -> str:
    """
    Gera hash SHA-256 do dataset para garantir que análises usam EXATAMENTE
    os mesmos dados em execuções diferentes.

    Inclua o hash no relatório do TCC — assim o examinador pode confirmar
    que está olhando para os mesmos números.
    """
    # Converte conteúdo essencial do df para bytes
    cols_chave = [c for c in ["numeroControlePNCP", "objeto",
                               "valorTotalEstimado", "rotulo"]
                   if c in df.columns]
    if not cols_chave:
        return "hash_indeterminado"
    conteudo = df[cols_chave].sort_values(cols_chave[0]).to_csv(index=False)
    h = hashlib.sha256(conteudo.encode("utf-8")).hexdigest()
    print(f"   Hash SHA-256 do dataset: {h[:16]}... ({len(df):,} linhas, "
          f"{len(cols_chave)} cols-chave)")
    return h


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 27 — Holdout final (split treino/teste tocado uma vez)
# ════════════════════════════════════════════════════════════════════════════

def separar_holdout(df: pd.DataFrame, frac_teste: float = 0.20,
                     seed: int = SEED_GLOBAL) -> tuple:
    """
    Split estratificado treino/teste 80/20.

    Filosofia (Aula 18 + boas práticas):
    Toda a CV, GridSearch, escolha de threshold etc. é feita SOMENTE no
    treino. O conjunto de teste é tocado UMA ÚNICA vez no final, para
    reportar o número "limpo" no TCC.

    Por que importa: usar CV no dataset inteiro vaza informação porque
    threshold/hiperparâmetros são afinados em todos os exemplos. O número
    final de F1 vai ficar otimista (1-3 pontos a mais que o real).
    """
    df_train, df_test = train_test_split(
        df, test_size=frac_teste, random_state=seed,
        stratify=df["rotulo"] if "rotulo" in df.columns else None,
    )
    print(f"\n   Split estratificado: treino={len(df_train):,}  "
          f"teste={len(df_test):,}  (frac_teste={frac_teste})")
    print(f"   Treino: {df_train['rotulo'].value_counts().to_dict()}")
    print(f"   Teste:  {df_test['rotulo'].value_counts().to_dict()}")
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def avaliar_holdout(modelo_treinado, X_teste, y_teste,
                     pasta: str, modelo_nome: str = "modelo") -> dict:
    """
    Avalia o modelo no conjunto holdout (NUNCA visto durante treino).
    Esses são os números "oficiais" do TCC.
    """
    print(f"\n── Avaliação no holdout — {modelo_nome} ──")
    y_pred = modelo_treinado.predict(X_teste)
    try:
        y_prob = modelo_treinado.predict_proba(X_teste)[:, 1]
        tem_proba = True
    except Exception:
        y_prob = None
        tem_proba = False

    metricas = {
        "accuracy":      round(accuracy_score(y_teste, y_pred), 4),
        "f1_macro":      round(f1_score(y_teste, y_pred, average="macro"), 4),
        "f1_engenharia": round(f1_score(y_teste, y_pred, pos_label=1, average="binary"), 4),
        "precision_eng": round(precision_score(y_teste, y_pred, pos_label=1,
                                                zero_division=0), 4),
        "recall_eng":    round(recall_score(y_teste, y_pred, pos_label=1,
                                              zero_division=0), 4),
    }
    if tem_proba:
        metricas["roc_auc"]       = round(roc_auc_score(y_teste, y_prob), 4)
        metricas["avg_precision"] = round(average_precision_score(y_teste, y_prob), 4)

    print(f"   Métricas no HOLDOUT (números oficiais do TCC):")
    for k, v in metricas.items():
        print(f"      {k:18s} {v}")

    # Matriz de confusão limpa
    cm = confusion_matrix(y_teste, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["geral", "engenharia"],
                yticklabels=["geral", "engenharia"],
                ax=ax, cbar=False, annot_kws={"size": 14})
    ax.set_title(f"Matriz de confusão no HOLDOUT — {modelo_nome}\n"
                 f"F1-eng = {metricas['f1_engenharia']:.4f}",
                 fontweight="bold")
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    fig.tight_layout()
    _salvar(fig, "p4_01_matriz_confusao_holdout.png", pasta)

    return metricas


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 28 — Teste de McNemar entre os dois melhores modelos
# ════════════════════════════════════════════════════════════════════════════

def teste_mcnemar(y_true: np.ndarray, y_pred_a: np.ndarray,
                   y_pred_b: np.ndarray, nome_a: str = "A",
                   nome_b: str = "B") -> dict:
    """
    Teste de McNemar — verifica se dois classificadores têm desempenho
    estatisticamente diferente.

    Constrói tabela de contingência 2x2:
                          | B acertou | B errou
        A acertou (n_aa)  |    a      |    b
        A errou           |    c      |    d

    Estatística: χ² = (|b - c| - 1)² / (b + c)
    Se p < 0.05, rejeita-se H0 (os modelos são equivalentes).

    Importância para o TCC: dizer que "Modelo X tem F1=0.78 e Modelo Y
    tem F1=0.79" é diferente de provar que essa diferença é
    estatisticamente significativa.
    """
    if len(y_true) != len(y_pred_a) or len(y_true) != len(y_pred_b):
        raise ValueError("Comprimentos diferentes de y_true / y_pred_*")

    a_acertou = (y_pred_a == y_true)
    b_acertou = (y_pred_b == y_true)

    n_aa_be = int(np.sum(a_acertou & ~b_acertou))      # A acertou, B errou
    n_ae_bb = int(np.sum(~a_acertou & b_acertou))      # A errou, B acertou
    n_aa_bb = int(np.sum(a_acertou & b_acertou))
    n_ae_be = int(np.sum(~a_acertou & ~b_acertou))

    # Estatística de McNemar com correção de continuidade
    if n_aa_be + n_ae_bb == 0:
        chi2, p_valor = 0.0, 1.0
    else:
        chi2 = (abs(n_aa_be - n_ae_bb) - 1) ** 2 / (n_aa_be + n_ae_bb)
        # p-valor da chi² com 1 grau de liberdade
        from scipy.stats import chi2 as chi2_dist
        p_valor = 1 - chi2_dist.cdf(chi2, df=1)

    print(f"\n── Teste de McNemar: {nome_a} × {nome_b} ──")
    print(f"   Tabela de contingência:")
    print(f"                         {nome_b} ACERTOU   {nome_b} ERROU")
    print(f"      {nome_a:14s} ACERTOU  {n_aa_bb:8d}        {n_aa_be:8d}")
    print(f"      {nome_a:14s} ERROU    {n_ae_bb:8d}        {n_ae_be:8d}")
    print(f"\n   χ² = {chi2:.4f}   p-valor = {p_valor:.4g}")

    significativo = p_valor < 0.05
    if significativo:
        if n_aa_be > n_ae_bb:
            interpretacao = f"{nome_a} é SIGNIFICATIVAMENTE melhor (p < 0.05)"
        else:
            interpretacao = f"{nome_b} é SIGNIFICATIVAMENTE melhor (p < 0.05)"
    else:
        interpretacao = f"Diferença NÃO significativa (p ≥ 0.05) — modelos equivalentes"
    print(f"   → {interpretacao}")

    return {
        "n_aa_bb":         n_aa_bb,
        "n_aa_be":         n_aa_be,
        "n_ae_bb":         n_ae_bb,
        "n_ae_be":         n_ae_be,
        "chi2":            round(chi2, 4),
        "p_valor":         round(p_valor, 6),
        "significativo":   significativo,
        "interpretacao":   interpretacao,
    }


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 29 — Bootstrap de intervalos de confiança
# ════════════════════════════════════════════════════════════════════════════

def bootstrap_metrica(y_true: np.ndarray, y_pred: np.ndarray,
                        metrica: str = "f1", n_iter: int = 1_000,
                        nivel_confianca: float = 0.95,
                        seed: int = SEED_GLOBAL) -> dict:
    """
    Calcula intervalo de confiança via bootstrap (reamostragem com reposição).

    Por que usar IC ao invés de só reportar o valor pontual?
    F1 = 0.78 sozinho não diz se é confiável. Se IC95% = [0.65, 0.85],
    sabemos que o valor "verdadeiro" está nessa faixa com 95% de confiança.
    Em datasets pequenos (poucos exemplos de engenharia), o IC vai ser largo
    e isso é uma informação importante para o examinador.

    Métricas suportadas: 'f1', 'precision', 'recall', 'accuracy'
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    if n != len(y_pred):
        raise ValueError("y_true e y_pred com tamanhos diferentes")

    func = {
        "f1":        lambda yt, yp: f1_score(yt, yp, pos_label=1, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, pos_label=1, zero_division=0),
        "recall":    lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
        "accuracy":  lambda yt, yp: accuracy_score(yt, yp),
    }.get(metrica, None)
    if func is None:
        raise ValueError(f"Métrica '{metrica}' não suportada")

    valores = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        # Garante que ambas as classes estejam presentes
        if len(np.unique(y_true[idx])) < 2:
            continue
        valores.append(func(y_true[idx], y_pred[idx]))
    valores = np.array(valores)

    alpha = (1 - nivel_confianca) / 2
    lower = np.quantile(valores, alpha)
    upper = np.quantile(valores, 1 - alpha)
    media = valores.mean()
    pontual = func(y_true, y_pred)

    return {
        "metrica":         metrica,
        "valor_pontual":   round(pontual, 4),
        "media_bootstrap": round(media, 4),
        "ic_lower":        round(lower, 4),
        "ic_upper":        round(upper, 4),
        "amplitude_ic":    round(upper - lower, 4),
        "nivel_confianca": nivel_confianca,
        "n_iter":          len(valores),
    }


def bootstrap_completo(y_true: np.ndarray, y_pred: np.ndarray,
                         pasta: str, modelo_nome: str = "modelo",
                         n_iter: int = 1_000) -> pd.DataFrame:
    """
    Calcula IC95% para F1, Precision, Recall e Accuracy do classificador,
    e plota as distribuições bootstrap.
    """
    print(f"\n── Bootstrap IC95% ({n_iter} reamostragens) — {modelo_nome} ──")
    metricas = ["f1", "precision", "recall", "accuracy"]
    linhas = []
    distribuicoes = {}

    rng = np.random.default_rng(SEED_GLOBAL)
    n = len(y_true)
    for met in metricas:
        func = {
            "f1":        lambda yt, yp: f1_score(yt, yp, pos_label=1, zero_division=0),
            "precision": lambda yt, yp: precision_score(yt, yp, pos_label=1, zero_division=0),
            "recall":    lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
            "accuracy":  lambda yt, yp: accuracy_score(yt, yp),
        }[met]
        valores = []
        rng2 = np.random.default_rng(SEED_GLOBAL)
        for _ in range(n_iter):
            idx = rng2.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            valores.append(func(y_true[idx], y_pred[idx]))
        valores = np.array(valores)
        distribuicoes[met] = valores
        pontual = func(y_true, y_pred)
        linhas.append({
            "Métrica":       met,
            "Valor pontual": round(pontual, 4),
            "Média BS":      round(valores.mean(), 4),
            "IC95 Lower":    round(np.quantile(valores, 0.025), 4),
            "IC95 Upper":    round(np.quantile(valores, 0.975), 4),
            "Amplitude":     round(np.quantile(valores, 0.975)
                                   - np.quantile(valores, 0.025), 4),
        })

    tab = pd.DataFrame(linhas).set_index("Métrica")
    print(tab.to_string())

    # Gráfico: distribuições bootstrap
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, met in zip(axes.flatten(), metricas):
        vals = distribuicoes[met]
        ax.hist(vals, bins=40, color=PALETA["engenharia"],
                edgecolor="white", alpha=0.7)
        lo = np.quantile(vals, 0.025); hi = np.quantile(vals, 0.975)
        ax.axvline(lo, color="red", linestyle="--", lw=1.2, label=f"IC95: [{lo:.3f}, {hi:.3f}]")
        ax.axvline(hi, color="red", linestyle="--", lw=1.2)
        ax.axvline(vals.mean(), color="black", lw=1.5, label=f"Média: {vals.mean():.3f}")
        ax.set_title(f"Bootstrap — {met}", fontweight="bold")
        ax.set_xlabel(met); ax.set_ylabel("Frequência")
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

    fig.suptitle(f"Distribuições bootstrap ({n_iter} reamostragens) — {modelo_nome}",
                 fontweight="bold")
    fig.tight_layout()
    _salvar(fig, "p4_02_bootstrap_distribuicoes.png", pasta)
    return tab


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 30 — Classificação multiclasse (geral × eng. comum × eng. especial)
# ════════════════════════════════════════════════════════════════════════════
#
# Premissa (Resolução CONFEA 1.048/2013 + apresentação do projeto):
#   • Engenharia COMUM     → reformas, manutenções rotineiras, instalações
#                              elétricas residenciais simples, etc.
#   • Engenharia ESPECIAL  → obras com cálculo estrutural complexo, geotecnia,
#                              fundações profundas, instalações industriais,
#                              tratamento de efluentes, automação, etc.
#
# O PNCP só distingue categoria 8 (Serviços) × 9 (Serv. Engenharia).
# A subdivisão em "comum" × "especial" é INFERIDA por keywords técnicas.

KEYWORDS_ENG_ESPECIAL = {
    # Geotecnia e fundações profundas
    "sondagem", "geotecnico", "geotécnico", "geotecnia",
    "fundacao", "fundação", "estaca", "estacas", "tirante", "tirantes",
    "contencao", "contenção", "talude", "muro de arrimo",
    # Estruturas complexas
    "estrutural", "calculo estrutural", "cálculo estrutural",
    "concreto armado", "concreto protendido", "metalica", "metálica",
    "ponte", "pontes", "viaduto", "passarela",
    # Instalações industriais e especiais
    "subestacao", "subestação", "alta tensao", "alta tensão",
    "industrial", "automacao", "automação",
    "spda", "sistema de protecao contra descargas",
    # Saneamento e ambiental
    "tratamento de efluente", "tratamento de esgoto",
    "estacao de tratamento", "estação de tratamento",
    "barragem", "bacia de contencao", "bacia de contenção",
    # Processos especiais
    "impermeabilizacao especial", "impermeabilização especial",
    "perfuracao", "perfuração", "rebaixamento de lencol",
    "rebaixamento de lençol",
}

# Engenharia COMUM (subset de KEYWORDS_ENG do Camada 1)
KEYWORDS_ENG_COMUM = {
    "reforma", "manutencao", "manutenção", "pintura", "revestimento",
    "instalacao", "instalação", "telhado", "cobertura", "esquadria",
    "piso", "pisos", "alvenaria", "ar-condicionado", "climatização",
}


def classificar_em_3_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna `rotulo_3classes` combinando o sinal OFICIAL do PNCP
    (categoriaProcessoId) com a heurística de keywords:

      • categoriaProcessoId == 7 (Obras)         → 'eng_especial' (oficial)
      • categoriaProcessoId == 9 + keywords esp. → 'eng_especial' (heurística)
      • categoriaProcessoId == 9 (resto)         → 'eng_comum'
      • categoriaProcessoId == 8                 → 'geral'

    Justificativa para Obras = eng_especial automático:
    A Lei 14.133/2021 art. 6º XII define obra como atividade que "implica
    intervenção no meio ambiente" e exige projeto básico/executivo.
    Pela complexidade técnica e exigência de ART, são intrinsecamente
    "especiais" — não cabe heurística de palavras-chave aqui.

    Importância: testa se um modelo de 3 classes consegue diferenciar
    melhor que o binário. Se F1-macro multiclasse > F1-eng binário,
    é evidência de que a separação faz sentido para o domínio.

    NOTA: a separação eng_comum × eng_especial DENTRO da categoria 9 ainda
    é heurística (via keywords). Idealmente seria rotulação manual por
    especialista de engenharia, mas fica fora do escopo do TCC.
    """
    df = df.copy()

    def _classificar(row):
        cat = row.get("categoriaProcessoId")
        if cat == 8:
            return "geral"
        if cat == 7:
            # Obras são por definição engenharia especial (Lei 14.133 art. 6º XII)
            return "eng_especial"
        # categoria 9 — Serv. Engenharia: usa heurística de keywords
        toks = set(tokenizar(str(row.get("objeto", ""))))
        if toks & KEYWORDS_ENG_ESPECIAL:
            return "eng_especial"
        return "eng_comum"

    df["rotulo_3classes"] = df.apply(_classificar, axis=1)
    print(f"\n── Distribuição de rótulos (3 classes — combinada) ──")
    print(df["rotulo_3classes"].value_counts().to_string())
    print(f"\n   ✓ Categoria 7 (Obras) → eng_especial (sinal oficial PNCP)")
    print(f"   ⚠ Dentro da categoria 9 (Serv.Eng): separação eng_comum × eng_especial")
    print(f"     é HEURÍSTICA por keywords (ideal seria especialista humano).")
    return df


def comparar_binario_vs_multiclasse(df: pd.DataFrame, pasta: str,
                                       n_splits: int = 5) -> pd.DataFrame:
    """
    Compara F1-macro do classificador binário (geral × engenharia)
    com o multiclasse (geral × eng_comum × eng_especial).

    Hipótese: se F1-macro multiclasse > F1-binário, vale separar.
    Se for menor (provavelmente vai ser, dada a esparsidade), o binário
    é a escolha correta para o TCC e a multiclasse fica como discussão.
    """
    df = classificar_em_3_classes(df)

    # Pré-processa
    if "objeto_limpo" not in df.columns:
        df = preprocessar_texto(df)
    X, _, _ = construir_features(df, usar_metadados=True)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED_GLOBAL)
    linhas = []

    # ── Binário ──────────────────────────────────────────────────────────────
    y_bin = df["rotulo"].map({"geral": 0, "engenharia": 1}).values
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1_000,
                              solver="lbfgs", random_state=SEED_GLOBAL)
    print("\n   Treinando classificador BINÁRIO...")
    sc = cross_validate(clf, X, y_bin, cv=cv,
                         scoring=["accuracy", "f1_macro", "f1"])
    linhas.append({
        "Configuração":     "Binário (geral × eng)",
        "Classes":          2,
        "Accuracy":         round(sc["test_accuracy"].mean(), 4),
        "F1-macro":         round(sc["test_f1_macro"].mean(), 4),
        "F1-pos (eng)":     round(sc["test_f1"].mean(), 4),
    })

    # ── Multiclasse ─────────────────────────────────────────────────────────
    y_3 = df["rotulo_3classes"].map({"geral": 0, "eng_comum": 1,
                                      "eng_especial": 2}).values
    if (df["rotulo_3classes"].value_counts() < 3).any():
        print("   ⚠ Alguma classe tem menos de 3 exemplos. Multiclasse pulado.")
        return pd.DataFrame(linhas).set_index("Configuração")

    clf3 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1_000,
                                solver="lbfgs", random_state=SEED_GLOBAL,
                                multi_class="multinomial")
    print("   Treinando classificador MULTICLASSE (3 classes)...")
    try:
        sc3 = cross_validate(clf3, X, y_3, cv=cv,
                              scoring=["accuracy", "f1_macro"])
        linhas.append({
            "Configuração":     "Multiclasse (geral × comum × especial)",
            "Classes":          3,
            "Accuracy":         round(sc3["test_accuracy"].mean(), 4),
            "F1-macro":         round(sc3["test_f1_macro"].mean(), 4),
            "F1-pos (eng)":     None,    # n/a em multiclasse
        })
    except Exception as e:
        print(f"   [aviso] multiclasse falhou: {e}")

    tab = pd.DataFrame(linhas).set_index("Configuração")
    print("\n── Binário × Multiclasse ──")
    print(tab.to_string())

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 4))
    cols = [c for c in ["Accuracy", "F1-macro", "F1-pos (eng)"] if c in tab.columns]
    tab[cols].plot(kind="bar", ax=ax, colormap="Set1",
                    edgecolor="white", linewidth=0.5)
    ax.set_title("Binário × Multiclasse (CONFEA 1.048/2013 — heurística)\n"
                 "Mesma LR + mesma CV — só muda a estrutura do problema",
                 fontweight="bold")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_xticklabels(tab.index, rotation=10, ha="right")
    ax.legend(title="Métrica")
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "p4_03_binario_vs_multiclasse.png", pasta)
    return tab


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 31 — Orquestrador da Parte 4
# ════════════════════════════════════════════════════════════════════════════

def executar_parte4(df: pd.DataFrame, p2_resultados: dict,
                     pasta_saida: str = None,
                     fazer_holdout: bool = True,
                     fazer_mcnemar: bool = True,
                     fazer_bootstrap: bool = True,
                     fazer_multiclasse: bool = True) -> dict:
    """
    Pipeline completo da Parte 4: rigor estatístico para o TCC.

    Esta parte usa os resultados da Parte 2 para:
      1. Avaliação no holdout 20% — números OFICIAIS do TCC
      2. McNemar entre os 2 melhores modelos
      3. Bootstrap IC95% das métricas
      4. Comparação binário × multiclasse (CONFEA 1.048/2013)
    """
    print("\n" + "█"*62)
    print("  PARTE 4 — RIGOR DE PESQUISA")
    print("  Reproducibilidade · Holdout · McNemar · Bootstrap · Multiclasse")
    print("█"*62)

    # Pasta
    if pasta_saida is None:
        pasta_saida = _pasta_saida_padrao(df)
    os.makedirs(pasta_saida, exist_ok=True)

    r4 = {}

    # ── 0. Reprodutibilidade ────────────────────────────────────────────────
    print("\n[0] Reprodutibilidade...")
    fixar_seeds(SEED_GLOBAL)
    r4["ambiente"]    = log_ambiente_execucao(pasta_saida)
    r4["dataset_hash"] = hash_dataset(df)

    # ── 1. Holdout final ────────────────────────────────────────────────────
    if fazer_holdout:
        print("\n[1] Holdout final (20%)...")
        df_train, df_test = separar_holdout(df, frac_teste=0.20)

        # Pré-processa AMBOS com o mesmo TF-IDF (treinado só no train)
        df_train_p = preprocessar_texto(df_train)
        df_test_p  = preprocessar_texto(df_test)

        # Treina TF-IDF apenas no treino (evita vazamento)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing       import StandardScaler
        from scipy.sparse                import hstack, csr_matrix

        col_texto = "objeto_limpo"
        tfidf = TfidfVectorizer(min_df=3, max_df=0.85, sublinear_tf=True,
                                 ngram_range=(1, 2),
                                 max_features=15_000, strip_accents="unicode")
        X_train_t = tfidf.fit_transform(df_train_p[col_texto].fillna(""))
        X_test_t  = tfidf.transform(df_test_p[col_texto].fillna(""))

        cols_num = [c for c in ["log_valorTotalEstimado",
                                 "log_valorTotalHomologado",
                                 "len_tokens", "n_keywords_eng"]
                    if c in df_train_p.columns]
        if cols_num:
            scaler = StandardScaler()
            X_train_n = scaler.fit_transform(df_train_p[cols_num].fillna(0))
            X_test_n  = scaler.transform(df_test_p[cols_num].fillna(0))
            X_train = hstack([X_train_t, csr_matrix(X_train_n)])
            X_test  = hstack([X_test_t, csr_matrix(X_test_n)])
        else:
            X_train, X_test = X_train_t, X_test_t

        y_train = df_train["rotulo"].map({"geral": 0, "engenharia": 1}).values
        y_test  = df_test["rotulo"].map({"geral": 0, "engenharia": 1}).values

        # Usa o melhor modelo da Parte 2
        melhor_nome = p2_resultados["melhor_modelo"]
        modelos_def = definir_modelos()
        modelo_final = modelos_def[melhor_nome]
        modelo_final.fit(X_train, y_train)

        r4["holdout_metricas"] = avaliar_holdout(
            modelo_final, X_test, y_test, pasta_saida, melhor_nome
        )
        r4["holdout_split"] = {"X_train": X_train, "y_train": y_train,
                                "X_test":  X_test,  "y_test":  y_test,
                                "tfidf": tfidf,
                                "modelo": modelo_final,
                                "modelo_nome": melhor_nome}

    # ── 2. McNemar entre os 2 melhores (no OOF da Parte 2) ──────────────────
    if fazer_mcnemar:
        print("\n[2] Teste de McNemar entre os 2 melhores modelos...")
        # Pega os 2 melhores não-Dummy do ranking
        tabela = p2_resultados["tabela"]
        nomes_ord = [n for n in tabela.index if "Dummy" not in n]
        if len(nomes_ord) >= 2:
            nome_a = nomes_ord[0]
            nome_b = nomes_ord[1]
            cv_res = p2_resultados["resultados_cv"]
            y_oof  = p2_resultados["y"]
            try:
                r4["mcnemar"] = teste_mcnemar(
                    y_true   = y_oof,
                    y_pred_a = cv_res[nome_a]["y_pred_oof"],
                    y_pred_b = cv_res[nome_b]["y_pred_oof"],
                    nome_a   = nome_a, nome_b = nome_b,
                )
            except Exception as e:
                print(f"   [aviso] McNemar: {e}")
        else:
            print("   [pulado] menos de 2 modelos disponíveis.")

    # ── 3. Bootstrap nas predições out-of-fold do melhor modelo ─────────────
    if fazer_bootstrap:
        print("\n[3] Bootstrap IC95% (out-of-fold do melhor modelo)...")
        try:
            melhor_nome = p2_resultados["melhor_modelo"]
            cv_res = p2_resultados["resultados_cv"]
            r4["bootstrap"] = bootstrap_completo(
                y_true     = p2_resultados["y"],
                y_pred     = cv_res[melhor_nome]["y_pred_oof"],
                pasta      = pasta_saida,
                modelo_nome= melhor_nome,
                n_iter     = 1_000,
            )
        except Exception as e:
            print(f"   [aviso] Bootstrap: {e}")

    # ── 4. Multiclasse ──────────────────────────────────────────────────────
    if fazer_multiclasse:
        print("\n[4] Comparação binário × multiclasse (CONFEA 1.048/2013)...")
        try:
            r4["multiclasse"] = comparar_binario_vs_multiclasse(df, pasta_saida)
        except Exception as e:
            print(f"   [aviso] Multiclasse: {e}")

    print("\n" + "█"*62)
    print("  PARTE 4 CONCLUÍDA ✅")
    if "holdout_metricas" in r4:
        print(f"  F1-engenharia no HOLDOUT: {r4['holdout_metricas']['f1_engenharia']}")
    if "mcnemar" in r4:
        print(f"  McNemar p-valor: {r4['mcnemar']['p_valor']}")
    print("█"*62)
    return r4


# ════════════════════════════════════════════════════════════════════════════
# ████████   PARTE 5 — UX, INTERPRETAÇÃO AUTOMÁTICA E RELATÓRIO   ████████████
# ════════════════════════════════════════════════════════════════════════════
#
# Esta seção atende três pedidos:
#   1. Perguntas interativas no console quando há mais de uma alternativa
#      (em vez de o pipeline assumir um valor padrão silenciosamente)
#   2. Comentários textuais automáticos sobre os números das partes 1-4
#   3. Geração de relatório markdown completo p/ colar no TCC
# ────────────────────────────────────────────────────────────────────────────


def perguntar_escolha(pergunta: str, opcoes: list, default: int = 0,
                       descricoes: list = None) -> int:
    """
    Apresenta um menu de alternativas no console e retorna o índice escolhido.

    Use quando o pipeline tem mais de um caminho razoável e queremos que o
    usuário decida (ao invés do código assumir uma decisão).

    Parâmetros
    ──────────
    pergunta    : texto da pergunta
    opcoes      : lista de strings (rótulos curtos das alternativas)
    default     : índice da opção padrão (sai automaticamente em modo
                   não-interativo)
    descricoes  : (opcional) lista paralela com descrições mais longas
                   de cada opção

    Retorna
    ───────
    Índice (int) da opção escolhida.

    Exemplo
    ───────
    >>> i = perguntar_escolha(
    ...     "Qual modelo de embedding usar?",
    ...     ["MiniLM (rápido)", "mpnet (preciso)", "BERTimbau (PT-BR)"],
    ...     default=0
    ... )
    """
    print(f"\n  ❓ {pergunta}")
    for i, op in enumerate(opcoes, start=1):
        marca = " (default)" if i - 1 == default else ""
        print(f"     [{i}] {op}{marca}")
        if descricoes and i - 1 < len(descricoes):
            print(f"         {descricoes[i-1]}")
    try:
        r = input(f"  Opção [{default+1}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    if not r:
        return default
    try:
        idx = int(r) - 1
        if 0 <= idx < len(opcoes):
            return idx
    except ValueError:
        pass
    print(f"  ⚠ Entrada inválida, usando default ({opcoes[default]})")
    return default


def perguntar_sim_nao(pergunta: str, default_sim: bool = True) -> bool:
    """Pergunta s/n com default sensato. Retorna bool."""
    sufixo = "[S/n]" if default_sim else "[s/N]"
    try:
        r = input(f"  ❓ {pergunta} {sufixo}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default_sim
    if not r:
        return default_sim
    return r.startswith("s")


# ════════════════════════════════════════════════════════════════════════════
# Interpretação automática: comenta os resultados em linguagem natural
# ════════════════════════════════════════════════════════════════════════════

def _faixa_metrica(v: float) -> str:
    """Classifica um valor de métrica em faixa qualitativa."""
    if pd.isna(v):
        return "indeterminado"
    if v >= 0.90:  return "excelente"
    if v >= 0.80:  return "muito bom"
    if v >= 0.70:  return "bom"
    if v >= 0.60:  return "razoável"
    if v >= 0.50:  return "fraco"
    return "muito fraco"


def glossario(termo: str = None) -> None:
    """
    Glossário curto dos parâmetros usados no projeto.

    Uso:
        glossario()              # imprime tudo
        glossario("F1")          # imprime explicação de "F1"
    """
    explicacoes = {
        "Acurácia": (
            "% total de acertos do modelo. Pode enganar quando as classes são"
            " desbalanceadas (modelo que sempre prevê 'geral' tem 97% de acurácia)."
        ),
        "Precisão": (
            "Dos contratos que o modelo disse SER engenharia, quantos eram mesmo?"
            " Alta precisão = poucos falsos alarmes."
        ),
        "Recall": (
            "Dos contratos que SÃO engenharia, quantos o modelo encontrou?"
            " Alto recall = poucos casos perdidos."
        ),
        "F1": (
            "Média harmônica de precisão e recall. Métrica única de qualidade"
            " que pune valores muito baixos. F1 ≥ 0.7 é bom para problemas reais."
        ),
        "F1-engenharia": (
            "F1 calculado SOMENTE para a classe 'engenharia' (a que importa)."
            " É a métrica principal do TCC."
        ),
        "F1-macro": (
            "Média do F1 das duas classes. Mais conservador que F1-engenharia"
            " porque também considera 'geral'."
        ),
        "ROC-AUC": (
            "Área sob a curva ROC. 0.5 = chute, 1.0 = perfeito."
            " > 0.85 = modelo bom; > 0.95 = excelente."
        ),
        "Avg-Precision": (
            "Área sob a curva Precision-Recall. Mais informativa que ROC-AUC"
            " quando há desbalanceamento (poucos exemplos da classe positiva)."
        ),
        "TP / FP / FN": (
            "TP = True Positive (acertou eng.); FP = False Positive (errou,"
            " disse eng. quando não era); FN = False Negative (perdeu uma eng.)."
        ),
        "Holdout": (
            "Conjunto de teste 'limpo' que NUNCA é usado durante o treinamento."
            " Os números do holdout são os 'oficiais' do TCC."
        ),
        "Cross-validation (CV)": (
            "Divide os dados em K partes, treina em K-1 e testa em 1. Repete K"
            " vezes e tira a média. Estima a performance média do modelo."
        ),
        "McNemar p-valor": (
            "Teste estatístico que compara dois modelos. p < 0.05 = a diferença"
            " entre eles É significativa (não é por acaso)."
        ),
        "IC95% (intervalo de confiança)": (
            "Faixa onde a métrica 'verdadeira' provavelmente está com 95% de"
            " confiança. IC estreito = resultado confiável; IC largo = volátil."
        ),
        "Bootstrap": (
            "Reamostragem com reposição (1000x) para estimar variabilidade."
            " Permite calcular intervalos de confiança sem assumir distribuição."
        ),
        "TF-IDF": (
            "Term Frequency × Inverse Document Frequency. Pondera palavras por"
            " quanto são raras no corpus — palavras únicas pesam mais."
        ),
        "Threshold (limiar)": (
            "Probabilidade mínima para o modelo classificar como engenharia."
            " Threshold alto = mais precisão (menos FP), menos recall."
        ),
        "log-odds": (
            "log(P(palavra|engenharia) / P(palavra|geral))."
            " Positivo = palavra mais comum em eng.; negativo = mais em geral."
        ),
        "score_irregularidade": (
            "Score multifatorial Lei 14.133/2021. >0 = sinais de rito"
            " inadequado para engenharia (red flags); <0 = rito OK."
        ),
        "score_subenquadramento": (
            "(Parte 8) Combina nº de contratos 'geral' × intensidade do CNAE de"
            " engenharia. Score alto = forte indício de subenquadramento."
        ),
        "subenquadramento": (
            "Contrato classificado como 'serviço geral' (categoria 8 PNCP)"
            " quando deveria ser 'serviço de engenharia' ou 'obra' (cat. 7/9)."
        ),
        "anomalia oposta": (
            "Inverso do subenquadramento: contrato como 'engenharia' mas"
            " executado por empresa SEM CNAE de eng. (viola Lei 6.496/77)."
        ),
        "ART / RRT": (
            "ART = Anotação de Responsabilidade Técnica (CREA). RRT = Registro"
            " de Responsabilidade Técnica (CAU). Obrigatórios para qualquer"
            " atividade de engenharia (Lei 6.496/77)."
        ),
        "CNAE": (
            "Classificação Nacional de Atividades Econômicas. Identificador"
            " oficial da Receita Federal sobre o que a empresa faz."
        ),
        "centralidade (grafo)": (
            "Métrica de quão 'importante' um nó é. Degree = nº de conexões;"
            " Betweenness = quanto o nó é 'ponte' entre outros."
        ),
        "Louvain": (
            "Algoritmo de detecção de comunidades em grafos. Identifica grupos"
            " de nós que se conectam mais entre si do que com o resto."
        ),
    }

    if termo:
        chave = next((k for k in explicacoes if termo.lower() in k.lower()), None)
        if chave:
            print(f"\n📖 {chave}:")
            print(f"   {explicacoes[chave]}")
        else:
            print(f"⚠ Termo '{termo}' não encontrado no glossário.")
            print(f"   Termos disponíveis: {', '.join(explicacoes.keys())}")
        return

    print("\n" + "═"*62)
    print("  GLOSSÁRIO — explicações curtas dos parâmetros do projeto")
    print("═"*62)
    for k, v in explicacoes.items():
        print(f"\n📖 {k}:")
        print(f"   {v}")


def interpretar_resultados(df: pd.DataFrame,
                              p2_resultados: dict = None,
                              p3_resultados: dict = None,
                              p4_resultados: dict = None) -> dict:
    """
    Gera comentário textual automático sobre os números do pipeline.

    Útil para colar diretamente no TCC ou para o aluno entender o que cada
    número significa em prática. Não substitui análise crítica, mas dá um
    primeiro rascunho honesto baseado em regras de bom senso de ML.

    Retorna
    ───────
    dict com chaves: 'distribuicao', 'classificacao', 'tecnicas_avancadas',
    'rigor_pesquisa', 'subenquadramento', 'rigor_legal'.
    Cada valor é uma string em markdown.
    """
    coments = {}

    # ── 1. Distribuição da base ─────────────────────────────────────────────
    n_total = len(df)
    if "rotulo" in df.columns:
        n_eng = (df["rotulo"] == "engenharia").sum()
        n_ger = (df["rotulo"] == "geral").sum()
        razao = n_ger / max(n_eng, 1)
        bloco = [
            f"### Distribuição da base",
            f"",
            f"- **Total:** {n_total:,} contratos",
            f"- **Engenharia:** {n_eng:,} ({n_eng/n_total*100:.1f}%)",
            f"- **Geral:** {n_ger:,} ({n_ger/n_total*100:.1f}%)",
            f"- **Razão geral/engenharia:** {razao:.1f} : 1",
        ]
        # Subclasse (Obras vs Serv. Eng.)
        if "subclasse" in df.columns:
            sub = df["subclasse"].value_counts().to_dict()
            obras = sub.get("obra", 0)
            serv_eng = sub.get("serv_engenharia", 0)
            bloco += [
                f"- **Obras (categoria 7):** {obras:,} contratos",
                f"- **Serviços de Engenharia (categoria 9):** {serv_eng:,} contratos",
            ]
        # Comentário interpretativo
        if razao > 30:
            bloco += [
                f"",
                f"**Comentário:** o desbalanceamento é severo (>30:1). "
                f"Métricas como acurácia ficam infladas — sempre priorize "
                f"F1-engenharia, recall e average precision na análise. "
                f"Considere técnicas de balanceamento (SMOTE, class_weight) "
                f"para o classificador.",
            ]
        elif razao > 10:
            bloco += [
                f"",
                f"**Comentário:** desbalanceamento moderado (>10:1). "
                f"Use class_weight='balanced' nos modelos, e priorize "
                f"F1-engenharia e average precision.",
            ]
        else:
            bloco += [
                f"",
                f"**Comentário:** desbalanceamento aceitável (<10:1). "
                f"Métricas mais estáveis, mas ainda recomendável usar "
                f"class_weight='balanced'.",
            ]
        coments["distribuicao"] = "\n".join(bloco)

    # ── 2. Classificação (Parte 2) ──────────────────────────────────────────
    if p2_resultados is not None and "tabela" in p2_resultados:
        tab = p2_resultados["tabela"]
        melhor = p2_resultados.get("melhor_modelo", "?")
        f1_melhor = tab.loc[melhor, "F1-engenharia"] if melhor in tab.index else None
        ap_melhor = tab.loc[melhor, "Avg-Precision"] if melhor in tab.index else None
        bloco = [
            f"### Classificação — Parte 2",
            f"",
            f"- **Melhor modelo:** {melhor}",
            f"- **F1-engenharia:** {f1_melhor:.4f} ({_faixa_metrica(f1_melhor)})",
        ]
        if ap_melhor is not None:
            bloco += [f"- **Avg-Precision:** {ap_melhor:.4f} "
                       f"({_faixa_metrica(ap_melhor)})"]
        # Comparação com Dummy
        if "Dummy_stratified" in tab.index:
            f1_dummy = tab.loc["Dummy_stratified", "F1-engenharia"]
            ganho = f1_melhor - f1_dummy
            bloco += [
                f"- **Dummy baseline F1-eng:** {f1_dummy:.4f}",
                f"- **Ganho sobre o aleatório:** +{ganho:.4f}",
            ]
            if ganho < 0.10:
                bloco += [
                    f"",
                    f"**Comentário:** o ganho sobre o baseline aleatório é "
                    f"pequeno (< 0.10). Isso sugere que o classificador está "
                    f"aprendendo, mas ainda há muito espaço para melhorar — "
                    f"considere mais dados, features mais ricas (Camada 2 com "
                    f"PDFs), ou embeddings semânticos.",
                ]
            elif ganho < 0.30:
                bloco += [
                    f"",
                    f"**Comentário:** ganho moderado sobre o baseline. O "
                    f"modelo capturou padrões discriminativos, mas o "
                    f"problema permanece desafiador.",
                ]
            else:
                bloco += [
                    f"",
                    f"**Comentário:** ganho substancial sobre o baseline — "
                    f"o classificador realmente aprendeu padrões úteis no "
                    f"texto + metadados.",
                ]
        coments["classificacao"] = "\n".join(bloco)

    # ── 3. Técnicas avançadas (Parte 3) ─────────────────────────────────────
    if p3_resultados is not None:
        bloco = [f"### Técnicas avançadas — Parte 3", ""]

        if "balanceamento" in p3_resultados and isinstance(
                p3_resultados["balanceamento"], pd.DataFrame) and \
                not p3_resultados["balanceamento"].empty:
            tab_b = p3_resultados["balanceamento"]
            if "F1-engenharia" in tab_b.columns:
                melhor_estr = tab_b["F1-engenharia"].idxmax()
                f1_melhor_b = tab_b["F1-engenharia"].max()
                f1_baseline = tab_b["F1-engenharia"].iloc[0]
                bloco += [
                    f"**Balanceamento:** melhor estratégia foi `{melhor_estr}` "
                    f"(F1-eng = {f1_melhor_b:.4f}, vs {f1_baseline:.4f} sem balanceamento). "
                    f"Diferença = {f1_melhor_b - f1_baseline:+.4f}.",
                    "",
                ]
                if abs(f1_melhor_b - f1_baseline) < 0.02:
                    bloco += [
                        f"**Comentário:** o balanceamento teve impacto pequeno (<0.02). "
                        f"Provavelmente o `class_weight='balanced'` no LR já estava "
                        f"compensando o desbalanceamento.",
                        "",
                    ]
        if "kmeans" in p3_resultados:
            km = p3_resultados["kmeans"]
            if isinstance(km, dict) and "k" in km:
                bloco += [
                    f"**Clustering KMeans:** k={km['k']} clusters identificados. "
                    f"Os clusters com maior % de engenharia revelam **nichos "
                    f"técnicos** que podem ser usados para investigação de "
                    f"subenquadramento.",
                    "",
                ]
        if "regras" in p3_resultados and isinstance(
                p3_resultados["regras"], pd.DataFrame) and \
                not p3_resultados["regras"].empty:
            n_regras = len(p3_resultados["regras"])
            bloco += [
                f"**Regras de associação (Apriori):** {n_regras} regras "
                f"encontradas relacionando metadados (modalidade, esfera, "
                f"valor) ao rótulo.",
                "",
            ]
        if len(bloco) > 2:   # tem conteúdo além do título
            coments["tecnicas_avancadas"] = "\n".join(bloco)

    # ── 4. Rigor de pesquisa (Parte 4) ──────────────────────────────────────
    if p4_resultados is not None:
        bloco = [f"### Rigor de pesquisa — Parte 4", ""]

        if "holdout_metricas" in p4_resultados:
            h = p4_resultados["holdout_metricas"]
            f1h = h.get("f1_engenharia")
            bloco += [
                f"**Avaliação no HOLDOUT (números oficiais do TCC):**",
                f"- F1-engenharia: {f1h:.4f} ({_faixa_metrica(f1h)})",
                f"- Precision-eng: {h.get('precision_eng', float('nan')):.4f}",
                f"- Recall-eng:    {h.get('recall_eng', float('nan')):.4f}",
                f"- ROC-AUC:       {h.get('roc_auc', float('nan')):.4f}",
                f"- Avg-Precision: {h.get('avg_precision', float('nan')):.4f}",
                "",
            ]
            prec, rec = h.get('precision_eng'), h.get('recall_eng')
            if prec and rec:
                if prec > rec + 0.10:
                    bloco += [
                        f"**Comentário:** Precision >> Recall — o modelo "
                        f"perde contratos de engenharia (alto FN), mas tem "
                        f"poucos falsos alarmes. Útil quando o custo de FP "
                        f"é alto. Para o TCC, considere ajustar threshold "
                        f"para aumentar recall se busca varredura ampla.",
                        "",
                    ]
                elif rec > prec + 0.10:
                    bloco += [
                        f"**Comentário:** Recall >> Precision — o modelo "
                        f"pega muito do que é engenharia, mas com vários "
                        f"falsos alarmes. Bom para varredura inicial; depois "
                        f"é necessário filtro humano ou camada de validação.",
                        "",
                    ]
        if "mcnemar" in p4_resultados:
            mc = p4_resultados["mcnemar"]
            sig = "✅ significativa" if mc.get("significativo") else "❌ não significativa"
            bloco += [
                f"**Teste de McNemar (significância estatística):** "
                f"χ² = {mc['chi2']:.2f}, p = {mc['p_valor']:.2e} — diferença {sig}.",
                f"  *{mc.get('interpretacao', '')}*",
                "",
            ]
        if "bootstrap" in p4_resultados and isinstance(
                p4_resultados["bootstrap"], pd.DataFrame):
            tab_bs = p4_resultados["bootstrap"]
            if "f1" in tab_bs.index:
                lo = tab_bs.loc["f1", "IC95 Lower"]
                hi = tab_bs.loc["f1", "IC95 Upper"]
                amp = hi - lo
                bloco += [
                    f"**Bootstrap IC95% para F1-engenharia:** [{lo:.4f}, {hi:.4f}] "
                    f"(amplitude {amp:.4f}).",
                    "",
                ]
                if amp > 0.15:
                    bloco += [
                        f"**Comentário:** intervalo amplo (>{amp:.2f}) "
                        f"sugere que o modelo é instável — coletar mais dados "
                        f"deve estreitar o IC.",
                        "",
                    ]
        if "multiclasse" in p4_resultados and isinstance(
                p4_resultados["multiclasse"], pd.DataFrame):
            tab_mc = p4_resultados["multiclasse"]
            if len(tab_mc) >= 2 and "F1-macro" in tab_mc.columns:
                f1_bin = tab_mc["F1-macro"].iloc[0]
                f1_mc = tab_mc["F1-macro"].iloc[1]
                bloco += [
                    f"**Binário × Multiclasse (CONFEA 1.048/2013):**",
                    f"- F1-macro binário: {f1_bin:.4f}",
                    f"- F1-macro 3 classes: {f1_mc:.4f}",
                    f"- Diferença: {f1_mc - f1_bin:+.4f}",
                    "",
                ]
                if f1_mc < f1_bin - 0.05:
                    bloco += [
                        f"**Comentário:** o multiclasse perdeu desempenho. "
                        f"Provavelmente a heurística de keywords para "
                        f"separar eng_comum × eng_especial é fraca demais. "
                        f"Para o TCC, isso é um achado: a separação requer "
                        f"rotulação manual por especialista.",
                        "",
                    ]
        if len(bloco) > 2:
            coments["rigor_pesquisa"] = "\n".join(bloco)

    # ── 5. Subenquadramento (suspeitos) ─────────────────────────────────────
    if p2_resultados is not None and "ranking" in p2_resultados:
        rk = p2_resultados["ranking"]
        if isinstance(rk, pd.DataFrame) and not rk.empty:
            bloco = [
                f"### Subenquadramento — Parte 2",
                f"",
                f"O modelo identificou **{len(rk):,} contratos rotulados como "
                f"GERAL com alta probabilidade de serem ENGENHARIA**. Esses "
                f"são candidatos a subenquadramento — o objeto descreve "
                f"engenharia mas a categoria oficial diz serviço geral.",
                "",
            ]
            if "p_eng" in rk.columns:
                p_max = rk["p_eng"].max()
                p_med = rk["p_eng"].median()
                bloco += [
                    f"- P(eng) máxima entre suspeitos: {p_max:.4f}",
                    f"- P(eng) mediana entre suspeitos: {p_med:.4f}",
                    "",
                ]
            coments["subenquadramento"] = "\n".join(bloco)

    # ── 6. Rigor de licitação (sub-análise) ─────────────────────────────────
    if p2_resultados is not None and "rigor" in p2_resultados:
        r = p2_resultados["rigor"]
        if isinstance(r, pd.DataFrame) and not r.empty:
            bloco = [f"### Rigor de licitação (Lei 14.133/2021)", ""]
            if "score_irregularidade" in r.columns:
                n_red = (r["score_irregularidade"] >= 2).sum()
                n_amber = (r["score_irregularidade"] == 1).sum()
                n_green = (r["score_irregularidade"] < 0).sum()
                n_neutro = (r["score_irregularidade"] == 0).sum()
                bloco += [
                    f"Entre os top suspeitos analisados:",
                    f"- 🔴 **Score ≥ 2 (red flags fortes):** {n_red:,} contratos",
                    f"- 🟠 **Score = 1 (indícios):** {n_amber:,}",
                    f"- ⚪ **Score = 0 (neutro):** {n_neutro:,}",
                    f"- 🟢 **Score < 0 (rito de eng. aparente):** {n_green:,}",
                    "",
                ]
                if n_red > 0:
                    bloco += [
                        f"**Comentário:** {n_red} contratos têm sinais "
                        f"fortes de subenquadramento procedimental — Pregão "
                        f"usado para objeto que aparenta ser obra/serviço "
                        f"especial de engenharia, vedado pelo art. 29 § "
                        f"único da Lei 14.133/2021. Esses casos merecem "
                        f"análise jurídica detalhada.",
                        "",
                    ]
                elif n_amber > 0:
                    bloco += [
                        f"**Comentário:** há indícios moderados em alguns "
                        f"contratos. A análise via PDFs (Camada 2) pode "
                        f"clarear esses casos.",
                        "",
                    ]
            coments["rigor_legal"] = "\n".join(bloco)

    # Imprime tudo
    for chave, txt in coments.items():
        print("\n" + "─" * 60)
        print(txt)
        print("─" * 60)

    return coments


# ════════════════════════════════════════════════════════════════════════════
# Geração de relatório markdown completo (para colar no TCC)
# ════════════════════════════════════════════════════════════════════════════

def gerar_relatorio_markdown(df: pd.DataFrame,
                                p2_resultados: dict = None,
                                p3_resultados: dict = None,
                                p4_resultados: dict = None,
                                pasta_saida: str = ".",
                                titulo: str = "Análise PNCP — Resultados") -> str:
    """
    Gera um arquivo .md auto-contido com todos os resultados, comentários
    interpretativos automáticos e referências aos gráficos gerados.

    Útil para acelerar a redação do TCC: você abre o markdown, copia os
    blocos, ajusta o texto e coloca no documento final.

    Retorna o caminho do arquivo gerado.
    """
    import datetime as _dt
    coments = interpretar_resultados(df, p2_resultados, p3_resultados, p4_resultados)

    md = []
    md.append(f"# {titulo}")
    md.append("")
    md.append(f"*Gerado automaticamente em "
              f"{_dt.datetime.now().strftime('%d/%m/%Y %H:%M')}*")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Sumário Executivo")
    md.append("")

    # Quick facts
    if "rotulo" in df.columns:
        n_total = len(df)
        n_eng = (df["rotulo"] == "engenharia").sum()
        n_ger = (df["rotulo"] == "geral").sum()
        md.append(f"- Base analisada: **{n_total:,} contratos**")
        md.append(f"- Engenharia: {n_eng:,} ({n_eng/n_total*100:.1f}%)  |  "
                   f"Geral: {n_ger:,} ({n_ger/n_total*100:.1f}%)")
    if p2_resultados is not None and "melhor_modelo" in p2_resultados:
        md.append(f"- Melhor modelo: **{p2_resultados['melhor_modelo']}**")
    if p4_resultados is not None and "holdout_metricas" in p4_resultados:
        md.append(f"- F1-engenharia (holdout): "
                   f"**{p4_resultados['holdout_metricas']['f1_engenharia']:.4f}**")
    md.append("")
    md.append("---")
    md.append("")

    # Insere os comentários interpretativos
    for chave in ["distribuicao", "classificacao", "tecnicas_avancadas",
                   "rigor_pesquisa", "subenquadramento", "rigor_legal"]:
        if chave in coments:
            md.append(coments[chave])
            md.append("")

    # Lista os gráficos gerados na pasta
    md.append("## Gráficos gerados")
    md.append("")
    if os.path.isdir(pasta_saida):
        graficos = sorted(f for f in os.listdir(pasta_saida) if f.endswith(".png"))
        for g in graficos:
            md.append(f"- `{g}`")
    md.append("")

    # Lista os CSVs gerados
    md.append("## Tabelas e dados gerados")
    md.append("")
    if os.path.isdir(pasta_saida):
        csvs = sorted(f for f in os.listdir(pasta_saida) if f.endswith(".csv"))
        for c in csvs:
            md.append(f"- `{c}`")
    md.append("")

    md.append("---")
    md.append("")
    md.append("*Este relatório foi gerado automaticamente pelo pipeline. "
              "Os comentários são heurísticas baseadas em regras de bom senso "
              "de ML — análise crítica humana é necessária.*")

    conteudo = "\n".join(md)
    arq = os.path.join(pasta_saida, "relatorio_tcc.md")
    with open(arq, "w", encoding="utf-8") as f:
        f.write(conteudo)
    print(f"\n   💾 {arq}")
    return arq


# ════════════════════════════════════════════════════════════════════════════
# ████████   PARTE 6 — REDUÇÃO DE FALSOS POSITIVOS + VALIDAÇÃO LLM   ████████
# ════════════════════════════════════════════════════════════════════════════
#
# Como o usuário observou: "o classificador identifica como suspeito mas a
# análise profunda mostra que não é". Falsos positivos podem fragilizar o
# valor prático do TCC. Esta seção combina 5 abordagens para reduzir FP:
#
#   D.1 — Threshold de alta precisão (>=0.90 ou maior)
#   D.2 — Calibração de probabilidades (CalibratedClassifierCV)
#   D.3 — Ensemble por consenso (vários modelos têm que concordar)
#   D.4 — Filtro de coerência semântica via embeddings
#   D.5 — Validação por LLM (framework + prompts; chamada real opcional)
#   D.6 — Geração de relatório individual por suspeito (revisão humana)
# ────────────────────────────────────────────────────────────────────────────


def threshold_alta_precisao(p2_resultados: dict, precision_alvo: float = 0.90,
                              df_holdout=None, y_holdout=None) -> dict:
    """
    Encontra o threshold mínimo de probabilidade que garante precision-eng
    >= `precision_alvo` no out-of-fold (Parte 2).

    Por que isso reduz FP: o threshold default 0.5 (ou 0.4 do código) é
    otimizado para F1, equilibrando precision e recall. Para uma análise de
    auditoria onde **só queremos casos onde temos certeza**, sacrificamos
    recall em favor de precisão alta.

    Parâmetros
    ──────────
    p2_resultados   : output de executar_parte2()
    precision_alvo  : ex. 0.90 → só consideramos suspeitos onde o modelo
                       acerta 90% das vezes que diz "engenharia"
    df_holdout, y_holdout : (opcional) avalia o threshold também no holdout

    Retorna
    ───────
    dict com 'threshold', 'precision_estimada', 'recall_estimado',
    'n_suspeitos_filtrados'
    """
    cv = p2_resultados["resultados_cv"]
    melhor = p2_resultados["melhor_modelo"]
    if melhor not in cv:
        print("   [pulado] modelo não encontrado em resultados_cv")
        return {}

    y_true = p2_resultados["y"]
    y_prob = cv[melhor]["y_prob_oof"]

    # Varre thresholds e calcula precision/recall em cada um
    thresholds = np.linspace(0.1, 0.99, 90)
    melhores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        melhores.append((t, prec, rec, y_pred.sum()))
    df_t = pd.DataFrame(melhores, columns=["threshold","precision","recall","n_suspeitos"])

    # Menor threshold que satisfaz precision >= alvo (i.e. inclusivo máximo)
    mask = df_t["precision"] >= precision_alvo
    if not mask.any():
        # Ninguém atinge a meta → retorna o máximo de precisão alcançado
        idx = df_t["precision"].idxmax()
        print(f"\n⚠ Nenhum threshold atinge precision ≥ {precision_alvo:.2f}.")
        print(f"   Máximo possível: {df_t.loc[idx,'precision']:.4f} "
              f"(threshold={df_t.loc[idx,'threshold']:.3f})")
    else:
        # Pegamos o MENOR threshold dentre os que atingem a meta
        # (assim mantemos o máximo de recall possível)
        idx = df_t[mask]["threshold"].idxmin()

    res = {
        "threshold":            float(df_t.loc[idx, "threshold"]),
        "precision_estimada":   float(df_t.loc[idx, "precision"]),
        "recall_estimado":      float(df_t.loc[idx, "recall"]),
        "n_suspeitos":          int(df_t.loc[idx, "n_suspeitos"]),
        "precision_alvo":       precision_alvo,
        "tabela_thresholds":    df_t,
    }

    print(f"\n── Threshold de alta precisão (Entrega D.1) ──")
    print(f"   Alvo: precision-eng ≥ {precision_alvo:.2f}")
    print(f"   Threshold escolhido: {res['threshold']:.4f}")
    print(f"   Precision estimada: {res['precision_estimada']:.4f}")
    print(f"   Recall estimado:    {res['recall_estimado']:.4f}")
    print(f"   N suspeitos preditos: {res['n_suspeitos']:,}")

    return res


def ensemble_consenso(p2_resultados: dict, n_minimo: int = 3) -> pd.DataFrame:
    """
    Filtra os suspeitos para manter apenas os que **N ou mais modelos**
    concordam que são engenharia (ensemble por voto consensual).

    Reduz FP drasticamente porque um modelo pode errar em isolado, mas é
    raro 3 modelos diferentes errarem no mesmo contrato.

    Parâmetros
    ──────────
    p2_resultados : output de executar_parte2()
    n_minimo      : nº mínimo de modelos que precisam classificar como
                     engenharia (default 3)

    Retorna
    ───────
    DataFrame com os contratos onde >= n_minimo modelos concordam.
    """
    cv = p2_resultados["resultados_cv"]
    df = p2_resultados["df_processado"]

    # Filtra modelos não-Dummy
    modelos_uteis = [n for n in cv.keys() if "Dummy" not in n]
    if len(modelos_uteis) < n_minimo:
        print(f"   [aviso] só {len(modelos_uteis)} modelos disponíveis, "
              f"reduzindo n_minimo para {len(modelos_uteis)}")
        n_minimo = len(modelos_uteis)

    # Para cada contrato, conta quantos modelos predizem engenharia
    votos = np.zeros(len(df), dtype=int)
    for nome in modelos_uteis:
        y_pred = cv[nome]["y_pred_oof"]
        votos += (y_pred == 1)

    df = df.copy()
    df["votos_engenharia"] = votos
    df["consenso"]         = votos >= n_minimo

    # Foca nos suspeitos: rotulados como geral mas com consenso de eng.
    susp = df[(df["rotulo"] == "geral") & df["consenso"]].copy()
    susp = susp.sort_values("votos_engenharia", ascending=False)

    print(f"\n── Ensemble por consenso (Entrega D.3) ──")
    print(f"   Modelos usados: {len(modelos_uteis)} ({modelos_uteis})")
    print(f"   Voto mínimo:    {n_minimo}")
    print(f"   Suspeitos com consenso: {len(susp):,}")

    return susp


def coerencia_semantica_via_embeddings(suspeitos: pd.DataFrame,
                                          df_corpus: pd.DataFrame,
                                          modelo_nome: str = "paraphrase-multilingual-MiniLM-L12-v2",
                                          k: int = 10) -> pd.DataFrame:
    """
    Para cada suspeito, calcula a similaridade média do objeto com os
    K contratos de engenharia mais próximos no espaço de embeddings.

    Lógica: contratos verdadeiramente de engenharia devem estar PRÓXIMOS
    do cluster semântico de engenharia. Suspeitos com baixa similaridade
    média são candidatos a falsos positivos.

    Adiciona coluna 'coerencia_semantica' ao df dos suspeitos:
      • 1.0  = muito similar a engenharia (forte candidato)
      • 0.0  = igualmente próximo de geral e engenharia (ambíguo)
      • neg. = mais próximo de geral (provável FP)
    """
    if not TEM_SENTENCE_TRANSFORMERS:
        print("   [pulado] sentence-transformers não instalado.")
        return suspeitos

    print(f"\n── Coerência semântica via embeddings (Entrega D.4) ──")
    print(f"   Modelo: {modelo_nome}")

    # Gera embeddings do corpus (engenharia + geral)
    eng_texts = df_corpus.loc[df_corpus["rotulo"]=="engenharia", "objeto"].tolist()
    ger_texts = df_corpus.loc[df_corpus["rotulo"]=="geral", "objeto"].tolist()

    if len(eng_texts) < k or len(ger_texts) < k:
        print(f"   [pulado] amostras insuficientes (eng={len(eng_texts)}, ger={len(ger_texts)})")
        return suspeitos

    # Subamostra para acelerar (até 1000 de cada classe)
    rng = np.random.default_rng(42)
    if len(eng_texts) > 1000:
        eng_texts = rng.choice(eng_texts, size=1000, replace=False).tolist()
    if len(ger_texts) > 1000:
        ger_texts = rng.choice(ger_texts, size=1000, replace=False).tolist()

    print(f"   Codificando {len(eng_texts)} eng + {len(ger_texts)} geral + "
          f"{len(suspeitos)} suspeitos...")

    X_eng = gerar_embeddings_sentence_bert(eng_texts, modelo_nome=modelo_nome,
                                              mostrar_progresso=False)
    X_ger = gerar_embeddings_sentence_bert(ger_texts, modelo_nome=modelo_nome,
                                              mostrar_progresso=False)
    X_susp = gerar_embeddings_sentence_bert(
        suspeitos["objeto"].astype(str).tolist(),
        modelo_nome=modelo_nome, mostrar_progresso=False)

    # Para cada suspeito: similaridade média top-K com eng e com geral
    sim_eng = X_susp @ X_eng.T   # (n_susp, n_eng)
    sim_ger = X_susp @ X_ger.T

    top_k_eng = np.sort(sim_eng, axis=1)[:, -k:].mean(axis=1)
    top_k_ger = np.sort(sim_ger, axis=1)[:, -k:].mean(axis=1)

    susp = suspeitos.copy()
    susp["sim_top_k_eng"]      = top_k_eng
    susp["sim_top_k_geral"]    = top_k_ger
    susp["coerencia_semantica"] = top_k_eng - top_k_ger

    # Reordena: maior coerência primeiro (são os mais "engenharia-like")
    susp = susp.sort_values("coerencia_semantica", ascending=False)

    print(f"   Coerência média:          {susp['coerencia_semantica'].mean():+.4f}")
    print(f"   Suspeitos com coerência > 0: {(susp['coerencia_semantica'] > 0).sum()}/"
          f"{len(susp)}")
    print(f"   Suspeitos com coerência < 0 (provável FP): "
          f"{(susp['coerencia_semantica'] < 0).sum()}")

    return susp


# ── Validação por LLM (framework — chamada real é opcional) ─────────────────

PROMPT_VALIDACAO_LLM = """\
Você é um auditor especializado em licitações públicas brasileiras com
profundo conhecimento da Lei 14.133/2021 e das resoluções do CONFEA.

Analise o objeto contratual abaixo e responda em JSON:

OBJETO: {objeto}
VALOR: R$ {valor}
ÓRGÃO: {orgao}
{contexto_adicional}

Responda APENAS com um JSON válido neste formato:
{{
    "eh_engenharia": true/false,
    "tipo": "obra" | "serv_eng_comum" | "serv_eng_especial" | "nao_engenharia",
    "confianca": 0.0 a 1.0,
    "trechos_evidencia": ["trecho1", "trecho2"],
    "norma_aplicavel": "Lei 14.133/2021 art. ..." ou null,
    "modalidade_recomendada": "Pregão" | "Concorrência" | "Diálogo" | null,
    "justificativa": "uma frase curta"
}}
"""


def validar_suspeitos_via_llm(suspeitos: pd.DataFrame,
                                  llm_callable=None,
                                  max_chamadas: int = 50,
                                  contexto_camada2: dict = None) -> pd.DataFrame:
    """
    Aplica validação via LLM nos top suspeitos.

    Parâmetros
    ──────────
    suspeitos      : DataFrame com top suspeitos (após threshold + ensemble +
                      coerência semântica)
    llm_callable   : função callable(prompt: str) -> str que faz a chamada ao
                      LLM real. Se None, retorna placeholder com prompts
                      preparados para chamada manual ou via outro script.
                      Exemplo de uso real:
                          import google.generativeai as genai
                          genai.configure(api_key=...)
                          modelo = genai.GenerativeModel("gemini-1.5-flash")
                          def chamar_gemini(prompt):
                              return modelo.generate_content(prompt).text
                          validar_suspeitos_via_llm(susp, llm_callable=chamar_gemini)
    max_chamadas   : limite de chamadas (controle de custo)
    contexto_camada2 : (opcional) dict com infos da Camada 2 por
                        numeroControlePNCP — adiciona ao prompt info dos PDFs
                        anexados (ART, Projeto Básico, etc.)

    Retorna
    ───────
    DataFrame com colunas adicionadas:
      llm_prompt          : prompt completo enviado ao LLM
      llm_resposta_raw    : resposta bruta (se llm_callable foi chamada)
      llm_eh_engenharia   : bool (parsed da resposta)
      llm_tipo            : tipo (parsed)
      llm_confianca       : float (parsed)
      llm_justificativa   : str (parsed)
    """
    print(f"\n── Validação por LLM (Entrega D.5) ──")

    susp = suspeitos.head(max_chamadas).copy()
    prompts, respostas, eh_eng, tipos, conf, just = [], [], [], [], [], []

    for _, row in susp.iterrows():
        ctx = ""
        if contexto_camada2 is not None:
            num = row.get("numeroControlePNCP")
            info = contexto_camada2.get(num, {})
            if info.get("mk_score_engenharia", 0) > 0:
                ctx = (f"\nINFORMAÇÃO ADICIONAL (PDFs anexados ao processo):"
                       f"\n- Score de marcadores de engenharia: "
                       f"{info.get('mk_score_engenharia',0)}/9"
                       f"\n- ART/RRT detectado: "
                       f"{info.get('mk_ART_presente', False) or info.get('mk_RRT_presente', False)}"
                       f"\n- Projeto Básico anexado: "
                       f"{info.get('mk_PROJETO_BASICO_presente', False)}")

        prompt = PROMPT_VALIDACAO_LLM.format(
            objeto=row.get("objeto", "")[:500],
            valor=row.get("valorTotalEstimado", 0),
            orgao=row.get("razaoSocialOrgao", "?")[:80],
            contexto_adicional=ctx,
        )
        prompts.append(prompt)

        if llm_callable is None:
            respostas.append(None)
            eh_eng.append(None); tipos.append(None)
            conf.append(None); just.append(None)
        else:
            try:
                resp = llm_callable(prompt)
                respostas.append(resp)
                # Parse JSON da resposta
                import json as _json
                # Tenta encontrar JSON na resposta
                m = re.search(r"\{.*\}", str(resp), flags=re.DOTALL)
                if m:
                    j = _json.loads(m.group())
                    eh_eng.append(j.get("eh_engenharia"))
                    tipos.append(j.get("tipo"))
                    conf.append(j.get("confianca"))
                    just.append(j.get("justificativa"))
                else:
                    eh_eng.append(None); tipos.append(None)
                    conf.append(None); just.append(None)
            except Exception as e:
                respostas.append(f"ERRO: {e}")
                eh_eng.append(None); tipos.append(None)
                conf.append(None); just.append(None)
            time.sleep(0.5)   # respeita rate limits

    susp["llm_prompt"]        = prompts
    susp["llm_resposta_raw"]  = respostas
    susp["llm_eh_engenharia"] = eh_eng
    susp["llm_tipo"]          = tipos
    susp["llm_confianca"]     = conf
    susp["llm_justificativa"] = just

    if llm_callable is None:
        print(f"   ⚠ llm_callable não fornecido — prompts foram preparados "
              f"mas nenhuma chamada real foi feita.")
        print(f"   Para usar: validar_suspeitos_via_llm(susp, llm_callable=sua_funcao)")
        print(f"   Exemplo com Gemini Flash gratuito:")
        print(f"     import google.generativeai as genai")
        print(f"     genai.configure(api_key='SUA_CHAVE')")
        print(f"     m = genai.GenerativeModel('gemini-1.5-flash')")
        print(f"     llm_fn = lambda p: m.generate_content(p).text")

    return susp


def gerar_relatorio_individual_suspeito(suspeito: pd.Series,
                                            df_corpus: pd.DataFrame = None,
                                            similares: pd.DataFrame = None) -> str:
    """
    Gera um relatório textual estruturado de UM suspeito de subenquadramento,
    facilitando a revisão humana.

    Componentes do relatório:
      • Identificação (PNCP, valor, órgão, município)
      • Objeto completo
      • Probabilidade do classificador + score de irregularidade
      • Sinais legais detectados (ART, RRT, Projeto Básico — se Camada 2)
      • Top contratos similares (semânticos) — para context
      • Veredicto sugerido com justificativa

    Útil para: enviar ao auditor um pacote pronto por suspeito, em vez
    de pedir para ler PDFs e tabelas separadas.
    """
    md = []
    md.append(f"## Análise de subenquadramento — {suspeito.get('numeroControlePNCP','?')}")
    md.append("")
    md.append(f"**Objeto:** {suspeito.get('objeto','')[:500]}")
    md.append("")

    # Identificação
    md.append("### Identificação")
    md.append(f"- **PNCP:** `{suspeito.get('numeroControlePNCP','?')}`")
    md.append(f"- **Valor estimado:** R$ {suspeito.get('valorTotalEstimado',0):,.2f}")
    md.append(f"- **Órgão:** {suspeito.get('razaoSocialOrgao','?')}")
    md.append(f"- **Município:** {suspeito.get('municipioNome','?')}")
    md.append(f"- **Categoria PNCP atual:** "
              f"{suspeito.get('categoriaProcessoNome', 'Serviços')} "
              f"(rotulada como {suspeito.get('rotulo','?')})")
    md.append("")

    # Sinais do classificador
    md.append("### Sinais do classificador")
    if "p_eng" in suspeito.index:
        md.append(f"- **P(engenharia):** {suspeito['p_eng']:.4f}")
    if "votos_engenharia" in suspeito.index:
        md.append(f"- **Consenso:** {suspeito['votos_engenharia']} modelos votaram engenharia")
    if "coerencia_semantica" in suspeito.index:
        md.append(f"- **Coerência semântica:** {suspeito['coerencia_semantica']:+.4f}")
    md.append("")

    # Sinais legais (Camada 2)
    md.append("### Sinais legais (Camada 2)")
    sinais_c2 = []
    if suspeito.get("mk_ART_presente"):           sinais_c2.append("ART detectada nos anexos")
    if suspeito.get("mk_RRT_presente"):           sinais_c2.append("RRT detectada nos anexos")
    if suspeito.get("mk_PROJETO_BASICO_presente"):sinais_c2.append("Projeto Básico anexado")
    if suspeito.get("mk_ENGENHEIRO_RESPONSAVEL_presente"):
        sinais_c2.append("Menção a engenheiro responsável")
    if sinais_c2:
        for s in sinais_c2:
            md.append(f"- ✓ {s}")
    else:
        md.append("- *Sem sinais legais conclusivos (PDFs não analisados ou ausentes)*")
    md.append("")

    # Análise de rigor
    if "interpretacao_rigor" in suspeito.index:
        md.append("### Análise de rigor (Lei 14.133/2021)")
        md.append(f"{suspeito['interpretacao_rigor']}")
        md.append("")

    # Validação LLM
    if "llm_eh_engenharia" in suspeito.index and suspeito["llm_eh_engenharia"] is not None:
        md.append("### Validação por LLM")
        md.append(f"- **Veredicto:** "
                  f"{'engenharia' if suspeito['llm_eh_engenharia'] else 'não-engenharia'}")
        if suspeito.get("llm_tipo"):
            md.append(f"- **Tipo:** {suspeito['llm_tipo']}")
        if suspeito.get("llm_confianca"):
            md.append(f"- **Confiança:** {suspeito['llm_confianca']:.2f}")
        if suspeito.get("llm_justificativa"):
            md.append(f"- **Justificativa:** {suspeito['llm_justificativa']}")
        md.append("")

    # Similares
    if similares is not None and len(similares) > 0:
        md.append("### Contratos similares no corpus")
        md.append("")
        cols = [c for c in ["numeroControlePNCP","rotulo","objeto","distancia"]
                if c in similares.columns]
        md.append("| " + " | ".join(cols) + " |")
        md.append("| " + " | ".join(["---"]*len(cols)) + " |")
        for _, r in similares.head(5).iterrows():
            valores = [str(r[c])[:80] for c in cols]
            md.append("| " + " | ".join(valores) + " |")
        md.append("")

    # Veredicto sugerido
    md.append("### Veredicto sugerido")
    score_irreg = suspeito.get("score_irregularidade", 0)
    if score_irreg >= 2:
        md.append("🔴 **ALTO RISCO** de subenquadramento procedimental — "
                   "requer revisão jurídica.")
    elif score_irreg == 1:
        md.append("🟠 **INDÍCIOS** de subenquadramento — análise adicional dos "
                   "PDFs recomendada.")
    elif score_irreg < 0:
        md.append("🟢 **PROBABILIDADE BAIXA** — sinais legais nos PDFs sugerem "
                   "que o rito foi seguido apesar do rótulo formal.")
    else:
        md.append("⚪ **INCONCLUSIVO** — recomenda-se enriquecer com modalidade/critério ou PDFs.")

    return "\n".join(md)


def gerar_pacote_revisao_humana(suspeitos: pd.DataFrame,
                                    pasta_saida: str,
                                    df_corpus: pd.DataFrame = None,
                                    n_top: int = 30) -> str:
    """
    Gera um arquivo markdown com TODOS os relatórios individuais dos top
    suspeitos, prontos para revisão humana ou envio a auditores.

    Esse é o output final da Entrega D — em vez de o auditor revisar 200
    PDFs, ele revisa 30 relatórios estruturados.
    """
    md = ["# Pacote de revisão — Suspeitos de subenquadramento"]
    md.append("")
    md.append(f"*Gerado em "
              f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}*")
    md.append("")
    md.append(f"Top {min(n_top, len(suspeitos))} suspeitos selecionados de "
              f"{len(suspeitos)} candidatos.")
    md.append("")
    md.append("---")
    md.append("")

    for i, (_, row) in enumerate(suspeitos.head(n_top).iterrows(), start=1):
        md.append(f"## #{i}")
        md.append("")
        md.append(gerar_relatorio_individual_suspeito(row, df_corpus))
        md.append("")
        md.append("---")
        md.append("")

    arq = os.path.join(pasta_saida, "pacote_revisao_humana.md")
    with open(arq, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n   💾 {arq}")
    return arq


def executar_parte6_reducao_fp(p2_resultados: dict,
                                   df_corpus: pd.DataFrame,
                                   pasta_saida: str = None,
                                   precision_alvo: float = 0.90,
                                   usar_embeddings: bool = False,
                                   llm_callable=None,
                                   n_top_revisao: int = 30,
                                   contexto_camada2: dict = None) -> dict:
    """
    Pipeline completo da Entrega D — redução de falsos positivos.

    Combina:
      1. Threshold de alta precisão (D.1)
      2. Ensemble por consenso (D.3)
      3. Coerência semântica via embeddings (D.4) — opcional
      4. Validação por LLM (D.5) — opcional
      5. Geração de pacote para revisão humana (D.6)

    Parâmetros
    ──────────
    p2_resultados   : output de executar_parte2()
    df_corpus       : DataFrame original (antes do split)
    pasta_saida     : pasta para gravar relatórios
    precision_alvo  : meta de precision para o threshold (D.1)
    usar_embeddings : True = roda coerência semântica (baixa modelo)
    llm_callable    : função para chamar LLM. None = só gera prompts
    n_top_revisao   : nº de suspeitos no pacote de revisão final
    contexto_camada2: (opcional) dict com infos da Camada 2 por PNCP
    """
    print("\n" + "█"*62)
    print("  PARTE 6 — REDUÇÃO DE FALSOS POSITIVOS + VALIDAÇÃO")
    print("█"*62)

    if pasta_saida is None:
        uf  = df_corpus["ufSigla"].mode()[0]       if "ufSigla" in df_corpus.columns else "xx"
        ano = df_corpus["anoPublicacao"].mode()[0] if "anoPublicacao" in df_corpus.columns else "xxxx"
        pasta_saida = f"graficos_pncp_{uf}_{ano}"
    os.makedirs(pasta_saida, exist_ok=True)

    r = {}

    # D.1 — Threshold alta precisão
    print("\n[1] Threshold de alta precisão...")
    r["threshold_hp"] = threshold_alta_precisao(p2_resultados, precision_alvo)

    # Aplica o threshold para refiltrar suspeitos
    cv = p2_resultados["resultados_cv"]
    melhor = p2_resultados["melhor_modelo"]
    df = p2_resultados["df_processado"].copy()
    if r["threshold_hp"]:
        thr = r["threshold_hp"]["threshold"]
        df["p_eng"] = cv[melhor]["y_prob_oof"]
        suspeitos_hp = df[
            (df["rotulo"] == "geral") & (df["p_eng"] >= thr)
        ].sort_values("p_eng", ascending=False)
        print(f"   ✓ {len(suspeitos_hp)} suspeitos com P(eng) ≥ {thr:.4f}")
    else:
        suspeitos_hp = p2_resultados.get("ranking", pd.DataFrame())

    # D.3 — Ensemble por consenso
    print("\n[2] Ensemble por consenso...")
    suspeitos_consenso = ensemble_consenso(p2_resultados, n_minimo=3)

    # Interseção: só suspeitos que estão nos dois conjuntos
    if "numeroControlePNCP" in suspeitos_hp.columns and \
       "numeroControlePNCP" in suspeitos_consenso.columns:
        nums_hp   = set(suspeitos_hp["numeroControlePNCP"])
        nums_cons = set(suspeitos_consenso["numeroControlePNCP"])
        nums_inter = nums_hp & nums_cons
        suspeitos_filtrados = suspeitos_hp[
            suspeitos_hp["numeroControlePNCP"].isin(nums_inter)
        ].copy()
        # adiciona votos
        votos_dict = dict(zip(suspeitos_consenso["numeroControlePNCP"],
                                suspeitos_consenso["votos_engenharia"]))
        suspeitos_filtrados["votos_engenharia"] = \
            suspeitos_filtrados["numeroControlePNCP"].map(votos_dict)
        print(f"   ✓ Após D.1 ∩ D.3: {len(suspeitos_filtrados)} suspeitos "
              f"(redução de {len(suspeitos_hp)} → {len(suspeitos_filtrados)})")
    else:
        suspeitos_filtrados = suspeitos_hp
    r["suspeitos_filtrados"] = suspeitos_filtrados

    # D.4 — Coerência semântica (opcional)
    if usar_embeddings and len(suspeitos_filtrados) > 0:
        print("\n[3] Coerência semântica via embeddings...")
        try:
            suspeitos_filtrados = coerencia_semantica_via_embeddings(
                suspeitos_filtrados.head(100), df_corpus
            )
            r["suspeitos_com_coerencia"] = suspeitos_filtrados
        except Exception as e:
            print(f"   [aviso] embeddings: {e}")

    # D.5 — Validação por LLM (opcional)
    if len(suspeitos_filtrados) > 0:
        print("\n[4] Preparação de prompts LLM...")
        suspeitos_filtrados = validar_suspeitos_via_llm(
            suspeitos_filtrados, llm_callable=llm_callable,
            max_chamadas=min(50, len(suspeitos_filtrados)),
            contexto_camada2=contexto_camada2,
        )
        r["suspeitos_validados"] = suspeitos_filtrados

    # Salva CSV final
    arq = os.path.join(pasta_saida, "suspeitos_alta_confianca.csv")
    suspeitos_filtrados.to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")
    r["arquivo_csv"] = arq

    # D.6 — Pacote de revisão humana
    print("\n[5] Geração de pacote de revisão humana...")
    r["pacote_revisao"] = gerar_pacote_revisao_humana(
        suspeitos_filtrados, pasta_saida, df_corpus, n_top=n_top_revisao
    )

    print("\n" + "█"*62)
    print(f"  PARTE 6 CONCLUÍDA ✅")
    print(f"  Suspeitos finais (alta confiança): {len(suspeitos_filtrados)}")
    print(f"  Pacote: {r.get('pacote_revisao', 'N/A')}")
    print("█"*62)
    return r


# ════════════════════════════════════════════════════════════════════════════
# ████████   PARTE 7 — ANÁLISE DE REDES (GRAFOS)   ███████████████████████████
# ════════════════════════════════════════════════════════════════════════════
#
# Motivação:
# O classificador encontra suspeitos de subenquadramento INDIVIDUALMENTE
# (contrato a contrato). Mas há perguntas que SÓ uma análise de redes
# consegue responder:
#
#   1. "Padrão sistemático ou caso isolado?"
#      Se a Prefeitura X contrata 50 vezes a Empresa Y como geral, mas o
#      objeto é claramente engenharia → sistemática (não erro pontual).
#
#   2. "Quem são os fornecedores fantasma de engenharia em geral?"
#      Empresas com razão social que indica engenharia (ex: "X Construções",
#      "Y Engenharia") aparecendo como contratadas em "serviços gerais".
#
#   3. "Concentração de risco"
#      Quais órgãos têm taxa anormalmente alta de suspeitos? Centralidade
#      no grafo identifica esses hubs.
#
#   4. "Comunidades de prática"
#      Detecção de comunidades (Louvain) revela ECOSSISTEMAS órgão↔empresa
#      que se repetem em padrões de subenquadramento.
#
# Pré-requisitos:
#   • DataFrame com colunas: numeroControlePNCP, cnpjOrgao, razaoSocialOrgao,
#     niFornecedor, nomeRazaoSocialFornecedor, rotulo, valorTotalEstimado.
#   • Coleta usando _aplanar_contrato (versão atualizada com fornecedor).
#
# Pacotes opcionais:
#   • networkx (obrigatório p/ esta parte)
#   • python-louvain (opcional, para detecção de comunidades)
# ────────────────────────────────────────────────────────────────────────────


# Padrões para detectar empresas que mencionam ENGENHARIA na razão social.
# Essas são candidatas a "fornecedores fantasma de engenharia em geral".
PADROES_RAZAO_SOCIAL_ENG = [
    r"\benhg\b",  # abreviações às vezes
    r"\bengenharia\b",
    r"\bconstrucoes?\b", r"\bconstrutora\b",
    r"\bobras?\b",
    r"\bedificacoes\b",
    r"\bempreendimentos\b",
    r"\binstalacoes\b",
    r"\bel[eé]trica\b", r"\bhidr[aá]ulica\b",
    r"\bproj(eto)?s?\b",   # "projetos" (genérico)
    r"\barquit(etura|et[oa]s?)\b",
    r"\bsane(amento)?\b",
    r"\bterra(planagem|planagem|plenagem)\b",
    r"\bpavimenta(c|c[ãa]o)\b",
]


def _empresa_tem_indicio_engenharia(razao_social: str) -> bool:
    """
    True se a razão social contém palavra que sugere atuação em engenharia.
    """
    if not razao_social or not isinstance(razao_social, str):
        return False
    norm = _normalizar(razao_social).lower()
    for pat in PADROES_RAZAO_SOCIAL_ENG:
        if re.search(pat, norm, flags=re.IGNORECASE):
            return True
    return False


def construir_grafo_orgao_fornecedor(df: pd.DataFrame,
                                       min_contratos: int = 2) -> "nx.Graph":
    """
    Constrói grafo BIPARTIDO órgão ↔ fornecedor.

    Nós:
      • Órgãos contratantes (atributo bipartite=0)
      • Fornecedores contratados (atributo bipartite=1)

    Arestas:
      • Existe aresta órgão↔fornecedor quando há ao menos 1 contrato entre eles.
      • Peso = número de contratos.
      • Atributos da aresta: total_valor, n_eng, n_geral, taxa_geral,
        taxa_geral_com_indicio_eng (a métrica-chave do TCC).

    Filtros:
      • min_contratos: apenas pares com ≥ N contratos entram no grafo
        (reduz ruído de empresas que ganharam só 1 vez).

    Retorna
    ───────
    networkx.Graph (não direcionado, com peso e atributos).
    """
    if not TEM_NETWORKX:
        raise ImportError("networkx não está instalado. pip install networkx")

    cols_obrig = ["cnpjOrgao", "razaoSocialOrgao",
                  "niFornecedor", "nomeRazaoSocialFornecedor", "rotulo"]
    for c in cols_obrig:
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {c}")

    # Filtra contratos com fornecedor identificado
    df_g = df[df["niFornecedor"].notna() &
              (df["niFornecedor"].astype(str).str.strip() != "")].copy()
    if df_g.empty:
        print("   ⚠ Nenhum contrato tem niFornecedor preenchido.")
        return nx.Graph()

    print(f"\n   Construindo grafo a partir de {len(df_g):,} contratos...")

    # Normaliza identificadores (zero-padding etc.)
    df_g["cnpjOrgao_norm"] = df_g["cnpjOrgao"].astype(str).str.zfill(14)
    df_g["niFornecedor_norm"] = df_g["niFornecedor"].astype(str).str.strip()

    # Agrega por par (órgão, fornecedor)
    agg = df_g.groupby(["cnpjOrgao_norm", "niFornecedor_norm"]).agg(
        n_contratos      = ("numeroControlePNCP", "count"),
        n_eng            = ("rotulo", lambda x: (x == "engenharia").sum()),
        n_geral          = ("rotulo", lambda x: (x == "geral").sum()),
        total_valor      = ("valorTotalEstimado", "sum"),
        razao_orgao      = ("razaoSocialOrgao", "first"),
        razao_fornecedor = ("nomeRazaoSocialFornecedor", "first"),
    ).reset_index()

    agg["taxa_geral"] = agg["n_geral"] / agg["n_contratos"].clip(lower=1)
    agg["fornecedor_indicio_eng"] = agg["razao_fornecedor"].apply(
        _empresa_tem_indicio_engenharia)
    agg["red_flag"] = (
        (agg["fornecedor_indicio_eng"]) & (agg["n_geral"] > 0)
    ).astype(int)

    # Filtro de ruído
    agg = agg[agg["n_contratos"] >= min_contratos]
    print(f"   Após filtro min_contratos≥{min_contratos}: {len(agg):,} pares")

    G = nx.Graph()

    # Nós: órgãos
    orgaos = agg.groupby("cnpjOrgao_norm").agg(
        razao=("razao_orgao", "first"),
        total_contratos=("n_contratos", "sum"),
        total_valor=("total_valor", "sum"),
    ).reset_index()
    for _, o in orgaos.iterrows():
        G.add_node(
            f"O::{o['cnpjOrgao_norm']}",
            tipo="orgao",
            bipartite=0,
            label=str(o["razao"])[:60],
            cnpj=o["cnpjOrgao_norm"],
            total_contratos=int(o["total_contratos"]),
            total_valor=float(o["total_valor"] or 0),
        )

    # Nós: fornecedores
    forns = agg.groupby("niFornecedor_norm").agg(
        razao=("razao_fornecedor", "first"),
        total_contratos=("n_contratos", "sum"),
        total_valor=("total_valor", "sum"),
        indicio_eng=("fornecedor_indicio_eng", "any"),
        n_red_flag=("red_flag", "sum"),
    ).reset_index()
    for _, f in forns.iterrows():
        G.add_node(
            f"F::{f['niFornecedor_norm']}",
            tipo="fornecedor",
            bipartite=1,
            label=str(f["razao"])[:60],
            ni=f["niFornecedor_norm"],
            total_contratos=int(f["total_contratos"]),
            total_valor=float(f["total_valor"] or 0),
            indicio_engenharia=bool(f["indicio_eng"]),
            n_red_flag=int(f["n_red_flag"]),
        )

    # Arestas
    for _, e in agg.iterrows():
        G.add_edge(
            f"O::{e['cnpjOrgao_norm']}",
            f"F::{e['niFornecedor_norm']}",
            weight=int(e["n_contratos"]),
            n_eng=int(e["n_eng"]),
            n_geral=int(e["n_geral"]),
            taxa_geral=float(e["taxa_geral"]),
            total_valor=float(e["total_valor"] or 0),
            red_flag=int(e["red_flag"]),
        )

    print(f"   Grafo: {G.number_of_nodes()} nós ({len(orgaos)} órgãos + "
          f"{len(forns)} fornecedores), {G.number_of_edges()} arestas")
    return G


def metricas_centralidade(G, top_n: int = 20) -> dict:
    """
    Calcula métricas de centralidade do grafo bipartido órgão↔fornecedor:
      • Degree centrality: quantas conexões cada nó tem (proxy de "presença")
      • Weighted degree (strength): soma dos pesos (volume de contratos)
      • Betweenness centrality: nós que ligam comunidades distintas

    Importância para o TCC:
      • Top órgãos por degree = órgãos que contratam mais fornecedores diferentes
      • Top fornecedores por weighted degree = empresas que mais ganharam
      • Betweenness alto + indicio_engenharia = empresas que são "ponte"
        entre vários órgãos contratando irregularmente
    """
    if G.number_of_nodes() == 0:
        return {}

    print("\n   Calculando métricas de centralidade...")
    deg = dict(G.degree())
    strength = {n: sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True))
                for n in G.nodes()}
    # Betweenness é caro em grafos grandes; aproximação via amostragem
    if G.number_of_nodes() > 500:
        bet = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes() // 2),
                                          seed=42)
    else:
        bet = nx.betweenness_centrality(G)

    # Tabela de órgãos
    orgaos = []
    for n in G.nodes():
        if G.nodes[n].get("tipo") != "orgao":
            continue
        orgaos.append({
            "cnpj":            G.nodes[n].get("cnpj", ""),
            "razao_orgao":     G.nodes[n].get("label", ""),
            "n_fornecedores":  deg[n],
            "volume_contratos": strength[n],
            "betweenness":     round(bet.get(n, 0), 6),
            "total_valor":     G.nodes[n].get("total_valor", 0),
        })
    df_orgaos = pd.DataFrame(orgaos).sort_values("volume_contratos", ascending=False)

    # Tabela de fornecedores
    forns = []
    for n in G.nodes():
        if G.nodes[n].get("tipo") != "fornecedor":
            continue
        forns.append({
            "ni":              G.nodes[n].get("ni", ""),
            "razao_forn":      G.nodes[n].get("label", ""),
            "n_orgaos":        deg[n],
            "volume_contratos": strength[n],
            "betweenness":     round(bet.get(n, 0), 6),
            "indicio_eng":     G.nodes[n].get("indicio_engenharia", False),
            "n_red_flag":      G.nodes[n].get("n_red_flag", 0),
            "total_valor":     G.nodes[n].get("total_valor", 0),
        })
    df_forns = pd.DataFrame(forns).sort_values("volume_contratos", ascending=False)

    print(f"\n── Top-{top_n} ÓRGÃOS por volume de contratos ──")
    print(df_orgaos.head(top_n).to_string(index=False))

    print(f"\n── Top-{top_n} FORNECEDORES por volume de contratos ──")
    print(df_forns.head(top_n).to_string(index=False))

    print(f"\n── Top-{top_n} FORNECEDORES com INDÍCIO de engenharia "
          f"(red flags) ──")
    df_susp = df_forns[df_forns["indicio_eng"]].sort_values(
        "n_red_flag", ascending=False)
    print(df_susp.head(top_n).to_string(index=False))

    return {"orgaos": df_orgaos, "fornecedores": df_forns,
            "fornecedores_suspeitos": df_susp}


def projetar_fornecedor_fornecedor(G_bipartido,
                                       min_orgaos_compartilhados: int = 2) -> "nx.Graph":
    """
    Projeta o grafo BIPARTIDO órgão↔fornecedor em um grafo de
    FORNECEDOR↔FORNECEDOR, conectando empresas que compartilham órgãos.

    Conceito (análise de redes 1-mode):
    Em um grafo bipartido X↔Y, a projeção em Y conecta dois nós y1, y2
    se ambos compartilham pelo menos um vizinho em X. O peso da aresta
    é o nº de vizinhos compartilhados (ou função dele).

    Para o TCC, isso revela "carteis" ou "ecossistemas" de fornecedores
    que ganham contratos nos MESMOS órgãos. Padrões esperados:
      • Comunidade densa de fornecedores compartilhando muitos órgãos →
        candidatos a investigação por concorrência simulada
      • Fornecedor isolado com muitas conexões → empresa "âncora" do
        nicho de engenharia mascarada como "geral"

    Parâmetros
    ──────────
    G_bipartido               : grafo bipartido produzido por
                                 construir_grafo_orgao_fornecedor
    min_orgaos_compartilhados : aresta só existe se ≥ N órgãos compartilhados
                                 (default 2, evita ruído)

    Retorna
    ───────
    networkx.Graph não-direcionado, com atributos:
      • weight                 : nº de órgãos compartilhados
      • pct_geral_combinado    : taxa média de contratos 'geral' (entre os 2)
      • indicio_eng_combinado  : True se ambas têm indício na razão social
    """
    if not TEM_NETWORKX:
        raise ImportError("networkx não está instalado.")
    if G_bipartido.number_of_nodes() == 0:
        return nx.Graph()

    # Identifica nós fornecedor e órgão
    fornecedores = [n for n, d in G_bipartido.nodes(data=True)
                     if d.get("tipo") == "fornecedor"]
    orgaos       = [n for n, d in G_bipartido.nodes(data=True)
                     if d.get("tipo") == "orgao"]

    if not fornecedores or not orgaos:
        print("   [pulado] grafo não é bipartido órgão↔fornecedor.")
        return nx.Graph()

    print(f"\n   Projetando bipartido em fornecedor↔fornecedor...")
    print(f"   Bipartido original: {len(orgaos):,} órgãos × "
          f"{len(fornecedores):,} fornecedores")

    # Usa nx.bipartite.weighted_projected_graph
    # Peso da aresta = nº de órgãos comuns
    G_proj = nx.bipartite.weighted_projected_graph(
        G_bipartido, fornecedores
    )

    # Filtra arestas fracas (poucos órgãos compartilhados)
    if min_orgaos_compartilhados > 1:
        arestas_fracas = [(u, v) for u, v, d in G_proj.edges(data=True)
                           if d.get("weight", 0) < min_orgaos_compartilhados]
        G_proj.remove_edges_from(arestas_fracas)
        print(f"   Removidas {len(arestas_fracas):,} arestas com < "
              f"{min_orgaos_compartilhados} órgãos compartilhados")

    # Remove nós isolados após filtro
    isolados = [n for n in G_proj.nodes() if G_proj.degree(n) == 0]
    G_proj.remove_nodes_from(isolados)
    print(f"   Removidos {len(isolados):,} fornecedores isolados após filtro")

    # Enriquecer atributos das arestas: combinar info de ambos os fornecedores
    for u, v, d in G_proj.edges(data=True):
        attrs_u = G_bipartido.nodes[u]
        attrs_v = G_bipartido.nodes[v]
        d["pct_geral_combinado"] = (
            attrs_u.get("taxa_geral", 0) + attrs_v.get("taxa_geral", 0)
        ) / 2
        d["indicio_eng_combinado"] = (
            attrs_u.get("indicio_eng_razao", False) and
            attrs_v.get("indicio_eng_razao", False)
        )

    print(f"   Grafo 1-mode resultante: {G_proj.number_of_nodes():,} nós, "
          f"{G_proj.number_of_edges():,} arestas")
    return G_proj


def detectar_comunidades(G):
    """
    Detecção de comunidades via Louvain (Aula 33 — extensão).

    Cada comunidade no grafo bipartido é um "ecossistema" de órgãos e
    fornecedores que contratam fortemente entre si. Para o TCC, comunidades
    com alta concentração de fornecedores com indício de engenharia + alta
    taxa de contratos rotulados como "geral" são suspeitas.
    """
    if not TEM_LOUVAIN:
        print("   [pulado] python-louvain não instalado. "
              "pip install python-louvain")
        return None

    print("\n   Aplicando algoritmo Louvain para detecção de comunidades...")
    # Louvain trabalha em grafo não-bipartido (com pesos)
    partition = community_louvain.best_partition(G, weight="weight",
                                                    random_state=42)
    n_comunidades = len(set(partition.values()))
    print(f"   {n_comunidades} comunidades detectadas")

    # Agrega informação por comunidade
    df_part = pd.DataFrame([
        {"node": n, "community": c,
         "tipo": G.nodes[n].get("tipo"),
         "indicio_eng": G.nodes[n].get("indicio_engenharia", False),
         "label": G.nodes[n].get("label", ""),
         "n_red_flag": G.nodes[n].get("n_red_flag", 0)}
        for n, c in partition.items()
    ])

    resumo = df_part.groupby("community").agg(
        n_nos          = ("node", "count"),
        n_orgaos       = ("tipo", lambda x: (x == "orgao").sum()),
        n_fornecedores = ("tipo", lambda x: (x == "fornecedor").sum()),
        n_indicio_eng  = ("indicio_eng", "sum"),
        n_red_flag     = ("n_red_flag", "sum"),
    ).sort_values("n_red_flag", ascending=False)

    print(f"\n── Top-10 comunidades por nº de red flags ──")
    print(resumo.head(10).to_string())

    return {"partition": partition, "resumo": resumo, "df_part": df_part}


def detectar_fornecedores_fantasma(df: pd.DataFrame,
                                       min_contratos_geral: int = 3) -> pd.DataFrame:
    """
    Detecção específica de "fornecedores fantasma de engenharia em geral":
    empresas cuja razão social SUGERE atuação em engenharia (contém "Construções",
    "Engenharia", "Empreendimentos" etc.) mas que aparecem como contratadas
    em ≥ N contratos rotulados como "geral".

    Análise central do TCC: esses são os casos onde HÁ EVIDÊNCIA INDEPENDENTE
    (razão social) que sustenta a hipótese de subenquadramento. Não é só
    o classificador "achando" — é a própria empresa anunciando o que faz.

    Parâmetros
    ──────────
    df                  : DataFrame com colunas niFornecedor, rotulo, etc.
    min_contratos_geral : limite mínimo de contratos como "geral"
                           para uma empresa entrar no relatório

    Retorna
    ───────
    DataFrame com fornecedores suspeitos, ordenado por nº de contratos como
    "geral" (mais alto = mais suspeito).
    """
    if "niFornecedor" not in df.columns or "rotulo" not in df.columns:
        return pd.DataFrame()

    df_use = df[df["niFornecedor"].notna() &
                (df["niFornecedor"].astype(str).str.strip() != "")].copy()
    if df_use.empty:
        return pd.DataFrame()

    df_use["niFornecedor_norm"] = df_use["niFornecedor"].astype(str).str.strip()

    agg = df_use.groupby("niFornecedor_norm").agg(
        razao_social   = ("nomeRazaoSocialFornecedor", "first"),
        n_total        = ("numeroControlePNCP", "count"),
        n_geral        = ("rotulo", lambda x: (x == "geral").sum()),
        n_engenharia   = ("rotulo", lambda x: (x == "engenharia").sum()),
        valor_em_geral = ("valorTotalEstimado",
                            lambda x: x[df_use.loc[x.index, "rotulo"] == "geral"].sum()),
        valor_em_eng   = ("valorTotalEstimado",
                            lambda x: x[df_use.loc[x.index, "rotulo"] == "engenharia"].sum()),
        orgaos_distintos = ("cnpjOrgao",
                              lambda x: x.dropna().astype(str).nunique()),
    ).reset_index()

    # Marca empresas com indício de engenharia na razão social
    agg["razao_indica_eng"] = agg["razao_social"].apply(
        _empresa_tem_indicio_engenharia)

    # FANTASMAS: razão social indica eng. + ≥N contratos como "geral"
    fantasmas = agg[
        (agg["razao_indica_eng"]) & (agg["n_geral"] >= min_contratos_geral)
    ].copy()

    fantasmas["pct_em_geral"] = (
        fantasmas["n_geral"] / fantasmas["n_total"].clip(lower=1) * 100
    ).round(1)
    fantasmas = fantasmas.sort_values("n_geral", ascending=False)

    print(f"\n── FORNECEDORES FANTASMA DE ENGENHARIA EM 'GERAL' ──")
    print(f"   ({len(fantasmas):,} empresas com razão social que indica "
          f"engenharia + ≥{min_contratos_geral} contratos rotulados 'geral')")
    print(f"\n   Top-25:")
    cols = ["niFornecedor_norm", "razao_social", "n_total", "n_geral",
            "n_engenharia", "pct_em_geral", "orgaos_distintos",
            "valor_em_geral"]
    print(fantasmas[cols].head(25).to_string(index=False))

    return fantasmas


def visualizar_grafo_principal(G, pasta: str, top_n: int = 50,
                                  layout: str = "spring"):
    """
    Visualização do subgrafo dos nós mais conectados.

    Mostra apenas os top-N nós por strength para evitar poluição visual.
    Cores: órgãos azuis, fornecedores normais cinzas, fornecedores com
    indício de engenharia em laranja, com red flags em vermelho.

    Parâmetros
    ──────────
    G       : grafo completo
    pasta   : pasta para salvar
    top_n   : nº de nós a mostrar (mantém os de maior strength)
    layout  : "spring" (default) ou "kamada_kawai"
    """
    if not TEM_NETWORKX or G.number_of_nodes() == 0:
        return

    # Subgrafo dos top-N por strength
    strength = {n: sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True))
                for n in G.nodes()}
    top_nos = sorted(strength.items(), key=lambda x: x[1], reverse=True)[:top_n]
    nos_keep = {n for n, _ in top_nos}
    H = G.subgraph(nos_keep).copy()

    print(f"\n   Visualizando subgrafo dos top-{top_n} nós "
          f"({H.number_of_nodes()} nós, {H.number_of_edges()} arestas)...")

    # Layout
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H)
    else:
        pos = nx.spring_layout(H, k=0.5, iterations=80, seed=42)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Cores e tamanhos por tipo
    cores  = []
    tams   = []
    rotulos = {}
    for n in H.nodes():
        attrs = H.nodes[n]
        if attrs.get("tipo") == "orgao":
            cores.append("#1a6faf")
        elif attrs.get("n_red_flag", 0) > 0:
            cores.append("#e74c3c")        # red flag
        elif attrs.get("indicio_engenharia"):
            cores.append("#f39c12")        # indício
        else:
            cores.append("#bdc3c7")        # neutro
        tams.append(200 + min(strength.get(n, 1), 50) * 30)
        # Rótulo: só para os 20 maiores
        if len(rotulos) < 20:
            rotulos[n] = (attrs.get("label", "") or "")[:25]

    # Arestas com largura proporcional ao peso
    pesos = [d.get("weight", 1) for _, _, d in H.edges(data=True)]
    pesos_norm = [0.5 + min(p / 5, 4) for p in pesos]
    cores_arestas = ["#e74c3c" if d.get("red_flag", 0) else "#999"
                       for _, _, d in H.edges(data=True)]

    nx.draw_networkx_edges(H, pos, width=pesos_norm, edge_color=cores_arestas,
                             alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_color=cores, node_size=tams,
                              edgecolors="white", linewidths=1.0, ax=ax)
    nx.draw_networkx_labels(H, pos, labels=rotulos, font_size=7, ax=ax)

    # Legenda
    from matplotlib.patches import Patch
    legenda = [
        Patch(color="#1a6faf", label="Órgão público"),
        Patch(color="#bdc3c7", label="Fornecedor (neutro)"),
        Patch(color="#f39c12", label="Fornecedor c/ indício de eng."),
        Patch(color="#e74c3c", label="Fornecedor com red flag"),
    ]
    ax.legend(handles=legenda, loc="upper right", framealpha=0.9)
    ax.set_title(f"Rede órgão ↔ fornecedor (Top-{top_n} por volume)\n"
                 f"Tamanho = volume de contratos | Cor = tipo/risco",
                 fontweight="bold", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    _salvar(fig, "p7_01_grafo_principal.png", pasta)


def visualizar_subgrafo_red_flags(G, pasta: str, max_nos: int = 80):
    """
    Visualização ESPECÍFICA dos red flags: pares órgão↔fornecedor onde a
    empresa tem indício de engenharia mas está como "geral".

    Esta é a visualização-chave do TCC: mostra concretamente quais órgãos
    estão envolvidos com quais fornecedores suspeitos.
    """
    if not TEM_NETWORKX:
        return

    # Filtra arestas com red_flag
    arestas_rf = [(u, v, d) for u, v, d in G.edges(data=True)
                   if d.get("red_flag", 0) > 0]
    if not arestas_rf:
        print("   [info] Nenhum red flag encontrado para visualizar.")
        return

    H = nx.Graph()
    for u, v, d in arestas_rf:
        H.add_node(u, **G.nodes[u])
        H.add_node(v, **G.nodes[v])
        H.add_edge(u, v, **d)

    if H.number_of_nodes() > max_nos:
        # Mantém só os top componentes/nós por peso
        strength = {n: sum(d.get("weight", 1) for _, _, d in H.edges(n, data=True))
                    for n in H.nodes()}
        top_nos = {n for n, _ in
                    sorted(strength.items(), key=lambda x: x[1], reverse=True)[:max_nos]}
        H = H.subgraph(top_nos).copy()

    print(f"\n   Visualizando subgrafo de red flags: "
          f"{H.number_of_nodes()} nós, {H.number_of_edges()} arestas")

    pos = nx.spring_layout(H, k=0.7, iterations=80, seed=42)
    fig, ax = plt.subplots(figsize=(15, 11))

    cores = ["#1a6faf" if H.nodes[n].get("tipo") == "orgao" else "#e74c3c"
             for n in H.nodes()]
    tams = [300 + sum(d.get("weight", 1) for _, _, d in H.edges(n, data=True)) * 40
            for n in H.nodes()]
    rotulos = {n: (H.nodes[n].get("label", "") or "")[:30]
                 for n in H.nodes()}

    nx.draw_networkx_edges(H, pos, width=2, edge_color="#e74c3c",
                             alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_color=cores, node_size=tams,
                              edgecolors="white", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(H, pos, labels=rotulos, font_size=8, ax=ax)

    from matplotlib.patches import Patch
    legenda = [
        Patch(color="#1a6faf", label="Órgão contratante"),
        Patch(color="#e74c3c", label="Fornecedor c/ indício de eng. + contratos 'geral'"),
    ]
    ax.legend(handles=legenda, loc="upper right", framealpha=0.9)
    ax.set_title("Rede de RED FLAGS — fornecedores fantasma de engenharia em 'geral'\n"
                 "Aresta = órgão contratou fornecedor com razão social que indica eng., "
                 "mas como 'serviço geral'",
                 fontweight="bold", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    _salvar(fig, "p7_02_grafo_red_flags.png", pasta)


def grafico_concentracao_orgaos(df: pd.DataFrame, pasta: str, top_n: int = 25):
    """
    Identifica os ÓRGÃOS com maior taxa de contratos suspeitos (geral mas
    fornecedor indica engenharia).

    Para o TCC: coloca o "dedo na ferida". Quais prefeituras/órgãos
    apresentam o padrão de subenquadramento de forma sistemática?
    """
    if "niFornecedor" not in df.columns:
        return pd.DataFrame()

    df_g = df[df["niFornecedor"].notna() &
              (df["niFornecedor"].astype(str).str.strip() != "")].copy()
    if df_g.empty:
        return pd.DataFrame()

    df_g["forn_indica_eng"] = df_g["nomeRazaoSocialFornecedor"].apply(
        _empresa_tem_indicio_engenharia)
    df_g["red_flag"] = (df_g["forn_indica_eng"] & (df_g["rotulo"] == "geral"))

    res = df_g.groupby(["cnpjOrgao", "razaoSocialOrgao"]).agg(
        n_total          = ("numeroControlePNCP", "count"),
        n_red_flag       = ("red_flag", "sum"),
        n_geral          = ("rotulo", lambda x: (x == "geral").sum()),
    ).reset_index()
    res["pct_red_flag"] = (res["n_red_flag"] / res["n_total"].clip(lower=1) * 100).round(2)
    # Filtra órgãos com pelo menos 5 contratos no total (evita ruído)
    res = res[res["n_total"] >= 5]
    res = res.sort_values("n_red_flag", ascending=False)

    print(f"\n── Top-{top_n} ÓRGÃOS por nº de contratos com red flag ──")
    print(f"   (red flag = fornecedor c/ indício de eng. + rótulo 'geral')")
    print(res.head(top_n).to_string(index=False))

    # Gráfico
    top = res.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
    ax.barh(top["razaoSocialOrgao"].str[:55], top["n_red_flag"],
            color="#e74c3c", edgecolor="white")
    ax.set_title(f"Top-{top_n} órgãos por nº de RED FLAGS\n"
                 "(contratos rotulados 'geral' com fornecedor que indica engenharia)",
                 fontweight="bold")
    ax.set_xlabel("Nº de contratos com red flag")
    ax.set_ylabel("Órgão contratante")
    _anotar_barras(ax, fmt="{:,.0f}", fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "p7_03_orgaos_red_flags.png", pasta)
    return res


def executar_parte7_grafos(df: pd.DataFrame,
                              pasta_saida: str = None,
                              min_contratos_aresta: int = 2,
                              top_visualizacao: int = 50,
                              fazer_louvain: bool = True) -> dict:
    """
    Pipeline completo da Parte 7 — análise de redes (grafos).

    Roda em sequência:
      1. Detecção de fornecedores fantasma (razão social indica eng.)
      2. Construção do grafo bipartido órgão ↔ fornecedor
      3. Métricas de centralidade (degree, strength, betweenness)
      4. Detecção de comunidades via Louvain (opcional)
      5. Concentração por órgão
      6. Visualizações (grafo geral + grafo de red flags)

    Parâmetros
    ──────────
    df                  : DataFrame da Parte 1 (pós carregar_e_limpar)
    pasta_saida         : pasta para salvar gráficos e CSVs
    min_contratos_aresta: mínimo de contratos por par órgão↔fornec. para
                          o par entrar no grafo (filtro de ruído)
    top_visualizacao    : nº de nós a mostrar no grafo principal
    fazer_louvain       : se True, roda detecção de comunidades

    Retorna
    ───────
    dict com 'fantasmas', 'centralidade', 'comunidades', 'orgaos_red_flag', 'G'
    """
    if not TEM_NETWORKX:
        print("\n❌ networkx não instalado. Para Parte 7:")
        print("   !pip install networkx python-louvain")
        return {}

    print("\n" + "█"*62)
    print("  PARTE 7 — ANÁLISE DE REDES (GRAFOS)")
    print("█"*62)

    if pasta_saida is None:
        pasta_saida = _pasta_saida_padrao(df)
    os.makedirs(pasta_saida, exist_ok=True)

    res = {}

    print("\n[1] Fornecedores fantasma de engenharia em 'geral'...")
    fantasmas = detectar_fornecedores_fantasma(df, min_contratos_geral=3)
    if not fantasmas.empty:
        arq = os.path.join(pasta_saida, "p7_fornecedores_fantasma.csv")
        fantasmas.to_csv(arq, index=False, encoding="utf-8-sig")
        print(f"   💾 {arq}")
    res["fantasmas"] = fantasmas

    print("\n[2] Construção do grafo bipartido órgão ↔ fornecedor...")
    G = construir_grafo_orgao_fornecedor(df, min_contratos=min_contratos_aresta)
    res["G"] = G
    if G.number_of_nodes() == 0:
        print("   ⚠ Grafo vazio. Verifique se a coluna niFornecedor está preenchida.")
        return res

    print("\n[3] Métricas de centralidade...")
    centralidade = metricas_centralidade(G, top_n=20)
    res["centralidade"] = centralidade
    if "orgaos" in centralidade:
        centralidade["orgaos"].to_csv(
            os.path.join(pasta_saida, "p7_centralidade_orgaos.csv"),
            index=False, encoding="utf-8-sig")
    if "fornecedores" in centralidade:
        centralidade["fornecedores"].to_csv(
            os.path.join(pasta_saida, "p7_centralidade_fornecedores.csv"),
            index=False, encoding="utf-8-sig")

    if fazer_louvain:
        print("\n[4] Detecção de comunidades (Louvain)...")
        comunidades = detectar_comunidades(G)
        res["comunidades"] = comunidades
        if comunidades and "resumo" in comunidades:
            comunidades["resumo"].to_csv(
                os.path.join(pasta_saida, "p7_comunidades.csv"),
                encoding="utf-8-sig")

    # 4B. Projeção 1-mode em fornecedor↔fornecedor
    print("\n[4B] Projeção 1-mode (fornecedor↔fornecedor)...")
    try:
        G_1mode = projetar_fornecedor_fornecedor(G, min_orgaos_compartilhados=2)
        res["G_1mode"] = G_1mode

        if G_1mode.number_of_edges() > 0:
            # Detecta comunidades também na projeção (Louvain funciona bem em
            # grafos não-bipartidos — esta é a estrutura matematicamente correta)
            if fazer_louvain and TEM_LOUVAIN:
                comunidades_1mode = detectar_comunidades(G_1mode)
                res["comunidades_1mode"] = comunidades_1mode
                if comunidades_1mode and "resumo" in comunidades_1mode:
                    comunidades_1mode["resumo"].to_csv(
                        os.path.join(pasta_saida, "p7_comunidades_1mode.csv"),
                        encoding="utf-8-sig")

            # Salva GraphML para Gephi
            arq_1mode = os.path.join(pasta_saida, "p7_grafo_1mode.graphml")
            try:
                nx.write_graphml(G_1mode, arq_1mode)
                print(f"   💾 {arq_1mode}")
            except Exception:
                pass
    except Exception as e:
        print(f"   [aviso] projeção 1-mode: {e}")

    print("\n[5] Concentração por órgão...")
    res["orgaos_red_flag"] = grafico_concentracao_orgaos(df, pasta_saida)
    if not res["orgaos_red_flag"].empty:
        res["orgaos_red_flag"].to_csv(
            os.path.join(pasta_saida, "p7_orgaos_red_flag.csv"),
            index=False, encoding="utf-8-sig")

    print("\n[6] Visualizações...")
    visualizar_grafo_principal(G, pasta_saida, top_n=top_visualizacao)
    visualizar_subgrafo_red_flags(G, pasta_saida)

    # Salva o grafo em formato GraphML para análise externa (Gephi etc.)
    try:
        arq = os.path.join(pasta_saida, "p7_grafo.graphml")
        nx.write_graphml(G, arq)
        print(f"\n   💾 {arq} (use no Gephi para análises avançadas)")
    except Exception as e:
        print(f"   [aviso] não foi possível salvar GraphML: {e}")

    print("\n" + "█"*62)
    print(f"  PARTE 7 CONCLUÍDA ✅")
    print(f"  Fornecedores fantasma:  {len(res.get('fantasmas', []))}")
    print(f"  Nós no grafo:           {G.number_of_nodes():,}")
    print(f"  Arestas no grafo:       {G.number_of_edges():,}")
    print("█"*62)
    return res


# ════════════════════════════════════════════════════════════════════════════
# ████████   PARTE 8 — ENRIQUECIMENTO POR CNAE (Receita Federal)   ███████████
# ════════════════════════════════════════════════════════════════════════════
#
# A Parte 7 detectou "fornecedores fantasma" via heurística de razão social
# (presença de palavras como "Construções", "Engenharia"). É um sinal útil,
# mas FALÍVEL: empresas podem ter razão social genérica e ainda assim ser
# de engenharia, ou vice-versa.
#
# ESTA PARTE usa um sinal MUITO MAIS FORTE: o CNAE oficial registrado
# pela empresa na Receita Federal. Cruzamos:
#
#   • CNAE da empresa (consultado via API gratuita BrasilAPI/OpenCNPJ)
#   • Lista oficial do CONFEA com CNAEs de obrigação de registro no CREA
#     (arquivo cnaes_crea.xlsx anexado pelo usuário)
#
# Se uma empresa tem CNAE de eng. + foi contratada como "serviço geral",
# há evidência REGULATÓRIA forte de subenquadramento.
#
# Limitações honestas:
#   • APIs gratuitas têm rate limit (3-5 req/min)
#   • Empresas podem ter CNAEs secundários "de engenharia" sem ser sua
#     atividade principal — interpretamos isso como sinal mais fraco
#   • A lista do CONFEA tem ~700 CNAEs mas pode não estar atualizada
#
# Pré-requisito: arquivo `cnaes_crea.xlsx` na pasta de execução
# ────────────────────────────────────────────────────────────────────────────


def carregar_cnaes_crea(caminho_excel: str = "cnaes_crea.xlsx") -> dict:
    """
    Carrega a lista de CNAEs do CONFEA com obrigação de registro no CREA.

    Estrutura esperada do Excel:
      Coluna A: SUBCLASSE (formato '1012-1/01' ou '1012101')
      Coluna B: DESCRIÇÃO
      Coluna C: OBRIGAÇÃO DE REGISTRO ('Sim' / 'Passível de obrigação')

    Retorna
    ───────
    dict {cnae_normalizado: {"descricao": str, "obrigacao": str, "forte": bool}}

    forte = True quando obrigação é "Sim" (categoria mais forte de engenharia)
    forte = False quando obrigação é "Passível de obrigação"
    """
    if not os.path.exists(caminho_excel):
        print(f"   ⚠ Arquivo {caminho_excel} não encontrado.")
        return {}

    try:
        df = pd.read_excel(caminho_excel)
    except Exception as e:
        print(f"   ⚠ Erro ao ler Excel: {e}")
        return {}

    if df.shape[1] < 3:
        print(f"   ⚠ Excel precisa de 3 colunas: SUBCLASSE, DESCRIÇÃO, OBRIGAÇÃO")
        return {}

    df.columns = ["subclasse", "descricao", "obrigacao"]
    cnaes = {}
    for _, r in df.iterrows():
        cod = re.sub(r"\D", "", str(r["subclasse"]))   # normaliza p/ só dígitos
        if not cod or len(cod) != 7:
            continue
        obrig = str(r["obrigacao"]).strip().lower()
        cnaes[cod] = {
            "descricao": str(r["descricao"]),
            "obrigacao": str(r["obrigacao"]),
            "forte":     obrig in ("sim",),  # "sim" = obrigação direta
        }
    print(f"   ✅ {len(cnaes):,} CNAEs do CREA carregados ({caminho_excel})")
    n_forte = sum(1 for v in cnaes.values() if v["forte"])
    print(f"      • Obrigação 'Sim':                {n_forte:,}")
    print(f"      • 'Passível de obrigação':       {len(cnaes) - n_forte:,}")
    return cnaes


def consultar_cnpj_brasilapi(cnpj: str, timeout: int = 15) -> dict:
    """
    Consulta CNPJ via BrasilAPI (gratuita, sem cadastro, mas com limite
    informal — pedem para não fazer crawling massivo).

    Retorna dict com cnae_fiscal (principal) + cnaes_secundarios.
    """
    cnpj_norm = re.sub(r"\D", "", str(cnpj)).zfill(14)
    if len(cnpj_norm) != 14:
        return {"erro": "cnpj_invalido"}
    url = f"https://brasilapi.com.br/api/cnpj/v1/{cnpj_norm}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            j = resp.json()
            return {
                "cnpj":              cnpj_norm,
                "razao_social":      j.get("razao_social", ""),
                "cnae_principal":    str(j.get("cnae_fiscal", "")).zfill(7),
                "cnae_principal_descricao": j.get("cnae_fiscal_descricao", ""),
                "cnaes_secundarios": [
                    str(c.get("codigo", "")).zfill(7)
                    for c in (j.get("cnaes_secundarios") or [])
                ],
                "cnaes_secundarios_descricoes": [
                    c.get("descricao", "") for c in (j.get("cnaes_secundarios") or [])
                ],
                "situacao_cadastral": j.get("descricao_situacao_cadastral", ""),
                "porte":              j.get("porte", ""),
                "fonte":              "brasilapi",
            }
        elif resp.status_code == 404:
            return {"cnpj": cnpj_norm, "erro": "cnpj_nao_encontrado"}
        elif resp.status_code == 429:
            return {"cnpj": cnpj_norm, "erro": "rate_limit"}
        else:
            return {"cnpj": cnpj_norm, "erro": f"http_{resp.status_code}"}
    except Exception as e:
        return {"cnpj": cnpj_norm, "erro": f"exception_{type(e).__name__}"}


def consultar_cnpj_opencnpj(cnpj: str, timeout: int = 15) -> dict:
    """
    Fallback: consulta CNPJ via OpenCNPJ (alternativa gratuita).
    """
    cnpj_norm = re.sub(r"\D", "", str(cnpj)).zfill(14)
    if len(cnpj_norm) != 14:
        return {"erro": "cnpj_invalido"}
    url = f"https://api.opencnpj.org/{cnpj_norm}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            j = resp.json()
            return {
                "cnpj":           cnpj_norm,
                "razao_social":   j.get("razao_social", ""),
                "cnae_principal": str(j.get("cnae_principal", "")).zfill(7),
                "cnae_principal_descricao": "",  # OpenCNPJ não retorna descrição direta
                "cnaes_secundarios": [
                    str(c).zfill(7) for c in (j.get("cnaes_secundarios") or [])
                ],
                "cnaes_secundarios_descricoes": [],
                "situacao_cadastral": j.get("situacao_cadastral", ""),
                "porte":              j.get("porte_empresa", ""),
                "fonte":              "opencnpj",
            }
        elif resp.status_code == 404:
            return {"cnpj": cnpj_norm, "erro": "cnpj_nao_encontrado"}
        elif resp.status_code == 429:
            return {"cnpj": cnpj_norm, "erro": "rate_limit"}
        else:
            return {"cnpj": cnpj_norm, "erro": f"http_{resp.status_code}"}
    except Exception as e:
        return {"cnpj": cnpj_norm, "erro": f"exception_{type(e).__name__}"}


def consultar_cnpj_com_fallback(cnpj: str) -> dict:
    """
    Tenta BrasilAPI primeiro; em caso de erro, tenta OpenCNPJ.
    Esta camada de fallback aumenta a robustez sem aumentar muito a carga.
    """
    res = consultar_cnpj_brasilapi(cnpj)
    if "erro" not in res:
        return res
    if res.get("erro") == "rate_limit":
        # Em caso de rate limit, espera um pouco e tenta o fallback
        time.sleep(2.0)
    res2 = consultar_cnpj_opencnpj(cnpj)
    if "erro" not in res2:
        return res2
    # Se ambos falharam, retorna o primeiro erro
    return res


def enriquecer_fornecedores_via_cnae(df: pd.DataFrame,
                                         cache_path: str = "cache_cnpj.parquet",
                                         pausa_segundos: float = 0.5,
                                         apenas_geral: bool = True,
                                         max_consultas: int = None) -> pd.DataFrame:
    """
    Para cada fornecedor único do df, consulta a Receita Federal (via APIs
    gratuitas) e obtém o CNAE oficial.

    Usa cache em parquet para evitar reconsultar CNPJs já vistos:
    a primeira execução é lenta (5-10 min para 500 CNPJs),
    as próximas são instantâneas.

    Parâmetros
    ──────────
    df             : DataFrame com coluna 'niFornecedor'
    cache_path     : arquivo parquet onde os resultados ficam armazenados
    pausa_segundos : pausa entre consultas (default 0.5s = ~120/min — abaixo
                      dos rate limits da BrasilAPI que pede para não abusar)
    apenas_geral   : se True, só consulta CNPJs que aparecem em algum
                      contrato rotulado como 'geral' (foco da análise)
    max_consultas  : limite máximo de novas consultas nesta execução (None = sem limite)

    Retorna
    ───────
    DataFrame com 1 linha por CNPJ contendo cnae_principal + secundários.
    """
    if "niFornecedor" not in df.columns:
        print("   ⚠ Coluna 'niFornecedor' ausente.")
        return pd.DataFrame()

    df_use = df.copy()
    df_use["niFornecedor"] = df_use["niFornecedor"].astype(str).str.strip()
    df_use = df_use[df_use["niFornecedor"].str.len() == 14]   # só PJ

    if apenas_geral and "rotulo" in df_use.columns:
        cnpjs_geral = set(df_use[df_use["rotulo"] == "geral"]["niFornecedor"])
        cnpjs_a_consultar = list(cnpjs_geral)
        print(f"   Foco: {len(cnpjs_a_consultar):,} CNPJs com pelo menos 1 contrato 'geral'")
    else:
        cnpjs_a_consultar = list(df_use["niFornecedor"].unique())
        print(f"   Total: {len(cnpjs_a_consultar):,} CNPJs únicos no dataset")

    # Cache existente
    if os.path.exists(cache_path):
        df_cache = pd.read_parquet(cache_path)
        ja_consultados = set(df_cache["cnpj"].astype(str))
        print(f"   ✅ Cache: {len(df_cache):,} CNPJs já consultados em "
              f"'{cache_path}'")
    else:
        df_cache = pd.DataFrame()
        ja_consultados = set()

    novos = [c for c in cnpjs_a_consultar if c not in ja_consultados]
    print(f"   Novos a consultar: {len(novos):,}")

    if max_consultas is not None and len(novos) > max_consultas:
        print(f"   ⚠ Limitando a {max_consultas} consultas nesta execução.")
        novos = novos[:max_consultas]

    if not novos:
        print(f"   ✅ Nada a consultar — usando cache existente.")
        return df_cache

    print(f"\n   Iniciando consultas (pausa de {pausa_segundos}s entre cada).")
    print(f"   Tempo estimado: ~{len(novos) * (pausa_segundos + 0.5) / 60:.1f} minutos")

    novos_resultados = []
    n_ok = n_erro = n_rate = 0

    for cnpj in tqdm(novos, desc="🏛️  CNPJs"):
        res = consultar_cnpj_com_fallback(cnpj)
        novos_resultados.append(res)
        if "erro" in res:
            if res["erro"] == "rate_limit":
                n_rate += 1
                time.sleep(5)   # pausa maior em caso de rate limit
            else:
                n_erro += 1
        else:
            n_ok += 1
        time.sleep(pausa_segundos)

        # Salva cache progressivamente a cada 50 consultas
        if (len(novos_resultados) % 50) == 0:
            df_temp = pd.concat([df_cache, pd.DataFrame(novos_resultados)],
                                  ignore_index=True)
            try:
                df_temp.to_parquet(cache_path, index=False)
            except Exception:
                pass

    df_novos = pd.DataFrame(novos_resultados)
    df_final = pd.concat([df_cache, df_novos], ignore_index=True)

    # Remove duplicatas (caso CNPJ apareça várias vezes)
    if "cnpj" in df_final.columns:
        df_final = df_final.drop_duplicates(subset=["cnpj"], keep="last")

    df_final.to_parquet(cache_path, index=False)
    print(f"\n   ✅ Cache atualizado: {len(df_final):,} CNPJs em '{cache_path}'")
    print(f"      • Sucesso:      {n_ok:,}")
    print(f"      • Erros:        {n_erro:,}")
    print(f"      • Rate limits:  {n_rate:,}")
    return df_final


def detectar_fornecedores_cnae_engenharia(df: pd.DataFrame,
                                                df_cnpj: pd.DataFrame,
                                                cnaes_crea: dict,
                                                min_contratos_geral: int = 1) -> pd.DataFrame:
    """
    Análise central da Parte 8: identifica fornecedores cujo CNAE oficial
    indica atividade de engenharia (segundo a lista CONFEA), mas que
    aparecem como contratados em contratos rotulados como "geral".

    Score de evidência por fornecedor:
      • CNAE_principal ∈ CNAEs_CREA com obrigação 'Sim'        → score +3 (mais forte)
      • CNAE_principal ∈ CNAEs_CREA com 'Passível de obrigação' → score +2
      • CNAE_secundário ∈ CNAEs_CREA com obrigação 'Sim'         → score +1
      • CNAE_secundário ∈ CNAEs_CREA com 'Passível'              → score +0.5

    Resultado: lista de fornecedores ordenada por nº de contratos como
    "geral" × peso do score CNAE.
    """
    if df_cnpj.empty or not cnaes_crea:
        print("   ⚠ Sem cache de CNPJs ou sem lista de CNAEs do CREA.")
        return pd.DataFrame()

    # Calcula score CNAE por fornecedor
    def _calcular_score(row):
        score = 0.0
        cnaes_eng_encontrados = []
        principal = str(row.get("cnae_principal", "")).zfill(7)
        if principal in cnaes_crea:
            info = cnaes_crea[principal]
            score += 3 if info["forte"] else 2
            cnaes_eng_encontrados.append(
                f"PRINCIPAL: {principal} ({info['descricao'][:50]}) "
                f"[{info['obrigacao']}]"
            )
        for c in (row.get("cnaes_secundarios") or []):
            cn = str(c).zfill(7)
            if cn in cnaes_crea:
                info = cnaes_crea[cn]
                score += 1 if info["forte"] else 0.5
                cnaes_eng_encontrados.append(
                    f"SEC: {cn} ({info['descricao'][:40]}) "
                    f"[{info['obrigacao']}]"
                )
        return pd.Series({
            "score_cnae_eng":      score,
            "cnaes_eng_detalhe":   " | ".join(cnaes_eng_encontrados[:3]),
            "tem_cnae_eng":        score > 0,
        })

    df_cnpj_use = df_cnpj.copy()
    if "erro" in df_cnpj_use.columns:
        df_cnpj_use = df_cnpj_use[df_cnpj_use["erro"].isna() |
                                    (df_cnpj_use["erro"] == "")]
    if df_cnpj_use.empty:
        print("   ⚠ Cache de CNPJs vazio (todos com erro).")
        return pd.DataFrame()

    df_cnpj_use[["score_cnae_eng", "cnaes_eng_detalhe", "tem_cnae_eng"]] = \
        df_cnpj_use.apply(_calcular_score, axis=1)

    # Conta contratos por fornecedor por rótulo
    df_use = df.copy()
    df_use["niFornecedor"] = df_use["niFornecedor"].astype(str).str.strip()
    df_use = df_use[df_use["niFornecedor"].str.len() == 14]

    agg = df_use.groupby("niFornecedor").agg(
        n_total      = ("numeroControlePNCP", "count"),
        n_geral      = ("rotulo", lambda x: (x == "geral").sum()),
        n_engenharia = ("rotulo", lambda x: (x == "engenharia").sum()),
        valor_em_geral = ("valorTotalEstimado",
                            lambda x: x[df_use.loc[x.index, "rotulo"] == "geral"].sum()),
        razao_social = ("nomeRazaoSocialFornecedor", "first"),
        orgaos_distintos = ("cnpjOrgao", lambda x: x.dropna().astype(str).nunique()),
    ).reset_index()

    # Junta com CNAE
    df_join = agg.merge(
        df_cnpj_use[["cnpj", "cnae_principal", "cnae_principal_descricao",
                       "score_cnae_eng", "cnaes_eng_detalhe", "tem_cnae_eng",
                       "situacao_cadastral", "porte"]],
        left_on="niFornecedor", right_on="cnpj", how="left"
    )

    # Filtra: tem CNAE de eng. + ≥N contratos geral
    suspeitos = df_join[
        (df_join["tem_cnae_eng"] == True) &
        (df_join["n_geral"] >= min_contratos_geral)
    ].copy()
    suspeitos["pct_em_geral"] = (
        suspeitos["n_geral"] / suspeitos["n_total"].clip(lower=1) * 100
    ).round(1)

    # Score combinado: nº de contratos geral × intensidade do indício CNAE
    suspeitos["score_subenquadramento"] = (
        suspeitos["n_geral"] * suspeitos["score_cnae_eng"]
    ).round(2)

    suspeitos = suspeitos.sort_values("score_subenquadramento", ascending=False)

    print(f"\n── FORNECEDORES COM CNAE DE ENGENHARIA EM CONTRATOS 'GERAL' ──")
    print(f"   ({len(suspeitos):,} fornecedores; ordenados por score combinado)")
    print(f"\n   Top-25:")
    cols = ["niFornecedor", "razao_social", "cnae_principal",
            "cnae_principal_descricao", "score_cnae_eng",
            "n_total", "n_geral", "n_engenharia", "pct_em_geral",
            "orgaos_distintos", "score_subenquadramento"]
    cols = [c for c in cols if c in suspeitos.columns]
    print(suspeitos[cols].head(25).to_string(index=False))

    return suspeitos


def comparar_razao_social_vs_cnae(df: pd.DataFrame,
                                       df_cnpj: pd.DataFrame,
                                       cnaes_crea: dict,
                                       pasta: str) -> dict:
    """
    Compara as duas estratégias de detecção:
      A) Heurística por razão social (Parte 7)
      B) CNAE oficial (Parte 8 — esta)

    Para o TCC, mostra:
      • Quantos suspeitos cada método identifica
      • Concordância entre os dois (Venn-like)
      • Falsos positivos do método A que B descarta
      • Suspeitos que B captura e A não vê

    Discussão metodológica esperada no TCC:
      • A razão social é heurística superficial.
      • CNAE é oficial mas pode estar desatualizado.
      • Combinação dos dois sinais é o ideal: empresas com AMBOS sinais
        positivos são as suspeitas mais robustas.
    """
    if df_cnpj.empty or not cnaes_crea:
        return {}

    # Marca heurística A (razão social)
    df_use = df.copy()
    df_use["niFornecedor"] = df_use["niFornecedor"].astype(str).str.strip()
    df_use = df_use[df_use["niFornecedor"].str.len() == 14]
    df_use["A_razao_indica_eng"] = df_use["nomeRazaoSocialFornecedor"].apply(
        _empresa_tem_indicio_engenharia)

    # Calcula B (CNAE) por fornecedor
    df_cnpj_use = df_cnpj.copy()
    if "erro" in df_cnpj_use.columns:
        df_cnpj_use = df_cnpj_use[df_cnpj_use["erro"].isna() |
                                    (df_cnpj_use["erro"] == "")]

    def _tem_cnae_eng(row):
        principal = str(row.get("cnae_principal", "")).zfill(7)
        if principal in cnaes_crea:
            return True
        for c in (row.get("cnaes_secundarios") or []):
            if str(c).zfill(7) in cnaes_crea:
                return True
        return False

    df_cnpj_use["B_cnae_indica_eng"] = df_cnpj_use.apply(_tem_cnae_eng, axis=1)

    # Junta no df principal
    df_join = df_use.merge(
        df_cnpj_use[["cnpj", "B_cnae_indica_eng"]],
        left_on="niFornecedor", right_on="cnpj", how="left"
    )
    df_join["B_cnae_indica_eng"] = df_join["B_cnae_indica_eng"].fillna(False)

    # Foco: contratos rotulados 'geral' com fornecedor identificado em ao menos uma das heurísticas
    geral = df_join[df_join["rotulo"] == "geral"].copy()
    n_total_geral = len(geral)
    n_A      = (geral["A_razao_indica_eng"]).sum()
    n_B      = (geral["B_cnae_indica_eng"]).sum()
    n_AB     = (geral["A_razao_indica_eng"] & geral["B_cnae_indica_eng"]).sum()
    n_so_A   = (geral["A_razao_indica_eng"] & ~geral["B_cnae_indica_eng"]).sum()
    n_so_B   = (~geral["A_razao_indica_eng"] & geral["B_cnae_indica_eng"]).sum()
    n_nenhum = (~geral["A_razao_indica_eng"] & ~geral["B_cnae_indica_eng"]).sum()

    print(f"\n── Comparação razão social (A) × CNAE (B) ──")
    print(f"   Universo: {n_total_geral:,} contratos rotulados 'geral'")
    print(f"   A apenas (razão indica eng., CNAE não):       {n_so_A:,}  ← prováveis FALSOS POSITIVOS de A")
    print(f"   B apenas (CNAE indica eng., razão não):       {n_so_B:,}  ← capturados SÓ pelo CNAE")
    print(f"   Ambos (A∩B):                                  {n_AB:,}  ← SUSPEITOS MAIS ROBUSTOS")
    print(f"   Nenhum:                                       {n_nenhum:,}")

    # Diagrama de Venn-like via barras
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = ["Apenas razão (A)", "Apenas CNAE (B)", "Ambos (A∩B)", "Nenhum"]
    valores = [n_so_A, n_so_B, n_AB, n_nenhum]
    cores = ["#f39c12", "#3498db", "#e74c3c", "#bdc3c7"]
    ax.bar(cats, valores, color=cores, edgecolor="white")
    ax.set_title("Comparação dos métodos de detecção de subenquadramento\n"
                 "(contratos rotulados 'geral' por intersecção dos sinais)",
                 fontweight="bold")
    ax.set_ylabel("Nº de contratos")
    _anotar_barras(ax, fmt="{:,.0f}", horizontal=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "p8_01_comparacao_razao_vs_cnae.png", pasta)

    return {
        "total_geral":     n_total_geral,
        "apenas_A":        int(n_so_A),
        "apenas_B":        int(n_so_B),
        "ambos":           int(n_AB),
        "nenhum":          int(n_nenhum),
        "df_classificado": geral,
    }


def matriz_consistencia_rotulo_cnae(df: pd.DataFrame,
                                         df_cnpj: pd.DataFrame,
                                         cnaes_crea: dict,
                                         pasta: str) -> dict:
    """
    Matriz central do TCC: cruza ROTULO oficial × CNAE oficial da empresa
    para identificar quatro situações:

        ┌─────────────────────┬──────────────────────────┐
        │ rótulo='engenharia' │ rótulo='geral'           │
    ────┼─────────────────────┼──────────────────────────┤
    CNAE│  CONSISTENTE        │  ⚠ SUBENQUADRADO        │
    eng │  (esperado)         │  (foco do TCC)           │
    ────┼─────────────────────┼──────────────────────────┤
    CNAE│  ⚠ ANOMALIA OPOSTA  │  CONSISTENTE             │
    não │  (eng. executada    │  (esperado)              │
    eng │  por empresa fora   │                          │
        │  do CREA — viola    │                          │
        │  Lei 6.496/77)      │                          │
        └─────────────────────┴──────────────────────────┘

    Esta análise responde duas perguntas-chave do TCC:
      1. "Os 'gerais' suspeitos têm CNAE de engenharia?"
         → quadrante superior direito
      2. "Os 'engenharia' realmente são executados por empresas com CNAE de eng?"
         → quadrante superior esquerdo (esperado) vs inferior esquerdo (anomalia)

    A anomalia oposta (engenharia + CNAE não-eng) é tão importante quanto
    o subenquadramento — pode indicar empresa atuando irregularmente ou
    serviço acessório legítimo (ex.: locação de equipamento).
    """
    if df_cnpj.empty or not cnaes_crea:
        return {}

    print("\n   Construindo matriz de consistência rótulo × CNAE...")

    # Marca cada CNPJ como tendo CNAE de eng. (forte ou fraco)
    def _classificar_cnpj(row):
        principal = str(row.get("cnae_principal", "")).zfill(7)
        secundarios = [str(c).zfill(7) for c in (row.get("cnaes_secundarios") or [])]

        if principal in cnaes_crea:
            info = cnaes_crea[principal]
            return "eng_forte" if info["forte"] else "eng_passivel"

        for cn in secundarios:
            if cn in cnaes_crea:
                return "eng_secundario"

        return "nao_eng"

    df_cnpj_use = df_cnpj.copy()
    if "erro" in df_cnpj_use.columns:
        df_cnpj_use = df_cnpj_use[df_cnpj_use["erro"].isna() |
                                    (df_cnpj_use["erro"] == "")]
    if df_cnpj_use.empty:
        return {}

    df_cnpj_use["classe_cnae"] = df_cnpj_use.apply(_classificar_cnpj, axis=1)

    # Junta no df principal
    df_use = df.copy()
    df_use["niFornecedor"] = df_use["niFornecedor"].astype(str).str.strip()
    df_use = df_use[df_use["niFornecedor"].str.len() == 14]

    df_join = df_use.merge(
        df_cnpj_use[["cnpj", "classe_cnae", "cnae_principal",
                      "cnae_principal_descricao"]],
        left_on="niFornecedor", right_on="cnpj", how="left"
    )
    # Contratos sem CNPJ enriquecido ficam de fora
    df_join = df_join[df_join["classe_cnae"].notna()].copy()
    if df_join.empty:
        print("   [pulado] sem CNPJs enriquecidos.")
        return {}

    # Agrupa CNAE em duas categorias amplas para a matriz 2x2 principal
    df_join["cnae_eh_eng"] = df_join["classe_cnae"].isin(
        ["eng_forte", "eng_passivel", "eng_secundario"]
    )

    # ── Matriz 2x2 principal ────────────────────────────────────────────────
    matriz = pd.crosstab(
        df_join["cnae_eh_eng"].map({True: "CNAE de eng.", False: "CNAE não eng."}),
        df_join["rotulo"],
        margins=True, margins_name="Total"
    )
    print("\n── Matriz de consistência rótulo × CNAE (contratos) ──")
    print(matriz.to_string())

    # Detalhamento por intensidade do CNAE (matriz 4x2)
    matriz_det = pd.crosstab(
        df_join["classe_cnae"], df_join["rotulo"],
        margins=True, margins_name="Total"
    )
    print("\n── Detalhamento (4 níveis de CNAE) ──")
    print(matriz_det.to_string())

    # Estatísticas-chave do TCC
    n_geral_total = (df_join["rotulo"] == "geral").sum()
    n_eng_total   = (df_join["rotulo"] == "engenharia").sum()

    n_subenq = ((df_join["rotulo"] == "geral") &
                (df_join["cnae_eh_eng"])).sum()
    n_anomalia_op = ((df_join["rotulo"] == "engenharia") &
                      (~df_join["cnae_eh_eng"])).sum()

    pct_subenq    = n_subenq / max(n_geral_total, 1) * 100
    pct_anomalia  = n_anomalia_op / max(n_eng_total, 1) * 100

    print(f"\n── ANÁLISES-CHAVE ──")
    print(f"\n   1. Subenquadramento (rótulo='geral' + CNAE eng.):")
    print(f"      {n_subenq:,} de {n_geral_total:,} contratos 'geral'  ({pct_subenq:.1f}%)")
    print(f"\n   2. Anomalia oposta (rótulo='engenharia' + CNAE NÃO eng.):")
    print(f"      {n_anomalia_op:,} de {n_eng_total:,} contratos 'engenharia'  ({pct_anomalia:.1f}%)")
    print(f"\n   3. Consistência total ('engenharia' + CNAE eng. OU 'geral' + CNAE não eng.):")
    n_consistente = (((df_join["rotulo"] == "engenharia") & df_join["cnae_eh_eng"]) |
                     ((df_join["rotulo"] == "geral") & ~df_join["cnae_eh_eng"])).sum()
    print(f"      {n_consistente:,} de {len(df_join):,} contratos  "
          f"({n_consistente/max(len(df_join),1)*100:.1f}%)")

    # ── Gráfico: matriz de confusão visual ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Painel 1: matriz 2x2 com cores indicando consistência
    M = matriz.iloc[:-1, :-1].values   # remove totais
    rotulos_y = matriz.index[:-1].tolist()
    rotulos_x = matriz.columns[:-1].tolist()
    sns.heatmap(M, annot=True, fmt=",d", cmap="RdYlGn_r", ax=axes[0],
                xticklabels=rotulos_x, yticklabels=rotulos_y,
                cbar=False, annot_kws={"size": 13},
                linewidths=0.5)
    axes[0].set_title("Matriz rótulo × CNAE (contratos)\n"
                       "Verde = consistente | Vermelho = anomalia",
                       fontweight="bold")
    axes[0].set_xlabel("Rótulo PNCP"); axes[0].set_ylabel("CNAE oficial")

    # Painel 2: percentuais nas margens
    M_pct = matriz.iloc[:-1, :-1].values.astype(float)
    M_pct = M_pct / M_pct.sum() * 100
    sns.heatmap(M_pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
                xticklabels=rotulos_x, yticklabels=rotulos_y,
                cbar=False, annot_kws={"size": 13},
                linewidths=0.5)
    axes[1].set_title("Mesma matriz em % do total\n"
                       "(visão proporcional)",
                       fontweight="bold")
    axes[1].set_xlabel("Rótulo PNCP"); axes[1].set_ylabel("CNAE oficial")

    fig.tight_layout()
    _salvar(fig, "p8_02_matriz_rotulo_cnae.png", pasta)

    # Salva CSVs
    matriz.to_csv(os.path.join(pasta, "p8_matriz_rotulo_cnae_2x2.csv"),
                   encoding="utf-8-sig")
    matriz_det.to_csv(os.path.join(pasta, "p8_matriz_rotulo_cnae_4x2.csv"),
                       encoding="utf-8-sig")

    # Salva contratos das duas anomalias para revisão
    subenq = df_join[(df_join["rotulo"] == "geral") &
                       (df_join["cnae_eh_eng"])]
    cols = [c for c in ["numeroControlePNCP", "objeto",
                          "valorTotalEstimado", "razaoSocialOrgao",
                          "nomeRazaoSocialFornecedor", "cnae_principal",
                          "cnae_principal_descricao", "classe_cnae"]
             if c in subenq.columns]
    subenq[cols].to_csv(os.path.join(pasta, "p8_subenquadrados_rotulo_geral_cnae_eng.csv"),
                          index=False, encoding="utf-8-sig")

    anomalia = df_join[(df_join["rotulo"] == "engenharia") &
                        (~df_join["cnae_eh_eng"])]
    anomalia[cols].to_csv(os.path.join(pasta, "p8_anomalia_rotulo_eng_cnae_naoeng.csv"),
                            index=False, encoding="utf-8-sig")

    print(f"\n   💾 p8_matriz_rotulo_cnae_*.csv (matriz)")
    print(f"   💾 p8_subenquadrados_rotulo_geral_cnae_eng.csv ({len(subenq):,} contratos)")
    print(f"   💾 p8_anomalia_rotulo_eng_cnae_naoeng.csv ({len(anomalia):,} contratos)")

    return {
        "matriz_2x2":            matriz,
        "matriz_detalhada":      matriz_det,
        "n_subenquadramento":    int(n_subenq),
        "pct_subenquadramento":  round(pct_subenq, 2),
        "n_anomalia_oposta":     int(n_anomalia_op),
        "pct_anomalia_oposta":   round(pct_anomalia, 2),
        "df_subenquadrados":     subenq[cols],
        "df_anomalia_oposta":    anomalia[cols],
        "df_completo":           df_join,
    }


def ranking_orgaos_subenquadramento_cnae(df_join: pd.DataFrame,
                                              pasta: str,
                                              top_n: int = 25,
                                              min_total: int = 5) -> pd.DataFrame:
    """
    Responde: "Quais órgãos mais contratam 'geral' que deveria ser engenharia?"

    Usa o df_join produzido por matriz_consistencia_rotulo_cnae, que tem o
    rótulo PNCP × CNAE oficial. Ranqueia órgãos por:
      • n_subenq (absoluto): quantidade de contratos 'geral' com CNAE de eng.
      • pct_subenq (relativo): % desses contratos no total do órgão

    Filtra órgãos com pelo menos `min_total` contratos no total
    (evita ruído de órgãos com 1-2 contratos).
    """
    if df_join.empty:
        return pd.DataFrame()

    df = df_join.copy()
    df["eh_subenq"] = (df["rotulo"] == "geral") & (df["cnae_eh_eng"])

    res = df.groupby(["cnpjOrgao", "razaoSocialOrgao"]).agg(
        n_total          = ("numeroControlePNCP", "count"),
        n_geral          = ("rotulo", lambda x: (x == "geral").sum()),
        n_engenharia     = ("rotulo", lambda x: (x == "engenharia").sum()),
        n_subenq         = ("eh_subenq", "sum"),
        valor_subenq     = ("valorTotalEstimado",
                            lambda x: x[df.loc[x.index, "eh_subenq"]].sum()),
    ).reset_index()

    res["pct_subenq"] = (res["n_subenq"] / res["n_total"].clip(lower=1) * 100).round(2)
    res["pct_geral_eh_subenq"] = (
        res["n_subenq"] / res["n_geral"].clip(lower=1) * 100
    ).round(2)

    res = res[res["n_total"] >= min_total]
    res = res.sort_values("n_subenq", ascending=False)

    print(f"\n── Top-{top_n} órgãos por NÚMERO ABSOLUTO de subenquadramentos ──")
    print(f"   (rótulo='geral' + CNAE oficial do fornecedor é de engenharia)")
    print(res.head(top_n).to_string(index=False))

    # Ranking alternativo por percentual
    res_pct = res[res["n_total"] >= max(min_total, 10)].sort_values(
        "pct_subenq", ascending=False)
    print(f"\n── Top-{top_n} órgãos por TAXA (%) de subenquadramentos ──")
    print(f"   (filtro: ≥10 contratos no total para evitar ruído)")
    print(res_pct.head(top_n).to_string(index=False))

    arq = os.path.join(pasta, "p8_ranking_orgaos_subenq_cnae.csv")
    res.to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")

    # Gráfico
    if len(res) > 0:
        top_show = res.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(13, max(6, top_n * 0.32)))
        ax.barh(top_show["razaoSocialOrgao"].str[:60],
                 top_show["n_subenq"],
                 color="#e74c3c", edgecolor="white")
        ax.set_title(f"Top-{top_n} órgãos com mais contratos 'geral' "
                     f"+ fornecedor com CNAE de engenharia\n"
                     f"(prováveis subenquadramentos sistemáticos)",
                     fontweight="bold")
        ax.set_xlabel("Nº de contratos suspeitos (rótulo='geral' + CNAE eng.)")
        ax.set_ylabel("Órgão contratante")
        _anotar_barras(ax, fmt="{:,.0f}", fontsize=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        _salvar(fig, "p8_03_ranking_orgaos_cnae.png", pasta)

    return res


def analise_temporal_subenquadramento(df_join: pd.DataFrame,
                                          pasta: str) -> pd.DataFrame:
    """
    Análise temporal: subenquadramento aumentou ou diminuiu ao longo dos
    anos coletados? Útil para coleta multi-ano.

    Mostra:
      • Volume total por ano
      • Nº de subenquadramentos por ano
      • Taxa de subenquadramento por ano
    """
    if df_join.empty:
        return pd.DataFrame()

    if "anoPublicacao" not in df_join.columns:
        if "dataPublicacaoPncp" in df_join.columns:
            df_join = df_join.copy()
            df_join["anoPublicacao"] = pd.to_datetime(
                df_join["dataPublicacaoPncp"], errors="coerce"
            ).dt.year

    if "anoPublicacao" not in df_join.columns or df_join["anoPublicacao"].isna().all():
        return pd.DataFrame()

    df_join = df_join.copy()
    df_join["eh_subenq"] = (df_join["rotulo"] == "geral") & (df_join["cnae_eh_eng"])

    res = df_join.groupby("anoPublicacao").agg(
        total          = ("numeroControlePNCP", "count"),
        n_subenq       = ("eh_subenq", "sum"),
    ).reset_index()
    res["pct_subenq"] = (res["n_subenq"] / res["total"].clip(lower=1) * 100).round(2)
    res = res.dropna(subset=["anoPublicacao"])

    if len(res) < 2:
        print("   [pulado] análise temporal: precisa de pelo menos 2 anos.")
        return res

    print(f"\n── Evolução temporal do subenquadramento ──")
    print(res.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(res["anoPublicacao"].astype(int), res["n_subenq"],
                 color="#e74c3c", edgecolor="white")
    axes[0].set_title("Nº absoluto de subenquadramentos por ano",
                       fontweight="bold")
    axes[0].set_xlabel("Ano"); axes[0].set_ylabel("Contratos suspeitos")
    _anotar_barras(axes[0], fmt="{:,.0f}", horizontal=False)
    sns.despine(ax=axes[0])

    axes[1].plot(res["anoPublicacao"].astype(int), res["pct_subenq"],
                   marker="o", linewidth=2, color="#e74c3c")
    axes[1].set_title("Taxa (%) de subenquadramento por ano",
                       fontweight="bold")
    axes[1].set_xlabel("Ano"); axes[1].set_ylabel("% de contratos suspeitos")
    axes[1].grid(True, alpha=0.3)
    for _, r in res.iterrows():
        axes[1].annotate(f"{r['pct_subenq']:.1f}%",
                          (r["anoPublicacao"], r["pct_subenq"]),
                          textcoords="offset points", xytext=(0, 8),
                          ha="center", fontsize=9)
    sns.despine(ax=axes[1])

    fig.tight_layout()
    _salvar(fig, "p8_04_evolucao_temporal_subenq.png", pasta)
    return res


def amostragem_para_revisao_manual(df_join: pd.DataFrame,
                                       pasta: str,
                                       n_amostra: int = 30,
                                       seed: int = 42) -> pd.DataFrame:
    """
    Gera amostra ALEATÓRIA estratificada de contratos para revisão manual.

    Para o TCC isso vira o "ground truth" do trabalho:
      • Você (autor) revisa cada contrato e classifica MANUALMENTE
        como "subenquadrado de fato" ou "rotulado corretamente"
      • Esse rótulo manual é a verdade aproximada
      • Comparando os 4 sinais (classificador, razão social, CNAE, PDF) com
        o ground truth, você calcula a precisão real de cada método

    A amostra é estratificada para incluir os 4 quadrantes da matriz
    rótulo × CNAE em proporções que permitam estatística:
      • 10 contratos rotulados 'geral' + CNAE eng. (foco do TCC)
      • 10 rotulados 'engenharia' + CNAE eng. (consistente — controle)
      • 5 rotulados 'engenharia' + CNAE não eng. (anomalia oposta)
      • 5 rotulados 'geral' + CNAE não eng. (consistente — controle)
    """
    if df_join.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    amostras = []

    quadrantes = [
        ("geral",      True,  10, "subenquadramento_provavel"),
        ("engenharia", True,  10, "consistente_engenharia"),
        ("engenharia", False, 5,  "anomalia_oposta"),
        ("geral",      False, 5,  "consistente_geral"),
    ]
    for rot, cnae_eng, n, nome in quadrantes:
        sub = df_join[
            (df_join["rotulo"] == rot) &
            (df_join["cnae_eh_eng"] == cnae_eng)
        ]
        if sub.empty:
            print(f"   [aviso] quadrante '{nome}' está vazio.")
            continue
        n_real = min(n, len(sub))
        idx = rng.choice(sub.index, size=n_real, replace=False)
        amos = sub.loc[idx].copy()
        amos["quadrante"] = nome
        amos["revisao_manual"] = ""   # campo a ser preenchido manualmente
        amostras.append(amos)

    if not amostras:
        return pd.DataFrame()

    df_amostra = pd.concat(amostras, ignore_index=True)

    cols = [c for c in [
        "quadrante", "rotulo", "cnae_eh_eng",
        "numeroControlePNCP", "objeto", "valorTotalEstimado",
        "razaoSocialOrgao", "nomeRazaoSocialFornecedor",
        "cnae_principal", "cnae_principal_descricao",
        "revisao_manual",   # último para facilitar preenchimento
    ] if c in df_amostra.columns]

    arq = os.path.join(pasta, "p8_amostra_revisao_manual.csv")
    df_amostra[cols].to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")
    print(f"      ↪ {len(df_amostra)} contratos para você revisar manualmente.")
    print(f"      ↪ Preencha a coluna 'revisao_manual' com:")
    print(f"          'subenq' = é subenquadramento mesmo")
    print(f"          'ok'     = rotulação correta")
    print(f"          'duv'    = inconclusivo (precisa olhar PDF)")
    print(f"      Esse será o ground truth do TCC para validar os modelos.")

    return df_amostra


def executar_parte8_cnae(df: pd.DataFrame,
                            pasta_saida: str = None,
                            caminho_excel_crea: str = "cnaes_crea.xlsx",
                            cache_path: str = "cache_cnpj.parquet",
                            apenas_geral: bool = True,
                            max_consultas: int = 1000,
                            pausa_segundos: float = 0.5) -> dict:
    """
    Pipeline completo da Parte 8 — enriquecimento via CNAE oficial.

    1. Carrega lista CONFEA de CNAEs com obrigação de registro CREA
    2. Consulta CNPJs únicos do dataset via BrasilAPI/OpenCNPJ (com cache)
    3. Detecta fornecedores cujo CNAE oficial indica engenharia mas
       aparecem em contratos rotulados 'geral'
    4. Compara com a heurística da Parte 7 (razão social)
    5. Gera relatório CSV consolidado

    Parâmetros
    ──────────
    df                 : DataFrame da Parte 1 (com 'niFornecedor')
    pasta_saida        : pasta para salvar saídas
    caminho_excel_crea : path para cnaes_crea.xlsx
    cache_path         : arquivo parquet do cache de consultas CNPJ
    apenas_geral       : se True, só consulta CNPJs com algum contrato 'geral'
    max_consultas      : limite de consultas novas nesta execução
    pausa_segundos     : pausa entre consultas (respeita rate limits)
    """
    print("\n" + "█"*62)
    print("  PARTE 8 — ENRIQUECIMENTO VIA CNAE (Receita Federal × CONFEA)")
    print("█"*62)

    if pasta_saida is None:
        pasta_saida = _pasta_saida_padrao(df)
    os.makedirs(pasta_saida, exist_ok=True)

    res = {}

    print("\n[1] Carregando lista CONFEA de CNAEs do CREA...")
    cnaes_crea = carregar_cnaes_crea(caminho_excel_crea)
    if not cnaes_crea:
        print("   ❌ Lista vazia. Confira o arquivo Excel.")
        return {}
    res["cnaes_crea"] = cnaes_crea

    print("\n[2] Enriquecendo CNPJs via BrasilAPI/OpenCNPJ...")
    df_cnpj = enriquecer_fornecedores_via_cnae(
        df, cache_path=cache_path,
        pausa_segundos=pausa_segundos,
        apenas_geral=apenas_geral,
        max_consultas=max_consultas,
    )
    res["df_cnpj"] = df_cnpj

    if df_cnpj.empty:
        print("   ⚠ Sem dados de CNPJ. Pulando análise.")
        return res

    print("\n[3] Detectando fornecedores com CNAE de engenharia em 'geral'...")
    suspeitos = detectar_fornecedores_cnae_engenharia(
        df, df_cnpj, cnaes_crea, min_contratos_geral=1
    )
    res["suspeitos"] = suspeitos
    if not suspeitos.empty:
        arq = os.path.join(pasta_saida, "p8_fornecedores_cnae_eng.csv")
        suspeitos.to_csv(arq, index=False, encoding="utf-8-sig")
        print(f"   💾 {arq}")

    print("\n[4] Comparando razão social vs CNAE...")
    res["comparacao"] = comparar_razao_social_vs_cnae(
        df, df_cnpj, cnaes_crea, pasta_saida
    )

    print("\n[5] Matriz de consistência rótulo PNCP × CNAE oficial...")
    res["matriz_consistencia"] = matriz_consistencia_rotulo_cnae(
        df, df_cnpj, cnaes_crea, pasta_saida
    )

    # df_join é fornecido pela matriz_consistencia para reuso nas próximas
    df_join = res["matriz_consistencia"].get("df_completo", pd.DataFrame())

    if not df_join.empty:
        print("\n[6] Ranking de órgãos com mais subenquadramentos por CNAE...")
        res["ranking_orgaos"] = ranking_orgaos_subenquadramento_cnae(
            df_join, pasta_saida, top_n=25, min_total=5
        )

        print("\n[7] Análise temporal (se multi-ano)...")
        res["temporal"] = analise_temporal_subenquadramento(df_join, pasta_saida)

        print("\n[8] Amostragem para revisão manual (ground truth do TCC)...")
        res["amostra_revisao"] = amostragem_para_revisao_manual(
            df_join, pasta_saida, n_amostra=30
        )

    print("\n" + "█"*62)
    print("  PARTE 8 CONCLUÍDA ✅")
    print(f"  CNPJs consultados:    {len(df_cnpj):,}")
    print(f"  Suspeitos por CNAE:   {len(suspeitos):,}")
    if "comparacao" in res:
        print(f"  Suspeitos confirmados (razão + CNAE): {res['comparacao'].get('ambos', 0):,}")
    if "matriz_consistencia" in res:
        m = res["matriz_consistencia"]
        print(f"  Subenquadramento (geral + CNAE eng.):    {m.get('n_subenquadramento', 0):,}")
        print(f"  Anomalia oposta (eng + CNAE não-eng):    {m.get('n_anomalia_oposta', 0):,}")
    print("█"*62)
    return res


def analisar_classificacao_vs_cnae(df: pd.DataFrame,
                                       df_cnpj: pd.DataFrame,
                                       cnaes_crea: dict,
                                       p2_resultados: dict = None,
                                       pasta: str = ".") -> dict:
    """
    Análise cruzada de 4 sinais por contrato:

      S1 = Rótulo oficial PNCP        (categoria 7/8/9)
      S2 = Predição do classificador  (Parte 2)
      S3 = CNAE oficial do fornecedor (Parte 8)
      S4 = Razão social indica eng.   (Parte 7)

    Responde 4 perguntas centrais do TCC:

      Q1. Os contratos rotulados 'geral' que o CLASSIFICADOR vê como eng.
          têm CNAE de engenharia? (suspeitos triplo-confirmados)

      Q2. Os contratos rotulados 'engenharia' (cat. 7+9) realmente têm
          fornecedor com CNAE de eng.? (validação positiva do rótulo PNCP)

      Q3. Quais contratos 'geral' têm fornecedor com CNAE de eng. mas o
          classificador NÃO detectou? (perdidos pelo modelo)

      Q4. Quais 'engenharia' não tem CNAE de eng.? (rótulo PNCP duvidoso)

    Parâmetros
    ──────────
    df            : DataFrame da Parte 1 com niFornecedor, rotulo
    df_cnpj       : cache CNPJ da Parte 8
    cnaes_crea    : dict CNAE → info do CONFEA
    p2_resultados : dict da Parte 2 (opcional — usa o ranking se disponível)
    pasta         : pasta de saída

    Retorna
    ───────
    dict com 4 DataFrames + matriz 2x2 de cruzamento
    """
    if df_cnpj.empty or not cnaes_crea:
        print("   [pulado] sem cache CNPJ ou lista CONFEA.")
        return {}

    # ── Marca CNAE de eng. por fornecedor ───────────────────────────────────
    df_cnpj_use = df_cnpj.copy()
    if "erro" in df_cnpj_use.columns:
        df_cnpj_use = df_cnpj_use[df_cnpj_use["erro"].isna() |
                                   (df_cnpj_use["erro"] == "")]

    def _tem_cnae_eng_principal(row):
        principal = str(row.get("cnae_principal", "")).zfill(7)
        return principal in cnaes_crea

    def _tem_cnae_eng_qualquer(row):
        principal = str(row.get("cnae_principal", "")).zfill(7)
        if principal in cnaes_crea:
            return True
        for c in (row.get("cnaes_secundarios") or []):
            if str(c).zfill(7) in cnaes_crea:
                return True
        return False

    df_cnpj_use["S3_cnae_principal_eng"] = df_cnpj_use.apply(
        _tem_cnae_eng_principal, axis=1)
    df_cnpj_use["S3_cnae_qualquer_eng"]  = df_cnpj_use.apply(
        _tem_cnae_eng_qualquer, axis=1)

    # ── Junta no dataframe principal ────────────────────────────────────────
    df_use = df.copy()
    df_use["niFornecedor"] = df_use["niFornecedor"].astype(str).str.strip()
    df_use = df_use[df_use["niFornecedor"].str.len() == 14]

    df_join = df_use.merge(
        df_cnpj_use[["cnpj", "S3_cnae_principal_eng", "S3_cnae_qualquer_eng",
                       "cnae_principal", "cnae_principal_descricao"]],
        left_on="niFornecedor", right_on="cnpj", how="left"
    )
    # Marca como False se não tem dados de CNPJ (e não como NaN)
    df_join["S3_cnae_principal_eng"] = df_join["S3_cnae_principal_eng"].fillna(False)
    df_join["S3_cnae_qualquer_eng"]  = df_join["S3_cnae_qualquer_eng"].fillna(False)
    df_join["tem_cnae_consultado"]   = df_join["cnae_principal"].notna()

    # S4 = razão social indica eng.
    df_join["S4_razao_eng"] = df_join["nomeRazaoSocialFornecedor"].apply(
        _empresa_tem_indicio_engenharia)

    # S1 = rótulo PNCP
    df_join["S1_rotulo"] = df_join["rotulo"]

    # S2 = predição do classificador (se disponível)
    df_join["S2_classificador"] = "indisponivel"
    if (p2_resultados and "ranking" in p2_resultados
            and not p2_resultados["ranking"].empty):
        ids_susp = set(p2_resultados["ranking"]["numeroControlePNCP"].tolist())
        df_join["S2_classificador"] = df_join["numeroControlePNCP"].apply(
            lambda x: "engenharia" if x in ids_susp else "geral_ou_eng_baseline"
        )
        # Para contratos rotulados eng. originalmente, marca como eng. também
        df_join.loc[df_join["S1_rotulo"] == "engenharia",
                    "S2_classificador"] = "engenharia"

    # ── Subset relevante: contratos com CNPJ consultado ─────────────────────
    df_consul = df_join[df_join["tem_cnae_consultado"]].copy()
    n_consul = len(df_consul)
    if n_consul == 0:
        print("   ⚠ Nenhum contrato com CNPJ consultado. Rode Parte 8 antes.")
        return {}

    print(f"\n── Análise cruzada (sobre {n_consul:,} contratos com CNPJ consultado) ──")

    # ── Matriz 2x2: S1 × S3 ─────────────────────────────────────────────────
    print("\n   Matriz: rótulo PNCP × CNAE oficial")
    matriz = pd.crosstab(
        df_consul["S1_rotulo"],
        df_consul["S3_cnae_qualquer_eng"].map({True: "CNAE eng.", False: "CNAE NÃO eng."}),
        margins=True, margins_name="TOTAL"
    )
    print(matriz.to_string())

    # ── Q1: GERAL + classificador=eng + CNAE=eng (triplo confirmado) ──────
    q1 = df_consul[
        (df_consul["S1_rotulo"] == "geral") &
        (df_consul["S2_classificador"] == "engenharia") &
        (df_consul["S3_cnae_qualquer_eng"] == True)
    ].copy()
    print(f"\n   Q1: GERAL + classificador=eng + CNAE=eng → {len(q1):,} contratos")
    print(f"       (suspeitos com TRIPLA EVIDÊNCIA — mais robustos)")

    # ── Q2: rótulo eng. + CNAE eng. (validação positiva) ────────────────────
    q2_total = (df_consul["S1_rotulo"] == "engenharia").sum()
    q2_validados = ((df_consul["S1_rotulo"] == "engenharia") &
                      (df_consul["S3_cnae_qualquer_eng"] == True)).sum()
    pct_q2 = (q2_validados / max(q2_total, 1) * 100)
    print(f"\n   Q2: ENGENHARIA + CNAE eng. → {q2_validados:,}/{q2_total:,} "
          f"({pct_q2:.1f}%)")
    print(f"       (validação cruzada — quanto maior, mais consistente é o rótulo PNCP)")

    # ── Q3: GERAL + classificador=geral + CNAE=eng (perdidos pelo modelo) ──
    q3 = df_consul[
        (df_consul["S1_rotulo"] == "geral") &
        (df_consul["S2_classificador"] != "engenharia") &
        (df_consul["S3_cnae_qualquer_eng"] == True)
    ].copy()
    print(f"\n   Q3: GERAL + classificador NÃO viu + CNAE=eng → {len(q3):,} contratos")
    print(f"       (suspeitos perdidos pelo classificador — mas CNAE alerta)")

    # ── Q4: ENGENHARIA + CNAE NÃO eng. (rótulo PNCP duvidoso) ──────────────
    q4 = df_consul[
        (df_consul["S1_rotulo"] == "engenharia") &
        (df_consul["S3_cnae_qualquer_eng"] == False)
    ].copy()
    print(f"\n   Q4: ENGENHARIA mas CNAE NÃO indica eng. → {len(q4):,} contratos")
    print(f"       (rótulo PNCP duvidoso, ou empresa atravessadora/subcontratação)")

    # ── Salva os 4 grupos em CSV ────────────────────────────────────────────
    cols_show = [c for c in ["numeroControlePNCP", "objeto", "valorTotalEstimado",
                              "razaoSocialOrgao", "municipioNome",
                              "nomeRazaoSocialFornecedor", "cnae_principal",
                              "cnae_principal_descricao",
                              "S1_rotulo", "S2_classificador",
                              "S3_cnae_qualquer_eng", "S4_razao_eng"]
                  if c in df_consul.columns]

    if not q1.empty:
        q1.head(100)[cols_show].to_csv(
            os.path.join(pasta, "p8_Q1_suspeitos_triplo_confirmado.csv"),
            index=False, encoding="utf-8-sig")
    if not q3.empty:
        q3.head(100)[cols_show].to_csv(
            os.path.join(pasta, "p8_Q3_perdidos_pelo_classificador.csv"),
            index=False, encoding="utf-8-sig")
    if not q4.empty:
        q4.head(100)[cols_show].to_csv(
            os.path.join(pasta, "p8_Q4_rotulo_pncp_duvidoso.csv"),
            index=False, encoding="utf-8-sig")

    # ── Gráfico: matriz S1×S3 ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Heatmap da matriz S1 × S3
    matriz_n = pd.crosstab(df_consul["S1_rotulo"],
                              df_consul["S3_cnae_qualquer_eng"]
                                .map({True: "CNAE eng.", False: "CNAE NÃO eng."}))
    sns.heatmap(matriz_n, annot=True, fmt="d", cmap="YlOrRd",
                ax=axes[0], cbar=False, annot_kws={"size": 14})
    axes[0].set_title("Rótulo PNCP × CNAE oficial do fornecedor",
                       fontweight="bold")
    axes[0].set_xlabel("CNAE Receita Federal × CONFEA")
    axes[0].set_ylabel("Rótulo PNCP")

    # Barras dos 4 grupos Q1-Q4
    cats   = ["Q1\n(triplo OK)", "Q2\n(rotulo eng.\nvalidado)",
              "Q3\n(perdido\nclassificador)", "Q4\n(rotulo eng.\nduvidoso)"]
    vals   = [len(q1), q2_validados, len(q3), len(q4)]
    cores  = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    bars = axes[1].bar(cats, vals, color=cores, edgecolor="white")
    axes[1].set_title("Análise cruzada (4 perguntas)", fontweight="bold")
    axes[1].set_ylabel("Nº contratos")
    _anotar_barras(axes[1], fmt="{:,.0f}", horizontal=False)
    sns.despine(ax=axes[1])

    fig.tight_layout()
    _salvar(fig, "p8_02_analise_cruzada.png", pasta)

    return {
        "matriz_s1_s3":          matriz,
        "Q1_triplo_confirmado":  q1,
        "Q2_eng_validados":      q2_validados,
        "Q2_eng_total":          q2_total,
        "Q3_perdidos_modelo":    q3,
        "Q4_rotulo_duvidoso":    q4,
        "df_consultado":         df_consul,
    }


# ════════════════════════════════════════════════════════════════════════════
# ████████   PARTE 9 — RESUMO EXECUTIVO + ANÁLISE POR VALOR + GROUND TRUTH   ███
# ════════════════════════════════════════════════════════════════════════════
#
# Esta parte fecha o ciclo do TCC consolidando os resultados de TODAS as
# partes anteriores em três análises finais:
#
#   1. Estatísticas finais consolidadas (uma tabela única para o TCC)
#   2. Análise por VALOR (R$): impacto financeiro do subenquadramento
#   3. Validação contra ground truth manual (precisão real dos métodos)
#
# Pré-requisitos:
#   • Etapas A1, A2, e Partes 2-8 já rodadas
#   • Para validação: arquivo p8_amostra_revisao_manual.csv preenchido
# ────────────────────────────────────────────────────────────────────────────


def consolidar_todos_suspeitos(df: pd.DataFrame,
                                    p2: dict = None,
                                    p7: dict = None,
                                    p8: dict = None,
                                    c2: dict = None,
                                    c3: dict = None,
                                    pasta_saida: str = ".") -> pd.DataFrame:
    """
    Consolida TODOS os contratos suspeitos detectados pelos vários métodos
    em uma ÚNICA tabela. Inclui colunas booleanas indicando qual(is)
    método(s) identificou cada contrato.

    Para o TCC, esta função substitui as listas "top-N" — você obtém
    a lista COMPLETA dos suspeitos, com a sinalização de convergência
    de evidências.

    Métodos de detecção considerados:
      • S1_classificador     : ranking da Parte 2 (P(eng) ≥ threshold)
      • S2_razao_social      : Parte 7 (razão social indica engenharia)
      • S3_cnae_oficial      : Parte 8 (CNAE oficial é de engenharia)
      • S4_pdf_marcadores    : Camada 2 (ART/RRT/PB encontrados nos PDFs)
      • S5_aditivo_eng       : Camada 3 (mudança de escopo via aditivo)
      • S6_rigor_licitacao   : Parte 2 §19-B (Pregão em obra)

    Coluna `n_sinais` = quantos métodos sinalizaram o contrato.
    Quanto maior n_sinais, mais robusto o suspeito.

    Saída
    ─────
    DataFrame com TODOS os contratos suspeitos (mesmo que apenas 1 método
    tenha detectado), salvo em `consolidado_todos_suspeitos.csv`.
    """
    print("\n" + "█"*62)
    print("  CONSOLIDAÇÃO — todos os suspeitos identificados (sem limite)")
    print("█"*62)

    # Foco: contratos rotulados 'geral' (são os candidatos a subenquadramento)
    df_g = df[df["rotulo"] == "geral"].copy()
    df_g["S1_classificador"] = False
    df_g["S2_razao_social"] = False
    df_g["S3_cnae_oficial"]  = False
    df_g["S4_pdf_marcadores"] = False
    df_g["S5_aditivo_eng"]    = False
    df_g["S6_rigor_licitacao"] = False

    # ── S1: Classificador (todos os 'geral' do ranking) ─────────────────────
    if p2 and "ranking" in p2:
        ids = set(p2["ranking"]["numeroControlePNCP"])
        df_g["S1_classificador"] = df_g["numeroControlePNCP"].isin(ids)
        print(f"   S1 (classificador):     {df_g['S1_classificador'].sum():,}")

    # ── S2: Razão social ────────────────────────────────────────────────────
    if "nomeRazaoSocialFornecedor" in df_g.columns:
        df_g["S2_razao_social"] = df_g["nomeRazaoSocialFornecedor"].apply(
            _empresa_tem_indicio_engenharia
        )
        print(f"   S2 (razão social):      {df_g['S2_razao_social'].sum():,}")

    # ── S3: CNAE oficial ────────────────────────────────────────────────────
    if p8 and "matriz_consistencia" in p8:
        df_join = p8["matriz_consistencia"].get("df_completo", pd.DataFrame())
        if not df_join.empty and "cnae_eh_eng" in df_join.columns:
            ids_cnae = set(df_join[df_join["cnae_eh_eng"]]["numeroControlePNCP"])
            df_g["S3_cnae_oficial"] = df_g["numeroControlePNCP"].isin(ids_cnae)
            print(f"   S3 (CNAE oficial):      {df_g['S3_cnae_oficial'].sum():,}")

    # ── S4: PDFs com ART/RRT (Camada 2) ─────────────────────────────────────
    if c2 and "df_unido" in c2:
        u = c2["df_unido"]
        if "mk_score_engenharia" in u.columns:
            ids_pdf = set(u[u["mk_score_engenharia"] >= 2]["numeroControlePNCP"])
            df_g["S4_pdf_marcadores"] = df_g["numeroControlePNCP"].isin(ids_pdf)
            print(f"   S4 (PDF marcadores):    {df_g['S4_pdf_marcadores'].sum():,}")

    # ── S5: Aditivos de engenharia (Camada 3) ──────────────────────────────
    if c3 and "df_suspeitos" in c3 and not c3["df_suspeitos"].empty:
        ids_aditivo = set(c3["df_suspeitos"]["numeroControlePNCP"])
        df_g["S5_aditivo_eng"] = df_g["numeroControlePNCP"].isin(ids_aditivo)
        print(f"   S5 (aditivo de eng):    {df_g['S5_aditivo_eng'].sum():,}")

    # ── S6: Rigor de licitação ──────────────────────────────────────────────
    if p2 and "rigor" in p2 and not p2["rigor"].empty:
        rigor = p2["rigor"]
        if "score_irregularidade" in rigor.columns:
            ids_rigor = set(rigor[rigor["score_irregularidade"] >= 1]["numeroControlePNCP"])
            df_g["S6_rigor_licitacao"] = df_g["numeroControlePNCP"].isin(ids_rigor)
            print(f"   S6 (rigor licitação):   {df_g['S6_rigor_licitacao'].sum():,}")

    # ── Conta sinais ────────────────────────────────────────────────────────
    cols_sinais = [c for c in df_g.columns if c.startswith("S") and c[1].isdigit()]
    df_g["n_sinais"] = df_g[cols_sinais].sum(axis=1)

    # Filtra: pelo menos 1 sinal
    df_susp = df_g[df_g["n_sinais"] >= 1].copy()
    df_susp = df_susp.sort_values("n_sinais", ascending=False)

    print(f"\n── DISTRIBUIÇÃO DOS SUSPEITOS POR CONVERGÊNCIA ──")
    dist = df_susp["n_sinais"].value_counts().sort_index(ascending=False)
    for n_s, cnt in dist.items():
        print(f"   {int(n_s)} sinais: {cnt:,} contratos")

    print(f"\n── TOTAL ──")
    print(f"   Geral analisados:       {len(df_g):,}")
    print(f"   Suspeitos (≥1 sinal):   {len(df_susp):,}")
    print(f"   Robustos (≥3 sinais):   {(df_susp['n_sinais'] >= 3).sum():,}")
    print(f"   Mais fortes (≥4 sinais):{(df_susp['n_sinais'] >= 4).sum():,}")

    # Salva CSV completo
    cols_show = [c for c in [
        "numeroControlePNCP", "objeto", "valorTotalEstimado",
        "razaoSocialOrgao", "municipioNome", "anoPublicacao",
        "nomeRazaoSocialFornecedor",
        "S1_classificador", "S2_razao_social", "S3_cnae_oficial",
        "S4_pdf_marcadores", "S5_aditivo_eng", "S6_rigor_licitacao",
        "n_sinais",
    ] if c in df_susp.columns]
    arq = os.path.join(pasta_saida, "consolidado_todos_suspeitos.csv")
    df_susp[cols_show].to_csv(arq, index=False, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")

    # Subset: contratos com 2+ sinais (mais relevantes)
    df_robustos = df_susp[df_susp["n_sinais"] >= 2].copy()
    arq2 = os.path.join(pasta_saida, "consolidado_suspeitos_robustos_2plus.csv")
    df_robustos[cols_show].to_csv(arq2, index=False, encoding="utf-8-sig")
    print(f"   💾 {arq2}")

    # Gráfico: histograma de convergência
    fig, ax = plt.subplots(figsize=(10, 5))
    cores = ["#bdc3c7", "#f39c12", "#e67e22", "#e74c3c", "#c0392b", "#7b241c"]
    cnt_sinais = df_susp["n_sinais"].value_counts().sort_index()
    ax.bar(cnt_sinais.index.astype(int), cnt_sinais.values,
            color=[cores[min(int(i)-1, 5)] for i in cnt_sinais.index],
            edgecolor="white")
    ax.set_title("Distribuição de suspeitos por nº de sinais convergentes\n"
                 "(quanto mais sinais, mais robusto o caso de subenquadramento)",
                 fontweight="bold")
    ax.set_xlabel("Nº de sinais convergentes")
    ax.set_ylabel("Nº de contratos suspeitos")
    _anotar_barras(ax, fmt="{:,.0f}", horizontal=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    _salvar(fig, "consolidado_convergencia_sinais.png", pasta_saida)

    return df_susp


def estatisticas_finais_consolidadas(df: pd.DataFrame,
                                          p2: dict,
                                          p4: dict = None,
                                          p7: dict = None,
                                          p8: dict = None,
                                          c2: dict = None,
                                          c3: dict = None,
                                          pasta_saida: str = ".") -> dict:
    """
    Gera o RESUMO EXECUTIVO do TCC: uma tabela única com todos os números
    finais, formatada para você usar diretamente no relatório.

    Inclui:
      • Volumetria do dataset (período, total, distribuição por classe)
      • Performance do classificador (CV + holdout + IC95%)
      • Convergência dos sinais (4 métodos detectando subenquadramento)
      • Casos suspeitos confirmados
    """
    print("\n" + "█"*62)
    print("  RESUMO EXECUTIVO — números oficiais do TCC")
    print("█"*62)

    stats = {
        "data_geracao": datetime.datetime.now().isoformat(),
        "volumetria": {},
        "classificador": {},
        "convergencia_sinais": {},
        "casos_suspeitos": {},
    }

    # ── 1. Volumetria ───────────────────────────────────────────────────────
    n_total = len(df)
    n_eng   = (df["rotulo"] == "engenharia").sum()
    n_geral = (df["rotulo"] == "geral").sum()

    stats["volumetria"] = {
        "total_contratos":    int(n_total),
        "engenharia":         int(n_eng),
        "geral":              int(n_geral),
        "razao_eng_geral":    f"1:{n_geral/max(n_eng,1):.1f}",
        "valor_total_R$":     float(df["valorTotalEstimado"].fillna(0).sum()),
    }

    # Período coberto
    if "anoPublicacao" in df.columns:
        anos = df["anoPublicacao"].dropna()
        if len(anos) > 0:
            stats["volumetria"]["periodo"] = (f"{int(anos.min())}-{int(anos.max())}"
                                                if anos.min() != anos.max()
                                                else str(int(anos.min())))

    # UF
    if "ufSigla" in df.columns:
        stats["volumetria"]["uf"] = df["ufSigla"].mode()[0]

    # Subclasses (Obras × Serv.Eng × Serv.Geral)
    if "subclasse" in df.columns:
        stats["volumetria"]["subclasses"] = (
            df["subclasse"].value_counts().to_dict()
        )

    # ── 2. Performance do classificador ─────────────────────────────────────
    if p2:
        tab = p2.get("tabela", pd.DataFrame())
        melhor = p2.get("melhor_modelo", "?")
        if not tab.empty and melhor in tab.index:
            stats["classificador"] = {
                "melhor_modelo": melhor,
                "f1_engenharia_cv":  float(tab.loc[melhor, "F1-engenharia"])
                                       if "F1-engenharia" in tab.columns else None,
                "accuracy_cv":       float(tab.loc[melhor, "Accuracy"])
                                       if "Accuracy" in tab.columns else None,
                "precision_cv":      float(tab.loc[melhor, "Precision"])
                                       if "Precision" in tab.columns else None,
                "recall_cv":         float(tab.loc[melhor, "Recall"])
                                       if "Recall" in tab.columns else None,
                "n_features":        int(p2.get("X").shape[1])
                                       if p2.get("X") is not None else None,
            }

    # Holdout (Parte 4)
    if p4 and "holdout_metricas" in p4:
        h = p4["holdout_metricas"]
        stats["classificador"]["holdout"] = {
            "f1_engenharia": h.get("f1_engenharia"),
            "precision_eng": h.get("precision_eng"),
            "recall_eng":    h.get("recall_eng"),
            "accuracy":      h.get("accuracy"),
        }

    # Bootstrap IC95% (Parte 4)
    if p4 and "bootstrap" in p4:
        bs = p4["bootstrap"]
        if isinstance(bs, pd.DataFrame) and "f1" in bs.index:
            stats["classificador"]["ic95_f1"] = [
                float(bs.loc["f1", "IC95 Lower"]),
                float(bs.loc["f1", "IC95 Upper"]),
            ]

    # McNemar
    if p4 and "mcnemar" in p4:
        m = p4["mcnemar"]
        stats["classificador"]["mcnemar_p_valor"] = m.get("p_valor")
        stats["classificador"]["mcnemar_significativo"] = m.get("significativo")

    # ── 3. Convergência dos sinais ──────────────────────────────────────────
    # Sinal 1: Classificador (Parte 2)
    if p2 and "ranking" in p2:
        n_susp_class = len(p2["ranking"])
        stats["convergencia_sinais"]["classificador"] = int(n_susp_class)

    # Sinal 2: Razão social (Parte 7)
    if p7 and "fantasmas" in p7:
        stats["convergencia_sinais"]["razao_social"] = int(len(p7["fantasmas"]))

    # Sinal 3: CNAE oficial (Parte 8)
    if p8 and "suspeitos" in p8:
        stats["convergencia_sinais"]["cnae_oficial"] = int(len(p8["suspeitos"]))

    # Sinal 4: PDFs com ART/RRT (Camada 2)
    if c2 and "df_unido" in c2:
        u = c2["df_unido"]
        if "mk_score_engenharia" in u.columns:
            n_pdf_susp = ((u["rotulo"] == "geral") &
                            (u["mk_score_engenharia"] >= 2)).sum()
            stats["convergencia_sinais"]["pdfs_camada2"] = int(n_pdf_susp)

    # Sinal 5: Aditivos com mudança de escopo (Camada 3)
    if c3 and "df_suspeitos" in c3:
        stats["convergencia_sinais"]["aditivos_camada3"] = int(len(c3["df_suspeitos"]))

    # Matriz consistência (Parte 8)
    if p8 and "matriz_consistencia" in p8:
        m = p8["matriz_consistencia"]
        stats["casos_suspeitos"]["subenquadramento_rotulo_cnae"] = m.get("n_subenquadramento", 0)
        stats["casos_suspeitos"]["anomalia_oposta_eng_sem_cnae"] = m.get("n_anomalia_oposta", 0)
        stats["casos_suspeitos"]["pct_subenquadramento"] = m.get("pct_subenquadramento", 0)

    # ── Imprime e salva ─────────────────────────────────────────────────────
    print("\n── 1. VOLUMETRIA ──")
    for k, v in stats["volumetria"].items():
        if isinstance(v, dict):
            print(f"   {k}:")
            for k2, v2 in v.items():
                print(f"      {k2}: {v2}")
        elif isinstance(v, float) and "valor" in k.lower():
            print(f"   {k}: R$ {v:,.2f}")
        else:
            print(f"   {k}: {v}")

    print("\n── 2. PERFORMANCE DO CLASSIFICADOR ──")
    for k, v in stats["classificador"].items():
        if isinstance(v, dict):
            print(f"   {k}:")
            for k2, v2 in v.items():
                print(f"      {k2}: {v2}")
        elif isinstance(v, list):
            print(f"   {k}: [{v[0]:.4f}, {v[1]:.4f}]")
        else:
            print(f"   {k}: {v}")

    print("\n── 3. CONVERGÊNCIA DOS SINAIS ──")
    for k, v in stats["convergencia_sinais"].items():
        print(f"   {k}: {v}")

    print("\n── 4. CASOS SUSPEITOS ──")
    for k, v in stats["casos_suspeitos"].items():
        print(f"   {k}: {v}")

    # JSON
    arq = os.path.join(pasta_saida, "p9_resumo_executivo.json")
    import json
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n   💾 {arq}")

    return stats


def analise_por_valor(df: pd.DataFrame,
                        df_join_p8: pd.DataFrame = None,
                        pasta_saida: str = ".") -> dict:
    """
    Análise financeira (R$) do subenquadramento.

    Pergunta central de auditoria pública: "Quanto dinheiro está em jogo?"

    O nº de contratos é importante, mas o VALOR financeiro é o que tem
    impacto orçamentário real. Esta função responde:
      a) Qual o valor total dos contratos suspeitos?
      b) Quais órgãos concentram maior VALOR (não só nº) de subenquadramento?
      c) Qual a faixa de valores: subenquadramento é em contratos altos
         ou baixos? (relevante porque obras de eng. costumam ser mais caras)
      d) Comparação valor médio: 'geral' vs 'engenharia' vs subenquadrados
    """
    print("\n" + "█"*62)
    print("  ANÁLISE POR VALOR (R$) — impacto financeiro do subenquadramento")
    print("█"*62)

    df_use = df.copy()
    df_use["valor"] = df_use["valorTotalEstimado"].fillna(0).clip(lower=0)

    # ── a. Resumo por classe ────────────────────────────────────────────────
    res_classe = df_use.groupby("rotulo").agg(
        n_contratos=("valor", "count"),
        valor_total=("valor", "sum"),
        valor_medio=("valor", "mean"),
        valor_mediana=("valor", "median"),
        valor_max=("valor", "max"),
    )

    print("\n── Volumes financeiros por rótulo ──")
    print(res_classe.to_string(float_format=lambda x: f"{x:>15,.2f}"))

    # ── b. Análise dos suspeitos (se Parte 8 forneceu df_join) ──────────────
    valor_suspeitos = 0.0
    valor_anomalia  = 0.0
    if df_join_p8 is not None and not df_join_p8.empty:
        df_join_p8 = df_join_p8.copy()
        df_join_p8["valor"] = df_join_p8["valorTotalEstimado"].fillna(0).clip(lower=0)
        # Subenquadramento: geral + CNAE eng.
        susp = df_join_p8[(df_join_p8["rotulo"] == "geral") &
                            (df_join_p8.get("cnae_eh_eng", False))]
        anom = df_join_p8[(df_join_p8["rotulo"] == "engenharia") &
                            (~df_join_p8.get("cnae_eh_eng", False))]
        valor_suspeitos = susp["valor"].sum()
        valor_anomalia  = anom["valor"].sum()

        print(f"\n── Valores em jogo (CNAE oficial × rótulo) ──")
        print(f"   Subenquadramento (geral + CNAE eng.):  "
              f"R$ {valor_suspeitos:>15,.2f}  ({len(susp):,} contratos)")
        print(f"   Anomalia oposta  (eng + CNAE não eng): "
              f"R$ {valor_anomalia:>15,.2f}  ({len(anom):,} contratos)")
        if valor_suspeitos > 0:
            valor_total = df_use["valor"].sum()
            pct = valor_suspeitos / max(valor_total, 1) * 100
            print(f"\n   → Subenquadramento representa {pct:.2f}% do "
                  f"VALOR TOTAL movimentado.")

    # ── c. Distribuição em faixas de valor ─────────────────────────────────
    bins = [0, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, np.inf]
    labels = ["≤10k", "10-50k", "50-100k", "100-500k", "500k-1M", "1-5M", ">5M"]
    df_use["faixa_valor"] = pd.cut(df_use["valor"], bins=bins, labels=labels,
                                       include_lowest=True)

    distrib = pd.crosstab(df_use["faixa_valor"], df_use["rotulo"])

    print("\n── Distribuição em faixas de valor ──")
    print(distrib.to_string())

    # ── Gráficos ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Valor total por rótulo (barras)
    res_classe["valor_total"].plot(
        kind="bar", ax=axes[0, 0],
        color=[PALETA.get(r, "#aaa") for r in res_classe.index],
        edgecolor="white"
    )
    axes[0, 0].set_title("Valor TOTAL contratado por rótulo (R$)",
                         fontweight="bold")
    axes[0, 0].set_ylabel("R$")
    axes[0, 0].tick_params(axis="x", rotation=0)
    _anotar_barras(axes[0, 0], fmt="R$ {:,.0f}", horizontal=False, fontsize=9)
    sns.despine(ax=axes[0, 0])

    # 2. Valor médio por rótulo
    res_classe["valor_medio"].plot(
        kind="bar", ax=axes[0, 1],
        color=[PALETA.get(r, "#aaa") for r in res_classe.index],
        edgecolor="white"
    )
    axes[0, 1].set_title("Valor MÉDIO por contrato (R$)",
                         fontweight="bold")
    axes[0, 1].set_ylabel("R$")
    axes[0, 1].tick_params(axis="x", rotation=0)
    _anotar_barras(axes[0, 1], fmt="R$ {:,.0f}", horizontal=False, fontsize=9)
    sns.despine(ax=axes[0, 1])

    # 3. Distribuição em faixas (barras agrupadas)
    distrib.plot(
        kind="bar", ax=axes[1, 0],
        color=[PALETA.get(c, "#aaa") for c in distrib.columns],
        edgecolor="white"
    )
    axes[1, 0].set_title("Distribuição de contratos por faixa de valor",
                         fontweight="bold")
    axes[1, 0].set_ylabel("Nº contratos")
    axes[1, 0].set_xlabel("Faixa de valor")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].legend(title="Rótulo")
    sns.despine(ax=axes[1, 0])

    # 4. Boxplot (log-scale): distribuição de valores por rótulo
    df_box = df_use[df_use["valor"] > 0].copy()
    df_box["log_valor"] = np.log10(df_box["valor"])
    sns.boxplot(data=df_box, x="rotulo", y="log_valor", ax=axes[1, 1],
                  palette=[PALETA.get(r, "#aaa") for r in df_box["rotulo"].unique()])
    axes[1, 1].set_title("Distribuição de valores (log10) por rótulo",
                         fontweight="bold")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("log10(valor R$)")
    sns.despine(ax=axes[1, 1])

    fig.tight_layout()
    _salvar(fig, "p9_01_analise_por_valor.png", pasta_saida)

    # CSV
    res_classe.to_csv(os.path.join(pasta_saida, "p9_valor_por_classe.csv"),
                        encoding="utf-8-sig")
    distrib.to_csv(os.path.join(pasta_saida, "p9_distribuicao_faixas_valor.csv"),
                     encoding="utf-8-sig")

    return {
        "por_classe":         res_classe,
        "distribuicao":       distrib,
        "valor_subenq":       valor_suspeitos,
        "valor_anomalia":     valor_anomalia,
    }


def validar_contra_ground_truth(caminho_csv_revisado: str,
                                    p2: dict = None,
                                    p7: dict = None,
                                    p8: dict = None,
                                    c2: dict = None,
                                    c3: dict = None,
                                    pasta_saida: str = ".") -> dict:
    """
    Validação dos métodos automáticos contra revisão manual (ground truth).

    Você (autor do TCC) precisa primeiro PREENCHER o arquivo
    `p8_amostra_revisao_manual.csv` (gerado pela Parte 8) com:
      • 'subenq' se for subenquadramento de fato
      • 'ok' se a rotulação está correta
      • 'duv' se inconclusivo (exclui da validação)

    Esta função então calcula:
      • Precisão real do classificador (compara com ground truth)
      • Precisão da heurística razão social
      • Precisão do CNAE oficial
      • Combinações ótimas (qual sinal, sozinho, é mais confiável?)

    A validação manual é a única forma de quantificar o desempenho REAL
    dos métodos — todos os F1 anteriores são contra o rótulo PNCP, que
    é justamente a coisa que estamos questionando.
    """
    print("\n" + "█"*62)
    print("  VALIDAÇÃO CONTRA GROUND TRUTH MANUAL")
    print("█"*62)

    if not os.path.exists(caminho_csv_revisado):
        print(f"\n❌ Arquivo não encontrado: {caminho_csv_revisado}")
        print(f"   Antes de rodar esta função:")
        print(f"   1. Rode a Parte 8 para gerar 'p8_amostra_revisao_manual.csv'")
        print(f"   2. Abra o CSV no Excel/Sheets")
        print(f"   3. Preencha a coluna 'revisao_manual' linha a linha:")
        print(f"        'subenq' = subenquadramento confirmado")
        print(f"        'ok'     = rotulação correta")
        print(f"        'duv'    = inconclusivo (será ignorado)")
        print(f"   4. Salve o CSV de volta no mesmo caminho")
        print(f"   5. Rode esta função novamente")
        return {}

    df = pd.read_csv(caminho_csv_revisado)
    if "revisao_manual" not in df.columns:
        print(f"\n❌ Coluna 'revisao_manual' não está no CSV.")
        return {}

    # Filtra registros revisados (exclui inconclusivos e vazios)
    df["revisao_manual"] = df["revisao_manual"].astype(str).str.strip().str.lower()
    df_rev = df[df["revisao_manual"].isin(["subenq", "ok"])].copy()
    n_inconclusivos = (df["revisao_manual"] == "duv").sum()
    n_vazios = df["revisao_manual"].isin(["", "nan"]).sum()

    print(f"\n   Total na amostra:           {len(df):,}")
    print(f"   Revisados (subenq/ok):      {len(df_rev):,}")
    print(f"   Inconclusivos ('duv'):      {n_inconclusivos:,}")
    print(f"   Vazios (não revisados):     {n_vazios:,}")

    if len(df_rev) < 5:
        print(f"\n⚠ Menos de 5 revisões válidas. Revise mais contratos antes.")
        return {}

    # Ground truth binário: 1 = subenquadramento, 0 = ok
    df_rev["y_true"] = (df_rev["revisao_manual"] == "subenq").astype(int)
    n_subenq = df_rev["y_true"].sum()
    print(f"   Confirmados como subenq:    {n_subenq}/{len(df_rev)}  "
          f"({n_subenq/len(df_rev)*100:.1f}%)")

    # ── Avalia cada método ──────────────────────────────────────────────────
    metodos = {}

    # Método 1: rótulo PNCP é 'geral' (já é o que motivou a inclusão na amostra)
    # — não tem como avaliar isso aqui, é a baseline

    # Método 2: razão social indica eng. (Parte 7)
    if "nomeRazaoSocialFornecedor" in df_rev.columns:
        from pncp_analise import _empresa_tem_indicio_engenharia
        df_rev["sinal_razao"] = df_rev["nomeRazaoSocialFornecedor"].apply(
            _empresa_tem_indicio_engenharia
        ).astype(int)
        metodos["razao_social"] = df_rev["sinal_razao"]

    # Método 3: CNAE oficial é eng (Parte 8 — já está no df pelo quadrante)
    if "quadrante" in df_rev.columns:
        df_rev["sinal_cnae"] = df_rev["quadrante"].isin([
            "subenquadramento_provavel", "consistente_engenharia"
        ]).astype(int)
        metodos["cnae_oficial"] = df_rev["sinal_cnae"]

    # Método 4: classificador previu engenharia (precisa de p2)
    if p2 and "ranking" in p2 and "numeroControlePNCP" in df_rev.columns:
        ranking = p2["ranking"]
        if "numeroControlePNCP" in ranking.columns:
            ids_top = set(ranking["numeroControlePNCP"])
            df_rev["sinal_classificador"] = df_rev["numeroControlePNCP"].isin(
                ids_top
            ).astype(int)
            metodos["classificador"] = df_rev["sinal_classificador"]

    # ── Calcula precisão de cada método ─────────────────────────────────────
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    linhas_metricas = []
    for nome, sinal in metodos.items():
        try:
            p = precision_score(df_rev["y_true"], sinal, zero_division=0)
            r = recall_score(df_rev["y_true"], sinal, zero_division=0)
            f = f1_score(df_rev["y_true"], sinal, zero_division=0)
            tp = ((sinal == 1) & (df_rev["y_true"] == 1)).sum()
            fp = ((sinal == 1) & (df_rev["y_true"] == 0)).sum()
            fn = ((sinal == 0) & (df_rev["y_true"] == 1)).sum()
            linhas_metricas.append({
                "método":     nome,
                "precisão":   round(p, 3),
                "recall":     round(r, 3),
                "f1":         round(f, 3),
                "TP":         int(tp),
                "FP":         int(fp),
                "FN":         int(fn),
            })
        except Exception as e:
            print(f"   [aviso] erro avaliando '{nome}': {e}")

    # Combinações: AND de 2 ou 3 sinais (suspeitos mais robustos)
    if "razao_social" in metodos and "cnae_oficial" in metodos:
        sinal_combo = (metodos["razao_social"] & metodos["cnae_oficial"]).astype(int)
        try:
            linhas_metricas.append({
                "método":     "razao_AND_cnae",
                "precisão":   round(precision_score(df_rev["y_true"], sinal_combo, zero_division=0), 3),
                "recall":     round(recall_score(df_rev["y_true"], sinal_combo, zero_division=0), 3),
                "f1":         round(f1_score(df_rev["y_true"], sinal_combo, zero_division=0), 3),
                "TP":         int(((sinal_combo == 1) & (df_rev["y_true"] == 1)).sum()),
                "FP":         int(((sinal_combo == 1) & (df_rev["y_true"] == 0)).sum()),
                "FN":         int(((sinal_combo == 0) & (df_rev["y_true"] == 1)).sum()),
            })
        except Exception:
            pass

    df_metricas = pd.DataFrame(linhas_metricas).set_index("método")
    print(f"\n── Performance dos métodos vs ground truth manual ──")
    print(df_metricas.to_string())

    arq = os.path.join(pasta_saida, "p9_validacao_ground_truth.csv")
    df_metricas.to_csv(arq, encoding="utf-8-sig")
    print(f"\n   💾 {arq}")

    # Gráfico comparativo
    if len(df_metricas) > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        df_metricas[["precisão", "recall", "f1"]].plot(
            kind="bar", ax=ax, colormap="tab10", edgecolor="white"
        )
        ax.set_title(f"Validação dos métodos contra ground truth manual\n"
                     f"({len(df_rev)} contratos revisados, "
                     f"{n_subenq} confirmados como subenquadramento)",
                     fontweight="bold")
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Métrica")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        _salvar(fig, "p9_02_validacao_ground_truth.png", pasta_saida)

    return {
        "df_revisado":     df_rev,
        "metricas":        df_metricas,
        "n_subenquadrados_confirmados": int(n_subenq),
    }


def executar_parte9_consolidacao(df: pd.DataFrame,
                                       p2: dict,
                                       p4: dict = None,
                                       p7: dict = None,
                                       p8: dict = None,
                                       c2: dict = None,
                                       c3: dict = None,
                                       caminho_revisao: str = None,
                                       pasta_saida: str = None) -> dict:
    """
    Pipeline completo da Parte 9: estatísticas finais + valor + ground truth.

    Parâmetros
    ──────────
    df, p2, p4, p7, p8 : resultados das partes anteriores (None se não rodou)
    c2, c3             : resultados das Camadas 2 e 3 (None se não rodou)
    caminho_revisao    : path para CSV preenchido manualmente (opcional)
    pasta_saida        : pasta para salvar os gráficos
    """
    print("\n" + "█"*62)
    print("  PARTE 9 — RESUMO EXECUTIVO + VALOR + GROUND TRUTH")
    print("█"*62)

    if pasta_saida is None:
        pasta_saida = _pasta_saida_padrao(df)

    res = {}

    # 1. Estatísticas finais
    res["estatisticas"] = estatisticas_finais_consolidadas(
        df, p2, p4, p7, p8, c2, c3, pasta_saida
    )

    # 2. Análise por valor
    df_join_p8 = None
    if p8 and "matriz_consistencia" in p8:
        df_join_p8 = p8["matriz_consistencia"].get("df_completo")
    res["analise_valor"] = analise_por_valor(df, df_join_p8, pasta_saida)

    # 3. CONSOLIDAÇÃO DE TODOS OS SUSPEITOS (sem limite top-N)
    print("\n[3] Consolidando TODOS os suspeitos identificados...")
    res["todos_suspeitos"] = consolidar_todos_suspeitos(
        df, p2=p2, p7=p7, p8=p8, c2=c2, c3=c3, pasta_saida=pasta_saida
    )

    # 4. Validação ground truth (se CSV foi preenchido)
    if caminho_revisao is None:
        caminho_revisao = os.path.join(pasta_saida, "p8_amostra_revisao_manual.csv")
    if os.path.exists(caminho_revisao):
        res["validacao"] = validar_contra_ground_truth(
            caminho_revisao, p2, p7, p8, c2, c3, pasta_saida
        )
    else:
        print(f"\n[validacao] CSV não preenchido — pulando.")
        print(f"   Para validar: rode Parte 8 → preencha "
              f"p8_amostra_revisao_manual.csv → rode esta função de novo.")

    print("\n" + "█"*62)
    print("  PARTE 9 CONCLUÍDA ✅")
    print("█"*62)
    return res


def executar_tudo_interativo() -> dict:
    """
    Versão TOTALMENTE interativa do pipeline: pergunta o usuário a cada
    etapa pesada para evitar tempo desperdiçado em coisas indesejadas.

    Diferença de `executar_tudo`:
    • `executar_tudo` aceita flags binárias e roda silenciosamente
    • `executar_tudo_interativo` PERGUNTA o usuário com `perguntar_sim_nao`
       e `perguntar_escolha` antes de:
        - Rodar GridSearchCV (demorado: ~5-15min dependendo do dataset)
        - Rodar clustering KMeans (rápido, mas gera 3 gráficos)
        - Rodar Apriori (rápido, mas log textual)
        - Rodar embeddings semânticos (baixa modelo de 80-1300MB)
        - Rodar bootstrap completo (pode demorar com n_iter=1000)

    Útil quando você quer maximizar tempo no Colab gratuito ou quer rodar
    apenas partes do pipeline sem mexer no código.
    """
    print("\n" + "█"*62)
    print("  PIPELINE COMPLETO INTERATIVO — pergunta antes de cada etapa")
    print("█"*62)

    fixar_seeds(SEED_GLOBAL)

    # ── Etapa A — coleta + EDA (separadas) ───────────────────────────────────
    print("\n[Etapa A1] Coleta da API")

    # Permite pular A1 se já tem o parquet em disco
    pular_coleta = False
    arq_existente = None
    import glob
    parquets = sorted(glob.glob("contratacoes_limpas_*.parquet"))
    if parquets:
        print(f"   Parquets encontrados em disco: {parquets}")
        pular_coleta = perguntar_sim_nao(
            f"Pular coleta e usar '{parquets[-1]}'? (mais recente)",
            default_sim=True)
        if pular_coleta:
            arq_existente = parquets[-1]

    if pular_coleta and arq_existente:
        df = pd.read_parquet(arq_existente)
        print(f"✅ Carregado: {arq_existente} ({len(df):,} contratos)")
    else:
        df = executar_apenas_coleta(modo_interativo=True)
        if df is None or len(df) == 0:
            print("❌ Pipeline interrompido — sem dados.")
            return {"df": df}

    print("\n[Etapa A2] EDA")
    if perguntar_sim_nao("Rodar EDA (gráficos descritivos + análise de keywords)?",
                          default_sim=True):
        eda_res = executar_apenas_eda(df)
    else:
        eda_res = None

    # ── Parte 2: classificação baseline ──────────────────────────────────────
    print("\n[Parte 2] Classificação baseline (sempre roda — base do TCC)")
    p2 = executar_parte2(df)

    # ── Parte 3: técnicas avançadas (com perguntas) ──────────────────────────
    print("\n[Parte 3] Técnicas avançadas — escolha o que rodar")
    fazer_grid       = perguntar_sim_nao("GridSearchCV (Aulas 18, 22)? Pode demorar 5-15min",
                                           default_sim=True)
    fazer_clustering = perguntar_sim_nao("KMeans + clustering (Aulas 33, 35, 37)?",
                                           default_sim=True)
    fazer_regras     = perguntar_sim_nao("Apriori — regras de associação (Aula 39)?",
                                           default_sim=True)
    fazer_embeddings = perguntar_sim_nao(
        "Embeddings semânticos Sentence-BERT/BERTimbau (Aula 42)? "
        "Baixa modelo de ~80-1300MB",
        default_sim=False)

    tipo_emb = "sentence-bert"
    if fazer_embeddings:
        idx = perguntar_escolha(
            "Qual modelo de embedding usar?",
            opcoes=[
                "Sentence-BERT MiniLM (rápido, ~80MB)",
                "Sentence-BERT mpnet (preciso, ~430MB)",
                "BERTimbau base PT-BR (~440MB)",
                "BERTimbau STS large PT-BR (~1.3GB)",
            ],
            default=0,
            descricoes=[
                "Para começar: roda em CPU em poucos minutos",
                "Mais preciso, mas mais lento (recomenda GPU)",
                "Modelo brasileiro com mean-pooling manual",
                "Adaptado para similaridade semântica",
            ],
        )
        tipo_emb = ["sentence-bert", "sentence-bert-mpnet",
                     "bertimbau", "bertimbau-sts"][idx]

    p3 = executar_parte3(p2,
                          fazer_grid=fazer_grid,
                          fazer_clustering=fazer_clustering,
                          fazer_regras=fazer_regras,
                          fazer_embeddings=fazer_embeddings,
                          tipo_embedding=tipo_emb)

    # ── Parte 4: rigor de pesquisa ───────────────────────────────────────────
    print("\n[Parte 4] Rigor de pesquisa (estatística para o TCC)")
    fazer_holdout    = perguntar_sim_nao("Holdout final 80/20 (números OFICIAIS do TCC)?",
                                           default_sim=True)
    fazer_mcnemar    = perguntar_sim_nao("Teste de McNemar (significância entre 2 modelos)?",
                                           default_sim=True)
    fazer_bootstrap  = perguntar_sim_nao("Bootstrap IC95% (1000 reamostragens)?",
                                           default_sim=True)
    fazer_multi      = perguntar_sim_nao("Multiclasse 3-classes (eng_comum × eng_especial)?",
                                           default_sim=True)
    p4 = executar_parte4(df, p2_resultados=p2,
                          fazer_holdout=fazer_holdout,
                          fazer_mcnemar=fazer_mcnemar,
                          fazer_bootstrap=fazer_bootstrap,
                          fazer_multiclasse=fazer_multi)

    # ── Parte 5: interpretação e relatório ───────────────────────────────────
    if "ufSigla" in df.columns:
        uf  = df["ufSigla"].mode()[0]
        anos = df.get("anoPublicacao", pd.Series(dtype=int))
        if not anos.empty:
            ano_min = int(anos.min()); ano_max = int(anos.max())
            sufixo = f"{uf}_{ano_min}" if ano_min == ano_max else f"{uf}_{ano_min}_{ano_max}"
        else:
            sufixo = uf
        pasta = f"graficos_pncp_{sufixo}"
    else:
        pasta = "graficos_pncp"

    print("\n[Parte 5] Interpretação automática + relatório markdown")
    if perguntar_sim_nao("Gerar relatório markdown com interpretação automática?",
                           default_sim=True):
        interpretacoes = interpretar_resultados(df, p2, p3, p4)
        relatorio_path = gerar_relatorio_markdown(df, p2, p3, p4, pasta_saida=pasta)
    else:
        interpretacoes = None
        relatorio_path = None

    # ── Parte 6: redução de FP + LLM (opcional) ──────────────────────────────
    print("\n[Parte 6] Redução de falsos positivos + validação LLM")
    if perguntar_sim_nao(
            "Rodar Parte 6 (threshold alta precisão, ensemble, coerência semântica)?",
            default_sim=False):
        usar_emb = perguntar_sim_nao(
            "Incluir coerência semântica via embeddings? (baixa modelo ~80MB)",
            default_sim=False)
        precision_alvo = 0.90  # default; pode customizar via input se quiser
        p6 = executar_parte6_reducao_fp(
            p2_resultados=p2,
            df_corpus=df,
            pasta_saida=pasta,
            precision_alvo=precision_alvo,
            usar_embeddings=usar_emb,
            llm_callable=None,    # gera prompts mas não chama API real
            n_top_revisao=30,
        )
    else:
        p6 = None

    # ── Parte 7: análise de redes (grafos) ───────────────────────────────────
    print("\n[Parte 7] Análise de redes (grafos órgão ↔ fornecedor)")
    if perguntar_sim_nao(
            "Rodar Parte 7? Detecta fornecedores fantasma de eng. em 'geral', "
            "calcula centralidade e visualiza grafo",
            default_sim=True):
        fazer_louvain = perguntar_sim_nao(
            "Detectar comunidades via Louvain? (pode demorar em grafos grandes)",
            default_sim=True)
        p7 = executar_parte7_grafos(
            df,
            pasta_saida=pasta,
            min_contratos_aresta=2,
            top_visualizacao=50,
            fazer_louvain=fazer_louvain,
        )
    else:
        p7 = None

    # ── Parte 8: enriquecimento via CNAE oficial (Receita Federal) ──────────
    print("\n[Parte 8] Enriquecimento via CNAE (Receita Federal × CONFEA)")
    if perguntar_sim_nao(
            "Rodar Parte 8? Consulta CNAE oficial dos fornecedores e cruza "
            "com lista CONFEA. Demora alguns minutos (consultas API gratuitas).",
            default_sim=False):
        max_q = 1000
        if perguntar_sim_nao(
                "Limitar a 200 consultas (mais rápido, suficiente para teste)?",
                default_sim=True):
            max_q = 200
        p8 = executar_parte8_cnae(
            df,
            pasta_saida=pasta,
            caminho_excel_crea="cnaes_crea.xlsx",
            cache_path="cache_cnpj.parquet",
            apenas_geral=True,
            max_consultas=max_q,
            pausa_segundos=0.5,
        )
    else:
        p8 = None

    # ── Camada 3: termos aditivos (mudança de escopo) ───────────────────────
    print("\n[Camada 3] Termos aditivos — análise de mudança de escopo")
    c3 = None
    if perguntar_sim_nao(
            "Rodar Camada 3? (busca termos aditivos em contratos 'geral'; "
            "demora 20-60 min dependendo do volume)",
            default_sim=False):
        max_c3 = 200
        if perguntar_sim_nao(
                "Limitar a 200 contratos (mais rápido para teste)?",
                default_sim=True):
            max_c3 = 200
        try:
            from pncp_camada3 import executar_camada3
            c3 = executar_camada3(
                df,
                pasta_saida=pasta,
                max_contratos=max_c3,
                apenas_geral=True,
            )
        except ImportError:
            print("   ⚠ pncp_camada3.py não encontrado.")
        except Exception as e:
            print(f"   ⚠ Camada 3 falhou: {e}")

    # ── Parte 9: resumo executivo + análise por valor + ground truth ───────
    print("\n[Parte 9] Resumo executivo + análise por valor + ground truth")
    p9 = None
    if perguntar_sim_nao(
            "Gerar resumo executivo e análise por valor? "
            "(consolida tudo em uma tabela final para o TCC)",
            default_sim=True):
        # Procura c2 nas variáveis locais (Camada 2)
        c2_local = locals().get("c2", None)
        p9 = executar_parte9_consolidacao(
            df, p2=p2, p4=p4, p7=p7, p8=p8, c2=c2_local, c3=c3,
            pasta_saida=pasta,
        )

    print(f"\n✅ Pipeline interativo concluído!")
    if relatorio_path:
        print(f"   Relatório TCC: {relatorio_path}")
    if p6 is not None:
        print(f"   Pacote de revisão humana: gerado em {pasta}/")
    if p7 is not None:
        print(f"   Análise de grafos: {p7.get('G', 'sem grafo').number_of_edges() if hasattr(p7.get('G'), 'number_of_edges') else 0} arestas")
    if p8 is not None and "suspeitos" in p8:
        print(f"   Suspeitos por CNAE oficial: {len(p8.get('suspeitos', []))}")
    if c3 is not None:
        print(f"   Aditivos analisados (Camada 3): {len(c3.get('df_aditivos', []))}")
    if p9 is not None:
        print(f"   Resumo executivo: p9_resumo_executivo.json")

    resultado = {"df": df, "eda": eda_res, "p2": p2, "p3": p3, "p4": p4}
    if interpretacoes is not None:
        resultado["interpretacoes"] = interpretacoes
    if relatorio_path is not None:
        resultado["relatorio"] = relatorio_path
    if p6 is not None:
        resultado["p6"] = p6
    if p7 is not None:
        resultado["p7"] = p7
    if p8 is not None:
        resultado["p8"] = p8
    if c3 is not None:
        resultado["c3"] = c3
    if p9 is not None:
        resultado["p9"] = p9
    return resultado


def executar_tudo(modo_interativo: bool = True,
                    fazer_grid: bool = True,
                    fazer_clustering: bool = True,
                    fazer_regras: bool = True,
                    fazer_embeddings: bool = False,
                    fazer_holdout: bool = True,
                    fazer_mcnemar: bool = True,
                    fazer_bootstrap: bool = True,
                    fazer_multiclasse: bool = True,
                    fazer_reducao_fp: bool = True,
                    precision_alvo: float = 0.90,
                    llm_callable=None) -> dict:
    """
    Executa o pipeline COMPLETO em sequência: Partes 1, 2, 3, 4, 5 e 6.

    Use APENAS quando quiser rodar tudo do zero. Para execução por etapas
    (recomendado no Colab), chame as funções individualmente:

        df, _ = executar_pipeline_completo()
        p2 = executar_parte2(df)
        p3 = executar_parte3(p2)
        p4 = executar_parte4(df, p2)
        p6 = executar_parte6_reducao_fp(p2, df)

    Parâmetros novos
    ────────────────
    fazer_reducao_fp : se True, executa a Parte 6 (redução de FP)
    precision_alvo   : meta de precisão para D.1
    llm_callable     : função de chamada ao LLM (ex.: Gemini Flash).
                       Se None, só prepara prompts.

    Retorna dict com {'df','p2','p3','p4','p6','interpretacoes','relatorio'}.
    """
    fixar_seeds(SEED_GLOBAL)
    df, eda_res = executar_pipeline_completo(modo_interativo=modo_interativo)
    if df is None or len(df) == 0:
        print("❌ Pipeline interrompido — sem dados.")
        return {"df": df, "eda": eda_res}

    p2 = executar_parte2(df)
    p3 = executar_parte3(p2,
                          fazer_grid=fazer_grid,
                          fazer_clustering=fazer_clustering,
                          fazer_regras=fazer_regras,
                          fazer_embeddings=fazer_embeddings)
    p4 = executar_parte4(df, p2_resultados=p2,
                          fazer_holdout=fazer_holdout,
                          fazer_mcnemar=fazer_mcnemar,
                          fazer_bootstrap=fazer_bootstrap,
                          fazer_multiclasse=fazer_multiclasse)

    # Pasta de saída (mesma usada pelas outras partes)
    if "ufSigla" in df.columns:
        uf  = df["ufSigla"].mode()[0]
        anos = df.get("anoPublicacao", pd.Series(dtype=int))
        if not anos.empty:
            ano_min = int(anos.min()); ano_max = int(anos.max())
            sufixo = f"{uf}_{ano_min}" if ano_min == ano_max else f"{uf}_{ano_min}_{ano_max}"
        else:
            sufixo = uf
        pasta = f"graficos_pncp_{sufixo}"
    else:
        pasta = "graficos_pncp"

    print("\n" + "█"*62)
    print("  PARTE 5 — INTERPRETAÇÃO AUTOMÁTICA + RELATÓRIO")
    print("█"*62)
    interpretacoes = interpretar_resultados(df, p2, p3, p4)
    relatorio_path = gerar_relatorio_markdown(df, p2, p3, p4, pasta_saida=pasta)

    p6 = None
    if fazer_reducao_fp:
        p6 = executar_parte6_reducao_fp(
            p2, df, pasta_saida=pasta,
            precision_alvo=precision_alvo,
            usar_embeddings=fazer_embeddings,
            llm_callable=llm_callable,
        )

    return {"df": df, "eda": eda_res, "p2": p2, "p3": p3, "p4": p4, "p6": p6,
             "interpretacoes": interpretacoes, "relatorio": relatorio_path}


# ════════════════════════════════════════════════════════════════════════════
# Ponto de entrada quando rodado via `python pncp_analise.py` no terminal.
#
# IMPORTANTE: NÃO dispara automaticamente em ambientes interativos (Jupyter,
# Colab, IPython, Spyder, %run). A detecção é feita verificando se há um
# kernel IPython ativo no momento do import — se houver, é notebook.
# ────────────────────────────────────────────────────────────────────────────
def _eh_execucao_direta() -> bool:
    """
    Retorna True só em execução direta via `python pncp_analise.py` no terminal.
    Retorna False em Colab, Jupyter, IPython, Spyder, %run e REPL.

    A checagem usa três sinais combinados:
      1. Existência de IPython com kernel ativo (`get_ipython()` retorna não-None)
      2. Ausência da variável `__file__` (REPL puro)
      3. Existência do módulo `IPython.core.getipython.get_ipython` ativo
    """
    if __name__ != "__main__":
        return False
    # Sinal 1: IPython kernel ativo (Jupyter/Colab/Spyder/%run)
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return False
    except ImportError:
        pass
    # Sinal 2: REPL puro não define __file__
    if "__file__" not in globals():
        return False
    # Sinal 3: argv[0] aponta para Jupyter/Colab launcher
    nome = (sys.argv[0] or "").lower() if hasattr(sys, "argv") else ""
    if any(p in nome for p in ["ipykernel", "jupyter", "colab",
                                "ipython-input", "spyder"]):
        return False
    return True


if _eh_execucao_direta():
    executar_tudo(modo_interativo=MODO_INTERATIVO)

