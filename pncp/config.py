"""
Constantes e paths padrão do pipeline.

Tudo que precisar mudar entre ambientes (local, Colab, Drive) vive aqui.
Os outros módulos importam `config.PASTA_DADOS` e derivam seus subdiretórios.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# No Colab, geralmente: /content/drive/MyDrive/PNCP_TCC/dados
# Localmente: ./dados
PASTA_DADOS = Path(os.environ.get("PNCP_PASTA_DADOS", "dados"))


def caminho(*partes) -> Path:
    """Resolve um caminho relativo a PASTA_DADOS, criando o diretório-pai."""
    p = PASTA_DADOS.joinpath(*partes)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── Categorias PNCP (Lei 14.133/2021) ────────────────────────────────────────
CAT_OBRAS = 7          # Obras
CAT_SERV_GERAIS = 8    # Serviços gerais  ← pode esconder engenharia
CAT_SERV_ENG = 9       # Serviços de engenharia

# Apenas estas três categorias entram na análise de subenquadramento
CATEGORIAS_INTERESSE = (CAT_OBRAS, CAT_SERV_GERAIS, CAT_SERV_ENG)


def rotular(cat_id):
    """Mapeia categoriaProcessoId → rótulo legível."""
    if cat_id == CAT_OBRAS:
        return "obras"
    if cat_id == CAT_SERV_ENG:
        return "engenharia"
    if cat_id == CAT_SERV_GERAIS:
        return "geral"
    return "outro"


# ── API PNCP ─────────────────────────────────────────────────────────────────
# Há DUAS APIs:
#   1. consulta (https://pncp.gov.br/api/consulta) — lista contratos/contratações
#   2. integração (https://pncp.gov.br/api/pncp) — documentos (PDFs), aditivos
API_BASE = "https://pncp.gov.br/api/consulta"
API_INTEGRACAO = "https://pncp.gov.br/api/pncp"
PAUSA_PAGINA = 0.3      # rate limit cordial
TIMEOUT_HTTP = 30
TENTATIVAS_HTTP = 4

# ── TF-IDF / texto ───────────────────────────────────────────────────────────
# Defaults pensados para 1M+ contratos em Colab Free (12GB RAM):
#   max_features=30k cobre os termos importantes sem inflar matriz
#   min_df=10 elimina ruído raro (palavras em <10 docs)
TFIDF_MAX_FEATURES = 30_000
TFIDF_MIN_DF = 10
TFIDF_NGRAM = (1, 2)

# Termos usados para sinalizar engenharia no texto livre
# (nomes de áreas, materiais, equipamentos, normas, profissões CREA)
TERMOS_ENGENHARIA = (
    "obra", "construcao", "reforma", "edificacao", "pavimentacao", "asfalto",
    "drenagem", "terraplenagem", "fundacao", "estrutura", "concreto", "armado",
    "alvenaria", "telhado", "cobertura", "instalacao eletrica", "hidraulica",
    "saneamento", "esgoto", "agua", "iluminacao publica", "ponte", "viaduto",
    "calcada", "meio fio", "guia", "sarjeta", "muro", "passarela", "rampa",
    "engenharia civil", "engenheiro", "art ", "rrt ", "crea", "cau",
    "memorial descritivo", "projeto executivo", "as built", "abnt nbr",
)

# ── BERTimbau / embeddings ───────────────────────────────────────────────────
MODELO_BERTIMBAU = "neuralmind/bert-base-portuguese-cased"
MODELO_SBERT = "sentence-transformers/distiluse-base-multilingual-cased-v1"
EMB_BATCH = 64
EMB_DTYPE_DISCO = "float16"   # economiza ~50% no parquet/.npy


# ── Classificação ────────────────────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.2
N_BOOTSTRAP = 1000

# ── Camada 2 (PDFs) ──────────────────────────────────────────────────────────
PDFS_TIMEOUT = 60
PDFS_MAX_PAGINAS = 30          # corta documentos enormes
PDFS_USAR_OCR = True

# ── Subpastas de saída (criadas sob demanda) ─────────────────────────────────
SUB_COLETA = "coleta"
SUB_EDA = "eda"
SUB_P2 = "classificacao"
SUB_P3 = "avancado"
SUB_EMB = "embeddings"
SUB_C2 = "pdfs"
SUB_C3 = "aditivos"
SUB_P7 = "grafos"
SUB_P8 = "cnae"
SUB_P9 = "relatorio"
