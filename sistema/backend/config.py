"""Configuração central. Sistema AUTOSSUFICIENTE (não depende do notebook):
importa uma base inicial já baixada e, depois, ingere o PNCP periodicamente."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_DIR = Path(os.environ.get("PNCP_SISTEMA_DIR", BASE_DIR / "dados_sistema"))
DADOS_DIR.mkdir(parents=True, exist_ok=True)
(DADOS_DIR / "pdfs").mkdir(exist_ok=True)

DB_PATH = DADOS_DIR / "monitor.db"
MODELO = DADOS_DIR / "classificador.joblib"      # vetorizador + modelo do sistema
CACHE_PDFS = DADOS_DIR / "pdfs"
FRONTEND_DIR = BASE_DIR / "frontend"

# Base inicial já baixada (parquet/csv com colunas objeto, categoria[, orgao,
# valor, id, uf]). categoria ∈ {engenharia, obras, geral}. Se ausente, usa demo.
IMPORT_FILE = os.environ.get(
    "PNCP_IMPORT_FILE",
    str(Path(os.environ.get("PNCP_TCC_DIR", BASE_DIR)) / "dados" / "coleta"
        / "contratos.parquet"))

# APIs públicas do PNCP.
PNCP_CONSULTA = "https://pncp.gov.br/api/consulta/v1/contratos"
PNCP_PNCP = "https://pncp.gov.br/api/pncp/v1"

# categoriaProcessoId do PNCP → rótulo do sistema (ajuste conforme sua coleta).
CATEG_ENGENHARIA = {2, 4}     # obras, serviços de engenharia (exemplos)
CATEG_GERAL = {8}             # serviços (onde mora o subenquadramento)

CONFIG_PADRAO = {
    # classificação / aprendizado
    "limiar": "0.6",
    "retrain_modo": "por_feedbacks",       # por_feedbacks | por_tempo
    "retrain_n_feedbacks": "20",
    "retrain_intervalo_min": "1440",
    "peso_feedback": "5",                   # peso do rótulo humano no re-treino
    # ingestão contínua (mensal por padrão)
    "ingest_ativo": "0",
    "ingest_intervalo_dias": "30",
    "ingest_uf": "SP",
    # LLM (veredito + análise dos PDFs) — opcional
    "llm_ativo": "0",
    "llm_base_url": "http://127.0.0.1:11434",  # Ollama local
    "llm_modelo": "qwen2.5:7b",
    "llm_auto_veredito": "1",               # rodar veredito ao pontuar
    # rito
    "rito_max_docs": "3",
}
