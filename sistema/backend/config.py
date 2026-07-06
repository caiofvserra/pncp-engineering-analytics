"""Configuração central e caminhos. Tudo relativo à pasta de dados do sistema,
definível por PNCP_SISTEMA_DIR (default ./dados_sistema)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_DIR = Path(os.environ.get("PNCP_SISTEMA_DIR", BASE_DIR / "dados_sistema"))
DADOS_DIR.mkdir(parents=True, exist_ok=True)
(DADOS_DIR / "pdfs").mkdir(exist_ok=True)   # cache de PDFs do rito

DB_PATH = DADOS_DIR / "monitor.db"
MODELO_ONLINE = DADOS_DIR / "modelo_online.joblib"
CACHE_PDFS = DADOS_DIR / "pdfs"
FRONTEND_DIR = BASE_DIR / "frontend"

# Saída do notebook (ranking de suspeitos) usada para semear o sistema.
RANKING_CSV = Path(os.environ.get(
    "PNCP_RANKING_CSV",
    Path(os.environ.get("PNCP_TCC_DIR", BASE_DIR)) / "resultados_pesquisa"
    / "07_ranking_suspeitos.csv"))

# APIs públicas do PNCP.
PNCP_CONSULTA = "https://pncp.gov.br/api/consulta/v1/contratos"   # ingestão contínua
PNCP_PNCP = "https://pncp.gov.br/api/pncp/v1"                     # detalhe/arquivos (rito)

CONFIG_PADRAO = {
    "retrain_modo": "por_feedbacks",       # por_feedbacks | por_tempo
    "retrain_n_feedbacks": "20",
    "retrain_intervalo_min": "1440",
    "peso_feedback": "3",
    "limiar": "0.65",
    "ingest_ativo": "0",
    "ingest_intervalo_min": "1440",
    "ingest_uf": "SP",
    "blend_peso_online": "0.5",
    "rito_max_docs": "3",                  # PDFs por contrato no rito
}
