"""Configuração central e caminhos. Tudo relativo à pasta de dados do sistema,
definível por variável de ambiente PNCP_SISTEMA_DIR (default ./dados_sistema)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_DIR = Path(os.environ.get("PNCP_SISTEMA_DIR", BASE_DIR / "dados_sistema"))
DADOS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DADOS_DIR / "monitor.db"
MODELO_ONLINE = DADOS_DIR / "modelo_online.joblib"   # SGDClassifier incremental
FRONTEND_DIR = BASE_DIR / "frontend"

# Opcional: resultados do notebook para semear a fila (ranking de suspeitos).
# Se ausente, o sistema sobe em modo demonstração com dados de exemplo.
RANKING_CSV = Path(os.environ.get(
    "PNCP_RANKING_CSV",
    Path(os.environ.get("PNCP_TCC_DIR", BASE_DIR)) / "resultados_pesquisa"
    / "07_ranking_suspeitos.csv"))

# Endpoint público de consulta do PNCP (ingestão contínua de novos contratos).
PNCP_API = "https://pncp.gov.br/api/consulta/v1/contratos"

# Configurações padrão (persistidas na tabela `config`, editáveis pela UI).
CONFIG_PADRAO = {
    "retrain_modo": "por_feedbacks",       # "por_feedbacks" | "por_tempo"
    "retrain_n_feedbacks": "25",           # re-treina a cada N feedbacks novos
    "retrain_intervalo_min": "1440",       # ou a cada X minutos (modo por_tempo)
    "peso_feedback": "3",                   # peso amostral do rótulo humano
    "limiar": "0.65",                       # corte de suspeita
    "ingest_ativo": "0",                    # 0/1 — buscar novos contratos no PNCP
    "ingest_intervalo_min": "1440",        # frequência da ingestão
    "ingest_uf": "SP",
    "blend_peso_online": "0.5",            # peso do modelo online no score final
}
