"""
Pacote PNCP — análise de subenquadramento de contratos de engenharia.

TCC MBA IA & Big Data (ICMC/USP). Identifica contratos rotulados como
"serviços gerais" (categoriaProcessoId=8) que deveriam ser "engenharia/obras"
(7 ou 9), violando a Lei 14.133/2021.

Cada subpacote escreve seus artefatos em disco (parquet/json/npz) para que
o pipeline sobreviva a reinício de kernel no Colab.

Uso típico:

    import pncp
    pncp.config.PASTA_DADOS = "/content/drive/MyDrive/PNCP_TCC/dados"

    pncp.coleta.coletar(uf="SP", anos=range(2023, 2026))
    pncp.eda.executar()
    pncp.texto.preprocessar(...)
    pncp.texto.construir_tfidf(...)
    pncp.classificacao.executar()
    pncp.avancado.executar()
    pncp.grafos.executar()
    pncp.cnae.executar()
    pncp.pdfs.executar()
    pncp.aditivos.executar()
    pncp.relatorio.gerar()

Os submódulos pesados (embeddings, pdfs) são lazy — só importam torch /
PyMuPDF quando você de fato chama `pncp.embeddings.executar()`.
"""

import importlib

__version__ = "1.0.0"

_LAZY = {
    "config", "io_disco", "ram",
    "coleta", "texto", "eda", "classificacao", "avancado",
    "embeddings", "grafos", "cnae", "pdfs", "aditivos", "relatorio",
    "spark_extras",
}


def __getattr__(nome):
    """Lazy import: só carrega o submódulo quando ele é acessado."""
    if nome in _LAZY:
        modulo = importlib.import_module(f".{nome}", __name__)
        globals()[nome] = modulo
        return modulo
    raise AttributeError(f"módulo 'pncp' não tem atributo '{nome}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY))


# ── Helpers Colab ────────────────────────────────────────────────────────────
def montar_drive(pasta="/content/drive/MyDrive/PNCP_TCC"):
    """Monta o Google Drive e configura PASTA_DADOS para `pasta/dados`."""
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    from pncp import config
    from pathlib import Path
    config.PASTA_DADOS = Path(pasta) / "dados"
    config.PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    print(f"[drive] PASTA_DADOS = {config.PASTA_DADOS}")
    return config.PASTA_DADOS


def keep_alive():
    """Injeta JS no Colab para evitar desconexão por inatividade (~12h)."""
    try:
        from IPython.display import display, Javascript
        display(Javascript("""
            function keepAlive() {
              const btn = document.querySelector("colab-connect-button");
              if (btn) btn.click();
            }
            setInterval(keepAlive, 60000);
        """))
        print("[colab] keep-alive ativado")
    except Exception:
        pass
