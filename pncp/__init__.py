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
from pathlib import Path

__version__ = "1.0.0"

_LAZY = {
    "config", "io_disco", "ram",
    "coleta", "texto", "eda",
    "triagem", "outliers",
    "classificacao", "avancado",
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
def montar_drive(pasta="/content/drive/MyDrive/PNCP_TCC", force=True):
    """
    Monta o Google Drive e configura PASTA_DADOS para `pasta/dados`.

    `force=True` (padrão) refaz a montagem mesmo se o Colab achar que já
    está montado — resolve estados quebrados de sessões anteriores.
    Não apaga nem reescreve nenhum arquivo do Drive: só refaz a conexão.
    """
    from google.colab import drive
    drive.mount("/content/drive", force_remount=force)
    from pncp import config
    from pathlib import Path
    config.PASTA_DADOS = Path(pasta) / "dados"
    config.PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    print(f"[drive] PASTA_DADOS = {config.PASTA_DADOS}")
    return config.PASTA_DADOS


def atualizar(branch="refactor", repo_dir="/content/pncp-engineering-analytics"):
    """
    Faz git fetch + reset --hard na branch e recarrega o módulo `pncp`
    em memória. Use no INÍCIO de qualquer célula quando suspeitar que o
    código pode ter mudado no GitHub.

    Após chamar isso, é preciso fazer `import pncp` de novo na célula
    para usar a versão recarregada.
    """
    import subprocess, sys
    if not Path(repo_dir).exists():
        print(f"[atualizar] {repo_dir} não existe — clone primeiro")
        return
    subprocess.run(["git", "-C", repo_dir, "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", repo_dir, "checkout", branch], check=False)
    subprocess.run(["git", "-C", repo_dir, "reset", "--hard",
                    f"origin/{branch}"], check=True)
    out = subprocess.run(["git", "-C", repo_dir, "log", "-1", "--oneline"],
                         capture_output=True, text=True).stdout.strip()
    # Limpa o módulo da memória
    for mod in [m for m in list(sys.modules) if m.startswith("pncp")]:
        del sys.modules[mod]
    print(f"[atualizar] commit ativo: {out}")
    print(f"[atualizar] rode `import pncp` de novo na célula")


def keep_alive():
    """
    Injeta JavaScript no Colab para evitar desconexão por inatividade.

    Estratégia em duas camadas:
      1. Clica no botão de conectar a cada 60s (aciona heartbeat do runtime)
      2. Simula scroll/clique em elementos invisíveis a cada 30s para
         indicar atividade humana ao watchdog do Colab

    Não impede o limite duro de 12h em sessões free, mas evita desconexão
    por idle de ~30-90min que costuma cortar coletas longas.
    """
    try:
        from IPython.display import display, Javascript
        display(Javascript("""
            (function(){
              // Camada 1: aperta o botão de conectar (heartbeat oficial)
              setInterval(function(){
                const b = document.querySelector("colab-connect-button");
                if (b && typeof b.click === "function") b.click();
              }, 60000);

              // Camada 2: simula atividade do usuário
              setInterval(function(){
                document.dispatchEvent(new MouseEvent("mousemove",
                  {bubbles:true, clientX: Math.random()*5, clientY: Math.random()*5}));
              }, 30000);

              console.log("[pncp] keep-alive ativado (60s + 30s)");
            })();
        """))
        print("[colab] keep-alive ativado (heartbeat 60s + atividade 30s)")
    except Exception as e:
        print(f"[colab] keep-alive não pôde ser ativado: {e}")
