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
    "llm",
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

    Estratégia em três camadas — necessária quando o usuário trabalha em
    outra janela (PowerAutomate, outro navegador) e a aba do Colab fica
    em segundo plano:

      1. Clica `colab-connect-button` a cada 60s (heartbeat oficial)
      2. Dispara `mousemove` a cada 30s (sinaliza atividade humana)
      3. Toca um oscilador Web Audio inaudível continuamente — isto
         IMPEDE o navegador de aplicar throttling à aba em background
         (Chrome/Firefox não suspendem timers de abas que tocam áudio).

    A camada 3 é o que faz a diferença quando você não está olhando
    para o Colab. Limites:
      - Não escapa do timeout duro de 12h (free)
      - Se o navegador inteiro for fechado, perde tudo

    Quando isso falhar, a retomada automática de pncp.coleta.coletar()
    permite continuar do ponto exato onde parou.
    """
    try:
        from IPython.display import display, Javascript
        display(Javascript("""
            (function(){
              // Camada 1: heartbeat oficial do Colab
              setInterval(function(){
                const b = document.querySelector("colab-connect-button");
                if (b && typeof b.click === "function") b.click();
              }, 60000);

              // Camada 2: simula atividade do usuário
              setInterval(function(){
                document.dispatchEvent(new MouseEvent("mousemove",
                  {bubbles:true, clientX: Math.random()*5, clientY: Math.random()*5}));
              }, 30000);

              // Camada 3: áudio inaudível — impede o navegador de
              // suspender a aba quando ela está em background
              try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                gain.gain.value = 0.0001;            // praticamente mudo
                osc.frequency.value = 440;
                osc.connect(gain).connect(ctx.destination);
                osc.start();
                window._pncpKeepAliveAudio = {ctx, osc, gain};
                console.log("[pncp] keep-alive 3 camadas ativado");
              } catch(e) {
                console.log("[pncp] WebAudio não disponível: " + e.message);
              }
            })();
        """))
        print("[colab] keep-alive ativado (heartbeat 60s + atividade 30s "
              "+ WebAudio anti-throttle)")
        print("        ⚠ mantenha a aba do Colab aberta (mesmo em outra janela)")
    except Exception as e:
        print(f"[colab] keep-alive não pôde ser ativado: {e}")


def snapshot_auto(prefixo="run", incluir_pdfs_cache=False):
    """Snapshot com nome automático = prefixo_YYYY-MM-DD_HHMMSS.

    Use no fim de cada Run all completo para preservar o estado antes de
    rodar de novo com dados maiores (ex: depois de adicionar 2025-2026).
    """
    from datetime import datetime as _dt
    nome = f"{prefixo}_{_dt.now().strftime('%Y-%m-%d_%H%M%S')}"
    return snapshot(nome, incluir_pdfs_cache=incluir_pdfs_cache)


def snapshot(nome, incluir_pdfs_cache=False):
    """
    Guarda uma cópia do estado atual de `dados/` numa subpasta
    `dados/snapshots/<nome>/`. Útil antes de juntar mais dados (ex: você
    coletou 2024 inteiro, vai analisar, e depois quer baixar 2025-2026
    sem perder o estado "só 2024" para comparação no TCC).

    Por padrão NÃO inclui o cache de PDFs (pesado). Para incluir tudo:
    `pncp.snapshot('nome', incluir_pdfs_cache=True)`.

    Para restaurar manualmente, copie de volta os arquivos da subpasta.
    """
    import shutil
    from pncp import config

    base = config.PASTA_DADOS
    if not base.exists():
        print(f"[snapshot] {base} ainda não existe — nada a salvar")
        return None

    destino = base / "snapshots" / nome
    if destino.exists():
        print(f"[snapshot] '{nome}' já existe em {destino}")
        print(f"           use outro nome ou apague o anterior manualmente")
        return None

    destino.mkdir(parents=True)
    n_arq = 0
    for item in base.iterdir():
        if item.name == "snapshots":
            continue
        if not incluir_pdfs_cache and item.name == config.SUB_C2:
            # Copia features_pdfs.parquet e resumo.json mas pula o cache
            sub_dst = destino / item.name
            sub_dst.mkdir()
            for f in item.iterdir():
                if f.name == "cache_pdfs":
                    continue
                if f.is_file():
                    shutil.copy2(f, sub_dst / f.name)
                    n_arq += 1
            continue
        if item.is_dir():
            shutil.copytree(item, destino / item.name)
            n_arq += sum(1 for _ in (destino / item.name).rglob("*"))
        else:
            shutil.copy2(item, destino / item.name)
            n_arq += 1

    print(f"[snapshot] '{nome}' salvo em {destino} ({n_arq} arquivos)")
    return destino
