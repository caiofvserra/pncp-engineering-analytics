"""
Validação semântica de suspeitos via LLM.

Inspirado nas aulas 11/12 do MBA (Marcacini, ICMC/USP). Roda Llama 3.1
local via Ollama — gratuito, privado, não depende de API paga.

Uso típico:
    pncp.llm.iniciar_ollama()              # 1x por sessão
    pncp.llm.pull_modelo('llama3.1')       # 1x por sessão
    pncp.llm.validar_suspeitos(top_n=20)   # analisa os 20 mais suspeitos

A LLM recebe objeto + sinais detectados pelo pipeline + texto-trecho do
PDF (se houver) e dá veredicto estruturado em JSON: classe predita,
confiança, justificativa, indício de rito de engenharia, recomendação.
"""

import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from pncp import config
from pncp.io_disco import ler_parquet, salvar_parquet, salvar_json
from pncp.ram import com_gc


# ── Setup do Ollama (chamar 1x por sessão) ──────────────────────────────────
def iniciar_ollama(porta=11434):
    """
    Inicia o servidor Ollama em background. Idempotente — se já estiver
    rodando, não dispara segunda instância.
    """
    try:
        import requests
        r = requests.get(f"http://127.0.0.1:{porta}/api/tags", timeout=2)
        if r.status_code == 200:
            print("[llm] Ollama já está rodando")
            return True
    except Exception:
        pass

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=open("/tmp/ollama.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(5)
        import requests
        r = requests.get(f"http://127.0.0.1:{porta}/api/tags", timeout=5)
        ok = r.status_code == 200
        print(f"[llm] Ollama {'iniciado' if ok else 'falhou ao iniciar'}")
        return ok
    except FileNotFoundError:
        print("[llm] ollama não instalado. No Colab rode:")
        print('   !curl -fsSL https://ollama.com/install.sh | sh')
        return False


def pull_modelo(modelo="llama3.1"):
    """Baixa o modelo via `ollama pull <modelo>`. Idempotente."""
    try:
        out = subprocess.run(
            ["ollama", "pull", modelo],
            capture_output=True, text=True, timeout=600,
        )
        if out.returncode == 0:
            print(f"[llm] modelo {modelo} pronto")
            return True
        print(f"[llm] pull falhou: {out.stderr[:200]}")
        return False
    except Exception as e:
        print(f"[llm] erro ao pull modelo: {e}")
        return False


# ── Chamada ao LLM ──────────────────────────────────────────────────────────
def _chat_ollama(modelo, system, prompt, max_tokens=500, temperatura=0.2):
    """Wrapper para a API do Ollama (compatível com formato OpenAI)."""
    import requests
    try:
        r = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": modelo,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": temperatura,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        if r.status_code != 200:
            return None
        return r.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[llm] erro: {e}")
        return None


# ── Prompt engineering para classificação de contratos ──────────────────────
SYSTEM_PROMPT = """Você é um analista jurídico-administrativo especializado na
Lei 14.133/2021 (Nova Lei de Licitações). Sua tarefa é classificar contratos
públicos do PNCP (Portal Nacional de Contratações Públicas) entre três categorias:

- "engenharia": contrato cujo objeto envolve serviço de engenharia
  (art. 6º XII da Lei 14.133/2021). Caracteriza-se por exigir profissional
  inscrito no CREA/CAU, ART/RRT, projeto básico, memorial descritivo.
  Exemplos: manutenção elétrica predial, instalação hidráulica, reforma
  estrutural, drenagem, pavimentação, projeto arquitetônico.

- "obras": contrato cujo objeto envolve obra civil (art. 6º XX/XXI).
  Caracteriza-se por modificar estruturalmente o bem (construção, reforma
  ampla, demolição, ampliação). Exemplos: construção de prédio, ponte,
  pavimentação asfáltica, reforma estrutural completa.

- "geral": qualquer serviço comum que NÃO se enquadre acima.
  Exemplos: limpeza, vigilância, alimentação, fornecimento de material
  de escritório, manutenção de equipamentos não-prediais, transporte.

Responda APENAS no formato JSON estrito:
{
  "classe": "engenharia" | "obras" | "geral",
  "confianca": 0.0 a 1.0,
  "justificativa": "frase curta de até 30 palavras",
  "exige_art_rrt": true | false,
  "recomendacao": "string"
}"""


def _montar_prompt(contrato):
    """Monta o prompt para um contrato. Inclui sinais detectados."""
    obj = str(contrato.get("objeto", ""))[:600]
    rotulo = contrato.get("rotulo", "?")
    valor = contrato.get("valor")
    n_sinais = contrato.get("n_sinais", 0)
    prob = contrato.get("prob_engenharia")
    score_pdf = contrato.get("mk_score_engenharia", 0)

    sinais = []
    if prob is not None and prob > 0.5:
        sinais.append(f"ML deu {prob:.0%} de prob de engenharia")
    if score_pdf > 0:
        sinais.append(f"PDFs anexados têm {score_pdf} categoria(s) de marcadores legais "
                       f"(ART/RRT/CREA/Norma ABNT etc.)")
    if contrato.get("tem_cnae_eng"):
        sinais.append("Fornecedor tem CNAE de engenharia na Receita")
    if contrato.get("tem_mudanca_escopo"):
        sinais.append("Recebeu aditivo com mudança de escopo p/ engenharia")
    if contrato.get("eh_obvio_engenharia"):
        sinais.append("Objeto casa com padrão lexical óbvio de engenharia")

    bloco_sinais = ("\n".join(f"- {s}" for s in sinais)
                     if sinais else "(nenhum sinal automatizado)")
    return f"""Contrato a classificar:

OBJETO: "{obj}"

Rótulo atribuído pelo órgão: {rotulo}
Valor estimado: R$ {valor:,.2f}{'' if valor else ''}
Total de sinais positivos: {n_sinais}

Sinais detectados pelo pipeline automatizado:
{bloco_sinais}

Pergunta: este contrato é de fato 'geral' (serviço comum) ou foi
SUBENQUADRADO (deveria ser 'engenharia' ou 'obras')?

Responda no JSON estrito definido nas instruções."""


def _parse_resposta(texto):
    """Extrai o JSON do output do LLM (LLMs costumam adicionar prefácio)."""
    if not texto:
        return None
    # Procura primeiro bloco {...}
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", texto, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Tenta consertar aspas simples → duplas
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return None


# ── Pipeline principal ──────────────────────────────────────────────────────
@com_gc
def validar_suspeitos(top_n=20, modelo="llama3.1", forcar=False):
    """
    Pega os top-N suspeitos consolidados, manda para o LLM, salva resultados.

    Args:
      top_n: quantos contratos analisar (LLM local: ~5s/contrato)
      modelo: nome do modelo Ollama (precisa ter feito pull_modelo() antes)
      forcar: se True, re-analisa mesmo se já tem resultado

    Returns:
      DataFrame com colunas adicionais: llm_classe, llm_confianca,
      llm_justificativa, llm_exige_art_rrt, llm_recomendacao.
    """
    suspeitos_path = config.caminho(config.SUB_P9,
                                      "suspeitos_consolidados.parquet")
    if not suspeitos_path.exists():
        print("[llm] rode pncp.relatorio.gerar() primeiro")
        return None

    saida_path = config.caminho("llm", "validacao_llm.parquet")
    if not forcar and saida_path.exists():
        print("[llm] já rodou — use forcar=True para refazer")
        return ler_parquet(saida_path)

    df = ler_parquet(suspeitos_path).head(top_n)
    print(f"[llm] validando top {len(df)} suspeitos com {modelo}...")

    resultados = []
    for i, row in df.reset_index(drop=True).iterrows():
        prompt = _montar_prompt(row.to_dict())
        resp_txt = _chat_ollama(modelo, SYSTEM_PROMPT, prompt)
        resp = _parse_resposta(resp_txt) or {}

        resultados.append({
            "numeroControlePNCP": row.get("numeroControlePNCP"),
            "objeto": str(row.get("objeto", ""))[:200],
            "rotulo_original": row.get("rotulo"),
            "n_sinais": row.get("n_sinais", 0),
            "llm_classe": resp.get("classe", "?"),
            "llm_confianca": float(resp.get("confianca", 0) or 0),
            "llm_justificativa": resp.get("justificativa", "")[:300],
            "llm_exige_art_rrt": bool(resp.get("exige_art_rrt", False)),
            "llm_recomendacao": resp.get("recomendacao", "")[:300],
            "llm_resposta_bruta": (resp_txt or "")[:500],
        })

        if (i + 1) % 5 == 0:
            print(f"[llm] {i + 1}/{len(df)}")

    out = pd.DataFrame(resultados)
    salvar_parquet(out, saida_path)

    # Resumo
    if not out.empty:
        cont = out["llm_classe"].value_counts().to_dict()
        n_subenq = int((out["llm_classe"].isin(["engenharia", "obras"])).sum())
        n_rito = int(out["llm_exige_art_rrt"].sum())
        salvar_json({
            "n_avaliados": int(len(out)),
            "distribuicao_llm": cont,
            "n_subenquadramentos_apontados": n_subenq,
            "n_exigem_art_rrt": n_rito,
            "modelo": modelo,
        }, config.caminho("llm", "resumo.json"))
        print(f"\n[llm] resultado: {cont}")
        print(f"   subenquadramentos apontados pelo LLM: {n_subenq}/{len(out)}")
        print(f"   exigem ART/RRT: {n_rito}/{len(out)}")
    return out


def mostrar(top_n=10):
    """Imprime as N validações com justificativa do LLM."""
    p = config.caminho("llm", "validacao_llm.parquet")
    if not Path(p).exists():
        print("[llm.mostrar] rode pncp.llm.validar_suspeitos() primeiro")
        return
    df = ler_parquet(p).head(top_n)
    print(f"\n🤖 Validação LLM dos top {len(df)} suspeitos:\n")
    for _, r in df.iterrows():
        print(f"  • [{r['llm_classe']:>10s}] conf={r['llm_confianca']:.2f}  "
              f"ART/RRT={'✓' if r['llm_exige_art_rrt'] else '✗'}")
        print(f"    {r['numeroControlePNCP']}")
        print(f"    obj: {str(r['objeto'])[:100]}")
        print(f"    💡 {r['llm_justificativa']}")
        if r.get("llm_recomendacao"):
            print(f"    📋 {r['llm_recomendacao']}")
        print()
