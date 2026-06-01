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


# ── Extração de entidades estruturadas via LLM ──────────────────────────────
# Inspirado no notebook 14 (NER + LLM) e Tutoria 13_05 (extração)
SYSTEM_NER = """Você extrai entidades estruturadas de objetos de contratos
públicos brasileiros do PNCP. Identifique e retorne em JSON:

{
  "tipo_servico": "obra civil" | "manutenção" | "limpeza" | "vigilância" |
                   "alimentação" | "transporte" | "fornecimento" |
                   "serviço técnico de engenharia" | "outro",
  "tem_engenheiro_responsavel": true | false,
  "menciona_normas_tecnicas": true | false,
  "menciona_projeto": true | false,
  "local_servico": "string ou null",
  "objetos_concretos": ["lista de coisas físicas mencionadas"],
  "verbos_obra": ["construir", "reformar", etc. — se houver],
  "valor_aproximado_mencionado": "string ou null",
  "indicador_engenharia": "alto" | "medio" | "baixo" | "nenhum"
}

Responda APENAS no JSON, sem prefácio."""


def extrair_entidades_llm(top_n=30, modelo="llama3.1", forcar=False):
    """
    Para os top suspeitos, extrai entidades estruturadas via LLM.
    Notebook 14 + tutoria 13_05 — extração estruturada de informações.

    Mais rico que NER do Spacy: identifica intenção semântica do contrato
    além das menções nominais.
    """
    susp_path = config.caminho(config.SUB_P9,
                                 "suspeitos_consolidados.parquet")
    if not Path(susp_path).exists():
        print("[llm.ner] rode pncp.relatorio.gerar() primeiro")
        return None
    saida = config.caminho("llm", "entidades_extraidas.parquet")
    if not forcar and saida.exists():
        print("[llm.ner] já rodou — use forcar=True")
        return ler_parquet(saida)

    df = ler_parquet(susp_path).head(top_n)
    print(f"[llm.ner] extraindo entidades de {len(df)} suspeitos...")

    resultados = []
    for i, row in df.reset_index(drop=True).iterrows():
        objeto = str(row.get("objeto", ""))[:600]
        resp_txt = _chat_ollama(modelo, SYSTEM_NER,
                                 f"Objeto:\n{objeto}\n\nExtraia.",
                                 max_tokens=400)
        resp = _parse_resposta(resp_txt) or {}
        resultados.append({
            "numeroControlePNCP": row.get("numeroControlePNCP"),
            "objeto": objeto[:200],
            "tipo_servico": resp.get("tipo_servico", "?"),
            "tem_engenheiro_responsavel":
                bool(resp.get("tem_engenheiro_responsavel", False)),
            "menciona_normas_tecnicas":
                bool(resp.get("menciona_normas_tecnicas", False)),
            "menciona_projeto": bool(resp.get("menciona_projeto", False)),
            "indicador_engenharia": resp.get("indicador_engenharia", "?"),
            "verbos_obra": ", ".join(resp.get("verbos_obra", []) or []),
            "objetos_concretos": ", ".join(
                resp.get("objetos_concretos", []) or []),
        })
        if (i + 1) % 5 == 0:
            print(f"[llm.ner] {i + 1}/{len(df)}")

    out = pd.DataFrame(resultados)
    salvar_parquet(out, saida)
    print(f"[llm.ner] {len(out)} entidades extraídas → {saida}")
    return out


# ── Sumarização ─────────────────────────────────────────────────────────────
SYSTEM_RESUMO = """Você resume em uma frase curta (≤25 palavras) o objeto
de um contrato público brasileiro. Foque no QUE é contratado e ONDE.
Não interprete legalidade. Responda apenas a frase, sem aspas, sem prefácio."""


def resumir_objetos(amostra=20, modelo="llama3.1"):
    """
    Resume objetos longos em frases curtas via LLM.
    Útil para relatório do TCC — apresentar suspeitos de forma escaneavel.
    """
    df = ler_parquet(config.caminho(config.SUB_P9,
                                     "suspeitos_consolidados.parquet"))
    df = df.head(amostra)
    saida = []
    for _, row in df.iterrows():
        obj = str(row.get("objeto", ""))[:1000]
        if len(obj) < 50:
            saida.append(obj)
            continue
        resumo = _chat_ollama(modelo, SYSTEM_RESUMO, obj, max_tokens=80) or obj
        saida.append(resumo.strip())
    df = df.copy()
    df["resumo_llm"] = saida
    out_path = config.caminho("llm", "resumos.parquet")
    salvar_parquet(df[["numeroControlePNCP", "objeto", "resumo_llm"]],
                    out_path)
    print(f"[llm.resumo] {len(df)} resumos → {out_path}")
    return df


# ── Geração de indicadores por cluster ──────────────────────────────────────
# Notebook 12 P2: LLM analisa cluster de docs e gera indicador estruturado
SYSTEM_INDICADOR = """Você é analista de inteligência analítica especializado
em contratações públicas brasileiras. Recebe uma amostra de objetos de
contratos que foram agrupados por similaridade semântica (cluster).

Sua tarefa: identificar UM indicador-chave que sintetize o padrão de
subenquadramento (ou ausência dele) presente no cluster.

Responda APENAS no JSON:
{
  "nome": "Nome curto descritivo (até 12 palavras)",
  "categoria": "infraestrutura" | "manutenção" | "obra civil" | "serviço técnico" |
                "fornecimento" | "vigilância/limpeza" | "transporte" | "outro",
  "descricao": "1-2 frases explicando o padrão observado",
  "indicio_subenquadramento": "alto" | "medio" | "baixo" | "nenhum",
  "justificativa_juridica": "referência à Lei 14.133 art. específico, se aplicável",
  "recomendacao_acao": "uma ação concreta para auditoria/controle",
  "exemplos_objetos": ["3 trechos curtos dos objetos do cluster"]
}"""


def gerar_indicadores(top_n_clusters=10, n_por_cluster=5,
                       modelo="llama3.1", forcar=False):
    """
    Para cada um dos top-N clusters de contratos similares, gera um
    indicador-síntese via LLM. Notebook 12 P2.

    Pré-requisito: rodar pncp.grafos_semanticos.construir() antes.
    """
    from pncp import grafos_semanticos
    saida = config.caminho("llm", "indicadores.json")
    if not forcar and saida.exists():
        print("[llm.indicadores] já gerado — use forcar=True")
        return _ler_json_safe(saida)

    amostras = grafos_semanticos.amostrar_por_cluster(
        n_por_cluster=n_por_cluster, min_tamanho=10)
    if amostras is None or amostras.empty:
        print("[llm.indicadores] sem clusters — rode "
              "pncp.grafos_semanticos.construir() primeiro")
        return None

    # Pega top-N clusters por tamanho
    cluster_ids = (amostras.groupby("cluster")["_tamanho_cluster"]
                   .first().sort_values(ascending=False)
                   .head(top_n_clusters).index)

    indicadores = []
    for cid in cluster_ids:
        sub = amostras[amostras["cluster"] == cid]
        tam = int(sub["_tamanho_cluster"].iloc[0])
        objetos = "\n".join(f"- {str(o)[:300]}"
                              for o in sub["objeto"].tolist())
        prompt = (f"Cluster #{cid} ({tam} contratos similares).\n\n"
                  f"Amostra de objetos:\n{objetos}\n\n"
                  f"Gere o indicador no JSON.")
        resp = _chat_ollama(modelo, SYSTEM_INDICADOR, prompt, max_tokens=600)
        parsed = _parse_resposta(resp) or {}
        parsed["cluster_id"] = int(cid)
        parsed["tamanho_cluster"] = tam
        indicadores.append(parsed)
        print(f"[llm.indicadores] cluster {cid}: "
              f"{parsed.get('nome', '?')[:60]}")

    salvar_json({"indicadores": indicadores, "modelo": modelo}, saida)
    print(f"[llm.indicadores] {len(indicadores)} indicadores → {saida}")
    return indicadores


def _ler_json_safe(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Agente simples LLM + ferramentas ────────────────────────────────────────
# Notebook 18: agente com Tools. Versão enxuta sem LangChain (que é pesado
# e tem mudanças quebradiças). Loop: prompt → LLM escolhe tool → executa →
# retorna resposta.
SYSTEM_AGENTE = """Você é um agente analítico para auditoria de contratos
públicos. Tem acesso a estas ferramentas (uma por chamada):

- buscar_similares(texto): retorna contratos parecidos a um texto livre
- listar_suspeitos(n): retorna os top-N suspeitos consolidados
- contar_por_rotulo(): distribuição de rótulos na base
- buscar_municipio(nome): contratos suspeitos em um município

Quando precisar de dados, responda em JSON:
{"acao": "<nome_ferramenta>", "argumentos": {...}}

Quando tiver dados suficientes para responder ao usuário, responda em JSON:
{"acao": "responder", "resposta": "<texto natural>"}"""


def _tool_buscar_similares(texto, k=5):
    from pncp import similaridade
    df = similaridade.buscar_por_texto(texto, k=k)
    if df is None or df.empty:
        return "(nenhum contrato similar encontrado)"
    linhas = [f"- {r['numeroControlePNCP']} | {str(r['objeto'])[:100]} "
              f"(sim={r['similaridade']:.2f})"
              for _, r in df.iterrows()]
    return "\n".join(linhas)


def _tool_listar_suspeitos(n=10):
    p = config.caminho(config.SUB_P9, "suspeitos_consolidados.parquet")
    if not Path(p).exists():
        return "(sem suspeitos — rode pncp.relatorio.gerar())"
    df = ler_parquet(p).head(n)
    return "\n".join(f"- {r['numeroControlePNCP']} | "
                     f"sinais={r.get('n_sinais', 0)} | "
                     f"{str(r.get('objeto', ''))[:80]}"
                     for _, r in df.iterrows())


def _tool_contar_por_rotulo():
    df = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"),
                     colunas=["rotulo"])
    return df["rotulo"].value_counts().to_string()


def _tool_buscar_municipio(nome):
    p = config.caminho(config.SUB_P9, "suspeitos_consolidados.parquet")
    if not Path(p).exists():
        return "(sem suspeitos)"
    df = ler_parquet(p)
    if "municipioNome" not in df.columns:
        return "(municipio não disponível)"
    sub = df[df["municipioNome"].astype(str).str.contains(nome,
                                                              case=False,
                                                              na=False)]
    if sub.empty:
        return f"(nenhum suspeito em {nome})"
    return f"{len(sub)} suspeito(s) em municípios que casam '{nome}':\n" + \
           "\n".join(f"- {str(r.get('objeto', ''))[:80]}"
                     for _, r in sub.head(10).iterrows())


_TOOLS = {
    "buscar_similares": lambda **kw: _tool_buscar_similares(**kw),
    "listar_suspeitos": lambda **kw: _tool_listar_suspeitos(**kw),
    "contar_por_rotulo": lambda **kw: _tool_contar_por_rotulo(**kw),
    "buscar_municipio": lambda **kw: _tool_buscar_municipio(**kw),
}


def agente(pergunta, modelo="llama3.1", max_passos=4):
    """
    Agente LLM + ferramentas. Notebook 18.

    Loop: o LLM lê a pergunta, decide qual ferramenta chamar, recebe a
    saída e decide se continua ou responde.

    Exemplo:
        pncp.llm.agente("Quais os principais padrões de subenquadramento?")
        pncp.llm.agente("Tem contratos parecidos com obras em São Paulo?")
    """
    contexto = f"Pergunta do usuário: {pergunta}"
    for passo in range(max_passos):
        resp = _chat_ollama(modelo, SYSTEM_AGENTE, contexto, max_tokens=500)
        parsed = _parse_resposta(resp) or {}
        acao = parsed.get("acao")
        if acao == "responder":
            return parsed.get("resposta", "(sem resposta)")
        if acao in _TOOLS:
            args = parsed.get("argumentos", {})
            try:
                saida = _TOOLS[acao](**args)
            except Exception as e:
                saida = f"(erro na ferramenta: {e})"
            contexto += f"\n\nResultado da ferramenta {acao}:\n{saida}"
            print(f"[agente] passo {passo + 1}: usou {acao}")
        else:
            return parsed.get("resposta", str(parsed))
    return "(agente atingiu limite de passos sem responder)"
