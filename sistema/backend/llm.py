"""Módulo de LLM (opcional) — veredito do objeto e leitura do rito nos PDFs.

Usa um servidor Ollama local (ou compatível) por HTTP. É OPCIONAL: se estiver
desativado na configuração ou fora do ar, as funções devolvem None e o sistema
segue funcionando com o classificador + revisor humano. Sem chaves de API, sem
custo por chamada, sem dependência externa em tempo de execução."""
import json
import re
import requests
from . import db

_CTX = (
    "Você é engenheiro analista de contratações públicas (Lei 14.133/2021). "
    "CONTA como engenharia/obras: construção, reforma, ampliação, demolição, "
    "pavimentação/recapeamento, drenagem, terraplanagem, impermeabilização, "
    "pintura predial, troca de piso/telhado/revestimento, manutenção predial, "
    "instalação/manutenção elétrica/hidráulica/estrutural, e projetos/laudos de "
    "engenharia. NÃO conta: aquisição de bens, locação, evento/show, serviços "
    "administrativos, alimentação, transporte, manutenção de veículos, "
    "limpeza/jardinagem comum.")


def _cfg():
    c = db.get_config()
    return c.get("llm_ativo") == "1", c.get("llm_base_url", ""), c.get("llm_modelo", "")


def disponivel():
    ativo, url, _ = _cfg()
    if not ativo or not url:
        return False
    try:
        return requests.get(url.rstrip("/") + "/api/tags", timeout=3).ok
    except Exception:
        return False


def _chamar(system, prompt, max_tokens=400):
    ativo, url, modelo = _cfg()
    if not ativo or not url:
        return None
    try:
        r = requests.post(url.rstrip("/") + "/api/chat", timeout=120, json={
            "model": modelo, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": 0.1, "num_predict": max_tokens}})
        if not r.ok:
            return None
        txt = r.json().get("message", {}).get("content", "")
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def veredito(objeto):
    """Classifica o OBJETO. Devolve {classe, confianca, motivo} ou None."""
    sistema = _CTX + (' Responda SÓ JSON: {"classe":"engenharia"|"nao",'
                      '"confianca":0.0-1.0,"motivo":"até 25 palavras"}')
    return _chamar(sistema, f"Objeto do contrato: {str(objeto)[:600]}")


def analisar_rito(objeto, texto_tr):
    """Lê o trecho do TR e diz se há rito de engenharia. Devolve {rito, motivo}."""
    sistema = (_CTX + " Você recebe o OBJETO e um TRECHO do Termo de Referência "
               "da licitação. Diga se o documento evidencia o RITO DE ENGENHARIA "
               "(ART/CREA, projeto básico, responsável técnico, ABNT, planilha "
               'orçamentária). Responda SÓ JSON: {"rito":"sim"|"nao"|"parcial",'
               '"motivo":"até 25 palavras"}')
    return _chamar(sistema, f"OBJETO: {str(objeto)[:300]}\n\nTRECHO DO TR:\n{texto_tr[:4000]}")
