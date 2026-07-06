"""Análise de RITO — etapa POSTERIOR à triagem (espelha a Etapa 11 do notebook).

Para um contrato já confirmado como engenharia na triagem, o sistema:
  1. resolve o vínculo contrato -> compra (numeroControlePNCP + API do PNCP);
  2. baixa os documentos da licitação (Edital / Termo de Referência / Projeto Básico);
  3. extrai o texto (PyMuPDF) e procura os MARCADORES do rito de engenharia;
  4. apresenta as evidências para o revisor dar o veredito final.

Robusto: PyMuPDF é importado sob demanda (o app roda sem ele em modo demo).
Contratos de demonstração usam trechos de TR embutidos (para o fluxo funcionar
offline). Falhas de rede/leitura viram 'indeterminado', nunca derrubam o app.
"""
import json
import re
import requests
from . import config

_HEAD = {"User-Agent": "Mozilla/5.0 (monitor-pncp)"}
_RX_NCP = re.compile(r"^(?P<cnpj>\d{14})-(?P<tipo>\d+)-(?P<seq>\d+)/(?P<ano>\d{4})$")

# Marcadores legais do rito de engenharia (só CREA/ART; sem CAU/RRT).
MARCADORES = {
    "ART/CREA": [r"\banota[çc][ãa]o\s+de\s+responsabilidade\s+t[ée]cnica\b",
                 r"\bART\b", r"\bCREA\b"],
    "Responsável técnico": [r"\brespons[áa]vel\s+t[ée]cnico\b",
                            r"\bengenheir[oa]?\s+respons[áa]vel\b"],
    "Projeto básico/executivo": [r"\bprojeto\s+b[áa]sico\b", r"\bprojeto\s+executivo\b",
                                 r"\bmemorial\s+descritivo\b"],
    "Planilha orçamentária": [r"\bplanilha\s+or[çc]ament[áa]ria\b",
                              r"\bcomposi[çc][ãa]o\s+de\s+custos\b"],
    "BDI": [r"\bBDI\b"],
    "SINAPI/SICRO": [r"\bSINAPI\b", r"\bSICRO\b"],
    "Cronograma físico-financeiro": [r"\bcronograma\s+f[íi]sico[- ]financeiro\b"],
    "Normas ABNT": [r"\bABNT\b", r"\bNBR\s*\d+"],
    "Atestado de cap. técnica": [r"\batestado\s+de\s+capacidade\s+t[ée]cnica\b"],
}
_RX_MARC = {k: [re.compile(p, re.IGNORECASE) for p in ps] for k, ps in MARCADORES.items()}

# Trechos de TR embutidos para os contratos de demonstração (id -> texto).
DEMO_TR = {
    "c02": ("TERMO DE REFERÊNCIA — REFORMA E AMPLIAÇÃO DA UBS. Exige-se do "
            "licitante a apresentação de projeto básico, memorial descritivo, "
            "planilha orçamentária com composição de custos e BDI, cronograma "
            "físico-financeiro, e a Anotação de Responsabilidade Técnica (ART) "
            "do engenheiro responsável junto ao CREA. Serão observadas as normas "
            "ABNT NBR aplicáveis."),
    "c05": ("PROJETO BÁSICO — ELABORAÇÃO DE PROJETO ELÉTRICO. O responsável "
            "técnico deverá recolher ART junto ao CREA e seguir as normas ABNT NBR "
            "5410. Apresentar memorial descritivo e planilha orçamentária."),
    "c03": ("TERMO DE REFERÊNCIA — TROCA DE PISO E REVESTIMENTO. Objeto: "
            "fornecimento de material e mão de obra para substituição do piso. "
            "Pagamento conforme medição. Não há exigência de projeto nem de "
            "responsável técnico registrado; contratação como serviço comum."),
    "c11": ("TERMO DE REFERÊNCIA — MANUTENÇÃO DE VEÍCULOS DA FROTA. Serviços "
            "mecânicos de manutenção preventiva e corretiva; peças e mão de obra. "
            "Serviço comum, sem exigências de engenharia."),
}


def _decompor(ncp):
    m = _RX_NCP.match(str(ncp).strip()) if ncp else None
    return ({"cnpj": m["cnpj"], "tipo": int(m["tipo"]),
             "ano": int(m["ano"]), "seq": int(m["seq"])} if m else None)


def detectar_marcadores(texto):
    achados = [k for k, rxs in _RX_MARC.items() if any(r.search(texto) for r in rxs)]
    return achados, len(achados)


def _resolver_compra(ncp):
    info = _decompor(ncp)
    if not info:
        return None
    if info["tipo"] == 1:                       # já é uma contratação
        return info
    try:
        r = requests.get(f'{config.PNCP_PNCP}/orgaos/{info["cnpj"]}/contratos/'
                         f'{info["ano"]}/{info["seq"]}', timeout=25, headers=_HEAD)
        if r.status_code == 200:
            d = r.json()
            for k, v in (d.items() if isinstance(d, dict) else []):
                if "compra" in k.lower() and isinstance(v, str) and _RX_NCP.match(v):
                    return _decompor(v)
            sq, an = d.get("sequencialCompra"), d.get("anoCompra")
            if sq and an:
                return {"cnpj": info["cnpj"], "tipo": 1, "ano": int(an), "seq": int(sq)}
    except Exception:
        return None
    return None


def _baixar_textos(info, max_docs):
    """Lista e baixa documentos da compra; extrai texto com PyMuPDF (lazy)."""
    try:
        import fitz  # PyMuPDF — só é necessário no download real
    except Exception:
        return "", 0, "PyMuPDF (pymupdf) não instalado no servidor."
    try:
        r = requests.get(f'{config.PNCP_PNCP}/orgaos/{info["cnpj"]}/compras/'
                         f'{info["ano"]}/{info["seq"]}/arquivos', timeout=25, headers=_HEAD)
        docs = (r.json() if r.status_code == 200 else []) or []
        docs = docs if isinstance(docs, list) else docs.get("data", [])
    except Exception:
        return "", 0, "falha ao listar documentos"
    textos, baixados = [], 0
    for d in docs[:max_docs]:
        url = next((d[c] for c in ("url", "uri", "link") if d.get(c)), None)
        if not url:
            continue
        try:
            pdf = requests.get(url, timeout=60, headers=_HEAD).content
            doc = fitz.open(stream=pdf, filetype="pdf")
            textos.append("\n".join(p.get_text() for p in list(doc)[:30]))
            doc.close(); baixados += 1
        except Exception:
            continue
    return "\n".join(textos), baixados, "ok"


def analisar(contrato, max_docs=3):
    """Executa a análise de rito e devolve as evidências (dict)."""
    cid = contrato["id"]
    if cid in DEMO_TR:                           # contrato de demonstração
        texto = DEMO_TR[cid]
        achados, score = detectar_marcadores(texto)
        return {"ncp_compra": "(demonstração)", "n_docs": 1, "chars": len(texto),
                "marcadores": json.dumps(achados, ensure_ascii=False),
                "mk_score": score, "trecho": texto[:1500], "obtido": True}
    info = _resolver_compra(cid)
    if not info:
        return {"ncp_compra": "", "n_docs": 0, "chars": 0, "marcadores": "[]",
                "mk_score": 0, "trecho": "Não foi possível resolver a compra do "
                "contrato no PNCP.", "obtido": False}
    ncp_compra = f'{info["cnpj"]}-1-{info["seq"]:06d}/{info["ano"]}'
    texto, n, msg = _baixar_textos(info, max_docs)
    if len(texto) < 200:
        return {"ncp_compra": ncp_compra, "n_docs": n, "chars": len(texto),
                "marcadores": "[]", "mk_score": 0,
                "trecho": f"Documento não obtido/ilegível ({msg}).", "obtido": False}
    achados, score = detectar_marcadores(texto)
    return {"ncp_compra": ncp_compra, "n_docs": n, "chars": len(texto),
            "marcadores": json.dumps(achados, ensure_ascii=False), "mk_score": score,
            "trecho": texto[:1500], "obtido": True}
