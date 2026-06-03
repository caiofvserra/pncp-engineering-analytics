"""
Marcadores legais e técnicos de engenharia em PDFs do PNCP.

Por que isso vale: a Lei 6.496/1977 obriga ART para qualquer atividade
de engenharia, mesmo quando o contrato é rotulado como serviço comum.
Contratos rotulados 'geral' que mencionam ART/CREA no TR ou edital
são candidatos prioritários a subenquadramento.

Escopo: apenas ENGENHARIA (CREA/ART) e OBRAS. Arquitetura (CAU/RRT) NÃO
entra na análise.

Os marcadores estão em 8 categorias (sigla, lei correspondente):
  ART (Lei 6.496/1977), CREA, ENGENHEIRO_RESPONSAVEL,
  ATC (atestado capacidade técnica), PROJETO_BASICO,
  OBRA_SERVICO_ENGENHARIA, ABNT_NORMAS,
  LEI_14133_ENGENHARIA (art. 6º XII e XX/XXI).

`detectar_marcadores(texto)` retorna dict com contagens, presença binária
e score agregado (0-9 = nº de categorias presentes).
"""

import re


# ── Tipos de documento PNCP (Manual §5.12) ──────────────────────────────────
MAPA_TIPO_DOCUMENTO = {
    1:  "Aviso de Contratação Direta",
    2:  "Edital",
    3:  "Minuta do Contrato",
    4:  "Termo de Referência",
    5:  "Anteprojeto",
    6:  "Projeto Básico",
    7:  "Estudo Técnico Preliminar",
    8:  "Projeto Executivo",
    9:  "Mapa de Riscos",
    10: "DOD/DFD",
    11: "Ata de Registro de Preço",
    12: "Contrato",
    13: "Termo de Rescisão",
    14: "Termo Aditivo",
    15: "Termo de Apostilamento",
    17: "Nota de Empenho",
}

# Documentos onde marcadores de engenharia, se existirem, têm que aparecer
TIPOS_RELEVANTES_ENGENHARIA = (4, 6, 7, 8, 5, 2, 3, 12, 14)


# ── Aditivo (alguns retornos da API só vêm com nome) ─────────────────────────
NOMES_TERMO_ADITIVO = ("termo aditivo", "aditivo", "termo de aditamento")


def eh_termo_aditivo(doc):
    """Identifica aditivo por ID OU por nome — API às vezes envia só um."""
    if doc.get("tipoDocumentoId") == 14:
        return True
    nome = (doc.get("tipoDocumentoNome") or "").lower()
    return any(t in nome for t in NOMES_TERMO_ADITIVO)


# ── Marcadores ──────────────────────────────────────────────────────────────
MARCADORES_ENGENHARIA = {
    # Lei 6.496/1977 — ART obrigatória para atividade de engenharia
    "ART": [
        r"\banota[çc][ãa]o\s+de\s+responsabilidade\s+t[ée]cnica\b",
        r"\bART\b(?:\s+do\s+CREA)?",
    ],
    "CREA": [
        r"\bCREA[/\s\-]?\w{0,2}\b",
        r"\bConselho\s+Regional\s+de\s+Engenharia\b",
        r"\bregistro\s+(?:no\s+)?CREA\b",
    ],
    "ENGENHEIRO_RESPONSAVEL": [
        r"\bengenheiro\s+respons[áa]vel\b",
        r"\bengenheira?\s+respons[áa]vel\b",
        r"\brespons[áa]vel\s+t[ée]cnico\b",
        r"\bRT\s+do\s+(?:contrato|servi[çc]o|edital)\b",
    ],
    # Atestado de Capacidade Técnica (ACT) — Lei 14.133 art. 67
    "ATESTADO_CAP_TECNICA": [
        r"\bquadro\s+de\s+respons[áa]veis\s+t[ée]cnicos\b",
        r"\batestado\s+de\s+capacidade\s+t[ée]cnica\b",
        r"\bACT\b(?:\s+do\s+CREA)?",
    ],
    "PROJETO_BASICO": [
        r"\bprojeto\s+b[áa]sico\b",
        r"\banteprojeto\s+de\s+engenharia\b",
        r"\bprojeto\s+executivo\b",
    ],
    "OBRA_SERVICO_ENGENHARIA": [
        r"\bobra\s+de\s+engenharia\b",
        r"\bservi[çc]o\s+(?:comum\s+)?de\s+engenharia\b",
        r"\bservi[çc]o\s+especial\s+de\s+engenharia\b",
    ],
    "ABNT_NORMA": [
        r"\bABNT\s+NBR\s*\d+",
        r"\bnorma\s+t[ée]cnica\s+(?:NBR\s*\d+|brasileira)\b",
    ],
    # Lei 14.133/2021 art. 6º XII (serv. eng.) e XX/XXI (obra)
    "LEI_14133_ENGENHARIA": [
        r"\bart\.?\s*6[°º]?,?\s*(?:inc(?:iso)?\.?\s*)?XII\b",
        r"\bart\.?\s*6[°º]?,?\s*(?:inc(?:iso)?\.?\s*)?XX(?:I+)?\b",
    ],
}

_RX_COMPILADO = {
    nome: [re.compile(p, flags=re.IGNORECASE) for p in pads]
    for nome, pads in MARCADORES_ENGENHARIA.items()
}


# ── Normalização de PDF ─────────────────────────────────────────────────────
# Sem isso, regex como `\bengenharia\b` perde 30-40% das ocorrências
# que estão quebradas em duas linhas como "engenha-\nria"
def normalizar_pdf_text(texto):
    if not texto:
        return ""
    # Hifenização de quebra de linha
    texto = re.sub(r"-\s*\n\s*", "", texto)
    # Quebras simples viram espaço (preserva parágrafos com \n\n)
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)
    # Espaços múltiplos
    texto = re.sub(r"[ \t]+", " ", texto)
    return texto


def detectar_marcadores(texto):
    """
    Retorna dict com:
      mk_<NOME>           contagem total de ocorrências
      mk_<NOME>_presente  bool
      mk_score_engenharia 0-9 (nº de categorias presentes)
    """
    t = normalizar_pdf_text(texto).lower() if texto else ""
    out = {}
    for nome, padroes in _RX_COMPILADO.items():
        total = sum(len(p.findall(t)) for p in padroes)
        out[f"mk_{nome}"] = total
        out[f"mk_{nome}_presente"] = total > 0
    out["mk_score_engenharia"] = sum(
        1 for k, v in out.items() if k.endswith("_presente") and v
    )
    return out


# Lista de nomes de colunas geradas (útil para agregação)
COLS_MARCADORES = [f"mk_{n}" for n in MARCADORES_ENGENHARIA]
COLS_PRESENCA = [f"mk_{n}_presente" for n in MARCADORES_ENGENHARIA]
