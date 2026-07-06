"""Semeadura e ingestão de contratos.

- semear(): popula a fila a partir do ranking do notebook (07_ranking_suspeitos.csv);
  se o arquivo não existir, cria dados de DEMONSTRAÇÃO para a UI funcionar já.
- ingerir_pncp(): busca contratos recentes na API pública do PNCP, pontua e
  adiciona à fila os que passam do limiar. Tolerante a falhas de rede.
"""
import datetime as dt
import numpy as np
import pandas as pd
import requests
from . import config, db, model

_DEMO = [
    ("SP-DEMO-0001", "REFORMA E AMPLIAÇÃO DA UNIDADE BÁSICA DE SAÚDE DO JARDIM SÃO JORGE",
     "SECRETARIA MUNICIPAL DE SAÚDE", 1_250_000, "reforma/edificação", 0.97),
    ("SP-DEMO-0002", "RECAPEAMENTO ASFÁLTICO DE VIAS NO DISTRITO INDUSTRIAL",
     "PREFEITURA MUNICIPAL", 3_400_000, "pavimentação", 0.99),
    ("SP-DEMO-0003", "SERVIÇOS DE MANUTENÇÃO DE VEÍCULOS DA FROTA OFICIAL",
     "DEPARTAMENTO DE TRANSPORTES", 220_000, "elétrica/instalação", 0.71),
    ("SP-DEMO-0004", "CONTRATAÇÃO DE EMPRESA PARA TROCA DE PISO E REVESTIMENTO DA ESCOLA",
     "SECRETARIA DE EDUCAÇÃO", 480_000, "reforma/edificação", 0.93),
    ("SP-DEMO-0005", "AQUISIÇÃO DE VALE-TRANSPORTE PARA SERVIDORES",
     "SECRETARIA DE ADMINISTRAÇÃO", 95_000, "reforma/edificação", 0.58),
    ("SP-DEMO-0006", "ELABORAÇÃO DE PROJETO ELÉTRICO E LAUDO TÉCNICO DA CRECHE MUNICIPAL",
     "SECRETARIA DE OBRAS", 130_000, "projetos/laudos", 0.9),
    ("SP-DEMO-0007", "SERVIÇOS DE DETECÇÃO DE VAZAMENTOS NA REDE HIDRÁULICA DO CAMPUS",
     "UNIVERSIDADE ESTADUAL", 310_000, "saneamento/hidráulica", 0.86),
    ("SP-DEMO-0008", "ESPETÁCULO MUSICAL E SONORIZAÇÃO PARA FESTIVIDADE DE ANIVERSÁRIO",
     "SECRETARIA DE CULTURA", 260_000, "elétrica/instalação", 0.55),
]


def _limiar():
    try:
        return float(db.get_config().get("limiar", 0.65))
    except Exception:
        return 0.65


def semear():
    """Popula a fila só se estiver vazia."""
    if db.n_contratos() > 0:
        return "já havia dados"
    rows = []
    if config.RANKING_CSV.exists():
        d = pd.read_csv(config.RANKING_CSV, sep=";", decimal=",", dtype=str)
        d.columns = [c.strip() for c in d.columns]
        prob = pd.to_numeric(d.get("prob_eng_obra"), errors="coerce").fillna(0.5)
        val = pd.to_numeric(d.get("valor"), errors="coerce")
        for i, r in d.iterrows():
            rows.append(dict(
                id=str(r.get("numeroControlePNCP", f"row{i}")),
                objeto=str(r.get("text", ""))[:500],
                orgao=str(r.get("razaoSocialOrgao", "") or ""),
                valor=float(val.iloc[i]) if pd.notna(val.iloc[i]) else None,
                tipo_eng=str(r.get("tipo_eng", "") or ""),
                prob_base=float(prob.iloc[i]), score=float(prob.iloc[i]),
                origem="notebook"))
        origem = f"ranking do notebook ({len(rows)} contratos)"
    else:
        for cid, obj, org, val, tp, pb in _DEMO:
            rows.append(dict(id=cid, objeto=obj, orgao=org, valor=val, tipo_eng=tp,
                             prob_base=pb, score=pb, origem="demo"))
        origem = "DEMONSTRAÇÃO (ranking do notebook não encontrado)"
    # score final já considera o modelo online, se houver
    sc = model.score_final([r["objeto"] for r in rows], [r["prob_base"] for r in rows])
    for r, s in zip(rows, sc):
        r["score"] = float(s)
    db.upsert_contratos(rows)
    return origem


def repontuar_fila():
    """Recalcula o score da fila pendente (após um re-treino do modelo online)."""
    pend = db.fila(limit=100000)
    if not pend:
        return 0
    sc = model.score_final([r["objeto"] for r in pend],
                           [r.get("prob_base") for r in pend])
    db.upsert_contratos([dict(id=r["id"], objeto=r["objeto"], orgao=r.get("orgao"),
                              valor=r.get("valor"), tipo_eng=r.get("tipo_eng"),
                              prob_base=r.get("prob_base"), score=float(s),
                              origem=r.get("origem", "notebook"))
                         for r, s in zip(pend, sc)])
    return len(pend)


def ingerir_pncp(paginas=2, tam=50):
    """Busca contratos recentes no PNCP e adiciona à fila os suspeitos.
    Silencioso em caso de falha de rede (retorna 0)."""
    cfg = db.get_config()
    uf = cfg.get("ingest_uf", "SP")
    hoje = dt.date.today()
    ini = (hoje - dt.timedelta(days=30)).strftime("%Y%m%d")
    fim = hoje.strftime("%Y%m%d")
    novos = []
    for pag in range(1, paginas + 1):
        try:
            r = requests.get(config.PNCP_API, timeout=30,
                             headers={"User-Agent": "Mozilla/5.0 (monitor-pncp)"},
                             params={"dataInicial": ini, "dataFinal": fim,
                                     "pagina": pag, "tamanhoPagina": tam})
            if r.status_code != 200:
                break
            data = r.json().get("data", [])
            if not data:
                break
            for c in data:
                if uf and (c.get("unidadeOrgao") or {}).get("ufSigla") != uf:
                    continue
                obj = (c.get("objetoContrato") or "").strip()
                if len(obj) < 20:
                    continue
                novos.append(dict(
                    id=str(c.get("numeroControlePNCP")), objeto=obj[:500],
                    orgao=(c.get("orgaoEntidade") or {}).get("razaosocial", ""),
                    valor=c.get("valorGlobal"), tipo_eng="",
                    prob_base=0.5, score=0.5, origem="pncp"))
        except Exception:
            break
    if not novos:
        return 0
    sc = model.score_final([r["objeto"] for r in novos],
                           [r["prob_base"] for r in novos])
    lim = _limiar()
    filtrados = [dict(r, score=float(s)) for r, s in zip(novos, sc) if s >= lim]
    db.upsert_contratos(filtrados)
    return len(filtrados)
