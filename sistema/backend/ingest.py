"""Semeadura (a partir do ranking do notebook) e ingestão contínua do PNCP.
Os contratos entram com status 'novo' (aguardando triagem do objeto)."""
import datetime as dt
import pandas as pd
import requests
from . import config, db, model

_DEMO = [
    ("c01", "RECAPEAMENTO ASFÁLTICO DE VIAS NO DISTRITO INDUSTRIAL", "Prefeitura Municipal", 3400000, "pavimentação", 0.99),
    ("c02", "REFORMA E AMPLIAÇÃO DA UNIDADE BÁSICA DE SAÚDE DO JARDIM SÃO JORGE", "Secretaria Municipal de Saúde", 1250000, "reforma/edificação", 0.97),
    ("c03", "CONTRATAÇÃO DE EMPRESA PARA TROCA DE PISO E REVESTIMENTO DA ESCOLA MUNICIPAL", "Secretaria de Educação", 480000, "reforma/edificação", 0.94),
    ("c04", "PAVIMENTAÇÃO E SINALIZAÇÃO VIÁRIA DO BAIRRO NOVA ESPERANÇA", "Prefeitura Municipal", 2100000, "pavimentação", 0.96),
    ("c05", "ELABORAÇÃO DE PROJETO ELÉTRICO E LAUDO TÉCNICO DA CRECHE MUNICIPAL", "Secretaria de Obras", 130000, "projetos/laudos", 0.91),
    ("c06", "CONSTRUÇÃO DE MURO DE ARRIMO E SISTEMA DE DRENAGEM NA ENCOSTA", "Prefeitura Municipal", 890000, "pavimentação", 0.90),
    ("c07", "IMPERMEABILIZAÇÃO DA LAJE E COBERTURA DO GINÁSIO MUNICIPAL", "Secretaria de Esportes", 260000, "reforma/edificação", 0.86),
    ("c08", "SERVIÇOS DE DETECÇÃO DE VAZAMENTOS NA REDE HIDRÁULICA DO CAMPUS", "Universidade Estadual", 310000, "saneamento/hidráulica", 0.88),
    ("c09", "REFORMA DOS SANITÁRIOS E INSTALAÇÕES HIDRÁULICAS DA ESCOLA ESTADUAL", "Secretaria de Educação", 340000, "saneamento/hidráulica", 0.89),
    ("c10", "PRESTAÇÃO DE SERVIÇOS DE MANUTENÇÃO ELÉTRICA PREDIAL NOS SETORES DA PREFEITURA", "Prefeitura Municipal", 175000, "elétrica/instalação", 0.83),
    ("c11", "SERVIÇOS DE MANUTENÇÃO DE VEÍCULOS DA FROTA OFICIAL", "Departamento de Transportes", 220000, "elétrica/instalação", 0.72),
    ("c12", "ESPETÁCULO MUSICAL E SONORIZAÇÃO PARA A FESTA DE ANIVERSÁRIO DA CIDADE", "Secretaria de Cultura", 260000, "elétrica/instalação", 0.60),
]


def _limiar():
    try:
        return float(db.get_config().get("limiar", 0.65))
    except Exception:
        return 0.65


def semear():
    if db.n_contratos() > 0:
        return "já havia dados"
    rows = []
    if config.RANKING_CSV.exists():
        d = pd.read_csv(config.RANKING_CSV, sep=";", decimal=",", dtype=str)
        d.columns = [c.strip() for c in d.columns]
        prob = pd.to_numeric(d.get("prob_eng_obra"), errors="coerce").fillna(0.5)
        val = pd.to_numeric(d.get("valor"), errors="coerce")
        for i, r in d.iterrows():
            rows.append(dict(id=str(r.get("numeroControlePNCP", f"row{i}")),
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
    sc = model.score_final([r["objeto"] for r in rows], [r["prob_base"] for r in rows])
    for r, s in zip(rows, sc):
        r["score"] = float(s)
    db.upsert_contratos(rows)
    return origem


def repontuar():
    """Recalcula o score da fila de triagem após um re-treino."""
    pend = db.pendentes_score()
    if not pend:
        return 0
    sc = model.score_final([r["objeto"] for r in pend], [r["prob_base"] for r in pend])
    db.upsert_contratos([dict(id=r["id"], objeto=r["objeto"], orgao=r["orgao"],
                              valor=r["valor"], tipo_eng=r["tipo_eng"],
                              prob_base=r["prob_base"], score=float(s),
                              origem=r["origem"]) for r, s in zip(pend, sc)])
    return len(pend)


def ingerir_pncp(paginas=2, tam=50):
    cfg = db.get_config(); uf = cfg.get("ingest_uf", "SP")
    hoje = dt.date.today()
    ini = (hoje - dt.timedelta(days=30)).strftime("%Y%m%d"); fim = hoje.strftime("%Y%m%d")
    novos = []
    for pag in range(1, paginas + 1):
        try:
            r = requests.get(config.PNCP_CONSULTA, timeout=30,
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
                novos.append(dict(id=str(c.get("numeroControlePNCP")), objeto=obj[:500],
                                  orgao=(c.get("orgaoEntidade") or {}).get("razaosocial", ""),
                                  valor=c.get("valorGlobal"), tipo_eng="",
                                  prob_base=0.5, score=0.5, origem="pncp"))
        except Exception:
            break
    if not novos:
        return 0
    sc = model.score_final([r["objeto"] for r in novos], [r["prob_base"] for r in novos])
    filtrados = [dict(r, score=float(s)) for r, s in zip(novos, sc) if s >= _limiar()]
    db.upsert_contratos(filtrados)
    return len(filtrados)
