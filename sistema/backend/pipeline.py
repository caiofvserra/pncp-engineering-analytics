"""Pipeline autossuficiente: importação inicial, classificação, veredito LLM e
ingestão contínua (mensal) do PNCP. Reproduz, dentro do sistema, o que o
notebook fazia offline — mas de forma leve, para uso institucional."""
import datetime as dt
import numpy as np
import pandas as pd
import requests
from . import config, db, classifier, llm

# ── base de DEMONSTRAÇÃO (usada se não houver arquivo de importação) ────────
_DEMO_REF = [  # positivos: engenharia/obras já rotulados pelo órgão
    ("e01", "EXECUÇÃO DE OBRA DE CONSTRUÇÃO DA NOVA ESCOLA MUNICIPAL", "engenharia"),
    ("e02", "REFORMA GERAL DO PRÉDIO ADMINISTRATIVO COM SUBSTITUIÇÃO DE ESQUADRIAS", "obras"),
    ("e03", "PAVIMENTAÇÃO ASFÁLTICA E DRENAGEM DE VIAS URBANAS", "obras"),
    ("e04", "CONSTRUÇÃO DE PONTE DE CONCRETO SOBRE O CÓRREGO", "engenharia"),
    ("e05", "REFORMA E AMPLIAÇÃO DO HOSPITAL REGIONAL", "obras"),
    ("e06", "ELABORAÇÃO DE PROJETO ESTRUTURAL E EXECUTIVO DO TERMINAL", "engenharia"),
    ("e07", "RECAPEAMENTO E SINALIZAÇÃO VIÁRIA DE RODOVIA MUNICIPAL", "obras"),
    ("e08", "CONSTRUÇÃO DE MURO DE ARRIMO E CONTENÇÃO DE ENCOSTA", "engenharia"),
]
_DEMO_GERAL = [
    ("c01", "RECAPEAMENTO ASFÁLTICO DE VIAS NO DISTRITO INDUSTRIAL", "Prefeitura Municipal", 3400000),
    ("c02", "REFORMA E AMPLIAÇÃO DA UNIDADE BÁSICA DE SAÚDE DO JARDIM SÃO JORGE", "Secretaria de Saúde", 1250000),
    ("c03", "CONTRATAÇÃO DE EMPRESA PARA TROCA DE PISO E REVESTIMENTO DA ESCOLA", "Secretaria de Educação", 480000),
    ("c04", "SERVIÇOS DE MANUTENÇÃO DE VEÍCULOS DA FROTA OFICIAL", "Departamento de Transportes", 220000),
    ("c05", "ELABORAÇÃO DE PROJETO ELÉTRICO E LAUDO TÉCNICO DA CRECHE MUNICIPAL", "Secretaria de Obras", 130000),
    ("c06", "ESPETÁCULO MUSICAL E SONORIZAÇÃO PARA A FESTA DA CIDADE", "Secretaria de Cultura", 260000),
    ("c07", "IMPERMEABILIZAÇÃO DA LAJE E COBERTURA DO GINÁSIO MUNICIPAL", "Secretaria de Esportes", 260000),
    ("c08", "AQUISIÇÃO DE VALE-TRANSPORTE PARA OS SERVIDORES", "Secretaria de Administração", 95000),
    ("c09", "REFORMA DOS SANITÁRIOS E INSTALAÇÕES HIDRÁULICAS DA ESCOLA ESTADUAL", "Secretaria de Educação", 340000),
    ("c10", "MANUTENÇÃO ELÉTRICA PREDIAL NOS SETORES DA PREFEITURA", "Prefeitura Municipal", 175000),
    ("c11", "PODA DE ÁRVORES E JARDINAGEM EM PRAÇAS E ÁREAS VERDES", "Secretaria de Meio Ambiente", 180000),
    ("c12", "CONSTRUÇÃO DE MURO DE ARRIMO E DRENAGEM NA ENCOSTA DO BAIRRO", "Prefeitura Municipal", 890000),
]


def _limiar():
    try:
        return float(db.get_config().get("limiar", 0.6))
    except Exception:
        return 0.6


def importar_inicial():
    """Carrega a base já baixada (uma vez). Reconhece parquet/csv com colunas
    objeto e categoria (+ orgao, valor, id, uf). Sem arquivo, usa demonstração."""
    if db.n_contratos() > 0:
        return "base já carregada"
    rows = []
    p = config.IMPORT_FILE
    try:
        df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
    except Exception:
        df = None
    if df is not None and len(df):
        col = {c.lower(): c for c in df.columns}
        obj = col.get("objeto") or col.get("text") or col.get("objetocontrato")
        cat = col.get("categoria") or col.get("rotulo")
        for i, r in df.iterrows():
            categoria = str(r[cat]).lower() if cat else "geral"
            if categoria not in ("engenharia", "obras", "geral"):
                categoria = "geral"
            rows.append(dict(
                id=str(r[col["numerocontrolepncp"]]) if "numerocontrolepncp" in col else f"imp{i}",
                objeto=str(r[obj])[:500] if obj else "",
                orgao=str(r[col["razaosocialorgao"]]) if "razaosocialorgao" in col else "",
                valor=float(r[col["valor"]]) if "valor" in col and pd.notna(r[col["valor"]]) else None,
                uf=str(r[col["uf"]]) if "uf" in col else "",
                categoria=categoria, score=None,
                status="referencia" if categoria != "geral" else "novo", origem="import"))
        origem = f"arquivo {p} ({len(rows)} contratos)"
    else:
        for cid, obj, cat in _DEMO_REF:
            rows.append(dict(id=cid, objeto=obj, orgao="", valor=None, uf="SP",
                             categoria=cat, score=None, status="referencia", origem="demo"))
        for cid, obj, org, val in _DEMO_GERAL:
            rows.append(dict(id=cid, objeto=obj, orgao=org, valor=val, uf="SP",
                             categoria="geral", score=None, status="novo", origem="demo"))
        origem = "DEMONSTRAÇÃO (arquivo de importação não encontrado)"
    db.upsert(rows)
    db.evento("import", f"Base inicial: {origem}")
    return origem


def pontuar_gerais(rodar_llm=None):
    """Classifica os 'serviços gerais', define o status pela suspeita e,
    opcionalmente, roda o veredito da LLM nos suspeitos."""
    if not classifier.treinado():
        classifier.treinar()
    gerais = db.gerais_para_pontuar()
    if not gerais:
        return 0
    probs = classifier.score([g["objeto"] for g in gerais])
    lim = _limiar()
    n_susp = 0
    cfg = db.get_config()
    if rodar_llm is None:
        rodar_llm = cfg.get("llm_ativo") == "1" and cfg.get("llm_auto_veredito") == "1"
    usar_llm = rodar_llm and llm.disponivel()
    for g, p in zip(gerais, probs):
        suspeito = p >= lim
        db.set_score(g["id"], float(p), "novo" if suspeito else "baixa")
        if suspeito:
            n_susp += 1
            if usar_llm:
                v = llm.veredito(g["objeto"])
                if v:
                    db.set_llm(g["id"], v.get("classe", ""), str(v.get("motivo", ""))[:200])
    db.evento("classificacao", f"{len(gerais)} gerais pontuados; {n_susp} suspeitos "
              f"(limiar {lim})" + ("; veredito LLM aplicado." if usar_llm else "."))
    return n_susp


def ingerir_pncp(paginas=3, tam=50):
    """Ingestão contínua: baixa contratos recentes do PNCP, salva os 'serviços
    gerais' novos e os classifica. Tolerante a falhas de rede."""
    cfg = db.get_config(); uf = cfg.get("ingest_uf", "SP")
    hoje = dt.date.today()
    dias = int(cfg.get("ingest_intervalo_dias", 30))
    ini = (hoje - dt.timedelta(days=dias)).strftime("%Y%m%d"); fim = hoje.strftime("%Y%m%d")
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
                cat_id = (c.get("categoriaProcesso") or {}).get("id")
                categoria = ("engenharia" if cat_id in config.CATEG_ENGENHARIA
                             else "geral" if cat_id in config.CATEG_GERAL else None)
                if categoria is None:
                    continue
                obj = (c.get("objetoContrato") or "").strip()
                if len(obj) < 20:
                    continue
                novos.append(dict(
                    id=str(c.get("numeroControlePNCP")), objeto=obj[:500],
                    orgao=(c.get("orgaoEntidade") or {}).get("razaosocial", ""),
                    valor=c.get("valorGlobal"), uf=uf, categoria=categoria, score=None,
                    status="referencia" if categoria != "geral" else "novo", origem="pncp"))
        except Exception:
            break
    if not novos:
        db.evento("ingestao", "Nenhum contrato novo do PNCP (ou API indisponível).")
        return 0
    db.upsert(novos)
    n = pontuar_gerais()
    db.evento("ingestao", f"{len(novos)} contratos novos do PNCP; {n} suspeitos à triagem.")
    return len(novos)
