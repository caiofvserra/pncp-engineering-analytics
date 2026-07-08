"""API FastAPI + frontend — sistema AUTOSSUFICIENTE (independe do notebook).
Etapas dentro do próprio sistema: importar base → classificar → veredito LLM →
triagem humana → análise de rito (baixa PDFs) → ingestão contínua (mensal).

Rodar (em sistema/):  uvicorn backend.main:app --reload
"""
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import config, db, pipeline, classifier, learning, rito, llm, scheduler

app = FastAPI(title="Monitor de Subenquadramento de Engenharia — PNCP")


@app.on_event("startup")
def _startup():
    db.init()
    origem = pipeline.importar_inicial()
    classifier.treinar()          # treina o classificador do sistema
    pipeline.pontuar_gerais()     # classifica os 'serviços gerais'
    scheduler.iniciar()
    print(f"[startup] {origem}")


class TriagemIn(BaseModel):
    contrato_id: str
    eh_eng: bool
    justificativa: str = ""
    usuario: str = "revisor"


class RitoVeredito(BaseModel):
    contrato_id: str
    rito_seguido: bool
    usuario: str = "revisor"


class ConfigIn(BaseModel):
    dados: dict


@app.get("/api/stats")
def stats():
    s = db.stats()
    s["modelo_treinado"] = classifier.treinado()
    s["llm_disponivel"] = llm.disponivel()
    return s


@app.get("/api/ranking")
def ranking(limit: int = 100, offset: int = 0):
    return {"itens": db.listar(categoria=["geral"], limit=limit, offset=offset)}


# ── Triagem ───────────────────────────────────────────────────────────────
@app.get("/api/triagem/fila")
def triagem_fila(limit: int = 30, offset: int = 0):
    return {"itens": db.listar(status=["novo"], limit=limit, offset=offset)}


@app.post("/api/triagem")
def triagem(t: TriagemIn):
    if not db.contrato(t.contrato_id):
        raise HTTPException(404, "contrato não encontrado")
    db.add_triagem(t.contrato_id, t.usuario, int(t.eh_eng), t.justificativa)
    db.set_status(t.contrato_id, "aguarda_rito" if t.eh_eng else "triagem_nao")
    retreino = learning.retreinar("limite atingido") if learning.precisa_retreinar() else None
    return {"ok": True, "ao_rito": bool(t.eh_eng), "retreino": retreino}


@app.post("/api/veredito/{cid}")
def veredito_llm(cid: str):
    """Roda o veredito da LLM para um contrato (opcional/on-demand)."""
    c = db.contrato(cid)
    if not c:
        raise HTTPException(404, "contrato não encontrado")
    if not llm.disponivel():
        raise HTTPException(400, "LLM desativada ou fora do ar (ver Configurações).")
    v = llm.veredito(c["objeto"]) or {}
    db.set_llm(cid, v.get("classe", ""), str(v.get("motivo", ""))[:200])
    return {"ok": True, "veredito": v}


# ── Rito ──────────────────────────────────────────────────────────────────
@app.get("/api/rito/fila")
def rito_fila(limit: int = 30):
    itens = db.listar(status=["aguarda_rito"], limit=limit)
    for it in itens:
        it["rito"] = db.get_rito(it["id"])
    return {"itens": itens}


@app.post("/api/rito/analisar/{cid}")
def rito_analisar(cid: str):
    c = db.contrato(cid)
    if not c:
        raise HTTPException(404, "contrato não encontrado")
    dados = rito.analisar(c, int(db.get_config().get("rito_max_docs", 3)))
    db.salva_rito(cid, dados)
    db.evento("rito", f"Analisado {cid}: {dados['mk_score']} marcadores, {dados['n_docs']} doc(s).")
    return {"ok": True, "rito": {**dados, "marcadores": json.loads(dados["marcadores"])}}


@app.post("/api/rito/veredito")
def rito_veredito(v: RitoVeredito):
    if not db.get_rito(v.contrato_id):
        raise HTTPException(400, "analise o documento antes de dar o veredito")
    db.veredito_rito(v.contrato_id, int(v.rito_seguido), v.usuario)
    novo = "rito_seguido" if v.rito_seguido else "subenq_real"
    db.set_status(v.contrato_id, novo)
    db.evento("veredito", f"{v.contrato_id}: "
              + ("rótulo incorreto, processo ok" if v.rito_seguido
                 else "SUBENQUADRAMENTO REAL (rito não seguido)"))
    return {"ok": True, "status": novo}


# ── Modelo / manutenção ───────────────────────────────────────────────────
@app.post("/api/modelo/treinar")
def modelo_treinar():
    return learning.retreinar("manual (botão)")


@app.post("/api/ingest")
def ingest_manual():
    return {"ok": True, "novos": pipeline.ingerir_pncp()}


@app.get("/api/config")
def get_config():
    return db.get_config()


@app.post("/api/config")
def set_config(c: ConfigIn):
    db.set_config(c.dados)
    scheduler.reconfigurar()
    db.evento("config", "Configurações atualizadas.")
    return {"ok": True, "config": db.get_config()}


@app.get("/api/historico")
def historico():
    return {"itens": db.eventos()}


# ── Frontend ──────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=config.FRONTEND_DIR), name="static")
