"""API FastAPI + frontend. Abas espelham o pipeline do notebook:
Painel · Ranking · Triagem · Análise de rito · Configurações · Histórico.

Rodar: uvicorn backend.main:app --reload   (dentro de sistema/)
"""
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import config, db, ingest, learning, rito, scheduler

app = FastAPI(title="Monitor de Subenquadramento de Engenharia — PNCP")


@app.on_event("startup")
def _startup():
    db.init()
    origem = ingest.semear()
    scheduler.iniciar()
    db.evento("sistema", f"Iniciado — fila semeada de: {origem}")
    print(f"[startup] {origem}")


class TriagemIn(BaseModel):
    contrato_id: str
    eh_eng: bool                 # True = é engenharia (subenquadramento)
    justificativa: str = ""
    usuario: str = "revisor"


class RitoVeredito(BaseModel):
    contrato_id: str
    rito_seguido: bool           # o rito de engenharia foi observado no documento?
    usuario: str = "revisor"


class ConfigIn(BaseModel):
    dados: dict


# ── Painel ───────────────────────────────────────────────────────────────
@app.get("/api/stats")
def stats():
    return db.stats()


# ── Ranking (etapa 7: saída do modelo) ───────────────────────────────────
@app.get("/api/ranking")
def ranking(limit: int = 50, offset: int = 0):
    return {"itens": db.listar(limit=limit, offset=offset)}


# ── Triagem do objeto (etapas 8/10) ──────────────────────────────────────
@app.get("/api/triagem/fila")
def triagem_fila(limit: int = 30, offset: int = 0):
    return {"itens": db.listar(status=["novo"], limit=limit, offset=offset)}


@app.post("/api/triagem")
def triagem(t: TriagemIn):
    c = db.contrato(t.contrato_id)
    if not c:
        raise HTTPException(404, "contrato não encontrado")
    db.add_triagem(t.contrato_id, t.usuario, int(t.eh_eng), t.justificativa)
    # é engenharia -> vai para a fila de rito; não é -> encerra
    db.set_status(t.contrato_id, "aguarda_rito" if t.eh_eng else "triagem_nao")
    retreino = learning.retreinar("limite atingido") if learning.precisa_retreinar() else None
    return {"ok": True, "encaminhado_ao_rito": bool(t.eh_eng), "retreino": retreino}


# ── Análise de rito (etapa 11 — posterior, baixa PDFs) ───────────────────
@app.get("/api/rito/fila")
def rito_fila(limit: int = 30):
    itens = db.listar(status=["aguarda_rito"], limit=limit)
    for it in itens:
        it["rito"] = db.get_rito(it["id"])   # evidências, se já analisado
    return {"itens": itens}


@app.post("/api/rito/analisar/{cid}")
def rito_analisar(cid: str):
    """Baixa os documentos da licitação e detecta os marcadores do rito."""
    c = db.contrato(cid)
    if not c:
        raise HTTPException(404, "contrato não encontrado")
    max_docs = int(db.get_config().get("rito_max_docs", 3))
    dados = rito.analisar(c, max_docs)
    db.salva_rito(cid, dados)
    db.evento("rito", f"Analisado {cid}: {dados['mk_score']} marcadores, "
              f"{dados['n_docs']} doc(s).")
    return {"ok": True, "rito": {**dados, "marcadores": json.loads(dados["marcadores"])}}


@app.post("/api/rito/veredito")
def rito_veredito(v: RitoVeredito):
    r = db.get_rito(v.contrato_id)
    if not r:
        raise HTTPException(400, "analise o documento antes de dar o veredito")
    db.veredito_rito(v.contrato_id, int(v.rito_seguido), v.usuario)
    novo = "rito_seguido" if v.rito_seguido else "subenq_real"
    db.set_status(v.contrato_id, novo)
    db.evento("veredito", f"{v.contrato_id}: "
              + ("rótulo incorreto, processo correto" if v.rito_seguido
                 else "SUBENQUADRAMENTO REAL (rito não seguido)"))
    return {"ok": True, "status": novo}


# ── Configurações / histórico / manutenção ───────────────────────────────
@app.get("/api/config")
def get_config():
    return db.get_config()


@app.post("/api/config")
def set_config(c: ConfigIn):
    db.set_config(c.dados)
    scheduler.reconfigurar()
    db.evento("config", "Configurações atualizadas.")
    return {"ok": True, "config": db.get_config()}


@app.post("/api/retrain")
def retrain():
    return learning.retreinar("manual (botão)")


@app.post("/api/ingest")
def ingest_manual():
    return {"ok": True, "novos": ingest.ingerir_pncp()}


@app.get("/api/historico")
def historico():
    return {"itens": db.eventos()}


# ── Frontend ─────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=config.FRONTEND_DIR), name="static")
