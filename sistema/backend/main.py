"""API FastAPI + serviço do frontend estático. Ponto de entrada do sistema.

Rodar:  uvicorn sistema.backend.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import config, db, ingest, learning, scheduler

app = FastAPI(title="Monitor de Subenquadramento de Engenharia — PNCP")


@app.on_event("startup")
def _startup():
    db.init()
    origem = ingest.semear()
    scheduler.iniciar()
    print(f"[startup] fila semeada de: {origem}")


# ── Modelos de entrada ──────────────────────────────────────────────────
class FeedbackIn(BaseModel):
    contrato_id: str
    concorda: bool                 # True = concordo que é subenquadramento
    rito_ok: bool | None = None    # o rito de engenharia foi seguido?
    justificativa: str = ""
    usuario: str = "anônimo"


class ConfigIn(BaseModel):
    dados: dict


# ── Endpoints ───────────────────────────────────────────────────────────
@app.get("/api/fila")
def get_fila(limit: int = 30, offset: int = 0):
    return {"itens": db.fila(limit, offset)}


@app.get("/api/contrato/{cid}")
def get_contrato(cid: str):
    c = db.contrato(cid)
    if not c:
        raise HTTPException(404, "contrato não encontrado")
    return c


@app.post("/api/feedback")
def post_feedback(f: FeedbackIn):
    if not db.contrato(f.contrato_id):
        raise HTTPException(404, "contrato não encontrado")
    # 1) persiste SEMPRE primeiro (nada se perde, mesmo se o re-treino falhar)
    db.add_feedback(f.contrato_id, f.usuario, int(f.concorda),
                    None if f.rito_ok is None else int(f.rito_ok), f.justificativa)
    db.marca_revisado(f.contrato_id)
    # 2) re-treina se a política de frequência disparar
    retreino = None
    if learning.precisa_retreinar():
        retreino = learning.retreinar(motivo="limite de feedbacks atingido")
    return {"ok": True, "retreino": retreino}


@app.get("/api/stats")
def get_stats():
    return db.stats()


@app.get("/api/config")
def get_config():
    return db.get_config()


@app.post("/api/config")
def post_config(c: ConfigIn):
    db.set_config(c.dados)
    scheduler.reconfigurar()
    return {"ok": True, "config": db.get_config()}


@app.post("/api/retrain")
def post_retrain():
    return learning.retreinar(motivo="manual (botão)")


@app.post("/api/ingest")
def post_ingest():
    n = ingest.ingerir_pncp()
    return {"ok": True, "novos": n}


@app.get("/api/historico")
def get_historico():
    return {"itens": db.historico()}


# ── Frontend estático ───────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=config.FRONTEND_DIR), name="static")
