"""Agendador em background (APScheduler), no mesmo processo do FastAPI.
- Ingestão contínua do PNCP (mensal por padrão) → classifica novos suspeitos.
- Re-treino por tempo (quando o modo é 'por_tempo')."""
from apscheduler.schedulers.background import BackgroundScheduler
from . import db, learning, pipeline

_sched = None


def _tick_ingest():
    if db.get_config().get("ingest_ativo") == "1":
        pipeline.ingerir_pncp()


def _tick_retrain():
    if db.get_config().get("retrain_modo") == "por_tempo":
        learning.retreinar("agendado (tempo)")


def iniciar():
    global _sched
    if _sched:
        return
    cfg = db.get_config()
    _sched = BackgroundScheduler(daemon=True)
    _sched.add_job(_tick_ingest, "interval",
                   days=max(1, int(cfg.get("ingest_intervalo_dias", 30))),
                   id="ingest", replace_existing=True)
    _sched.add_job(_tick_retrain, "interval",
                   minutes=max(5, int(cfg.get("retrain_intervalo_min", 1440))),
                   id="retrain", replace_existing=True)
    _sched.start()


def reconfigurar():
    if not _sched:
        return
    cfg = db.get_config()
    _sched.reschedule_job("ingest", trigger="interval",
                          days=max(1, int(cfg.get("ingest_intervalo_dias", 30))))
    _sched.reschedule_job("retrain", trigger="interval",
                          minutes=max(5, int(cfg.get("retrain_intervalo_min", 1440))))
