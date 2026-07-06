"""Agendador em background (APScheduler). Reconfigurável em tempo de execução.
Roda dentro do mesmo processo do FastAPI — barato, sem worker externo."""
from apscheduler.schedulers.background import BackgroundScheduler
from . import db, learning, ingest

_sched: BackgroundScheduler | None = None


def _tick_retrain_tempo():
    if db.get_config().get("retrain_modo") == "por_tempo":
        learning.retreinar(motivo="agendado (tempo)")


def _tick_ingest():
    if db.get_config().get("ingest_ativo") == "1":
        n = ingest.ingerir_pncp()
        if n:
            db.log_retrain(0, "ingest", f"{n} novos contratos suspeitos do PNCP")


def iniciar():
    global _sched
    if _sched:
        return
    cfg = db.get_config()
    _sched = BackgroundScheduler(daemon=True)
    _sched.add_job(_tick_retrain_tempo, "interval",
                   minutes=max(5, int(cfg.get("retrain_intervalo_min", 1440))),
                   id="retrain", replace_existing=True)
    _sched.add_job(_tick_ingest, "interval",
                   minutes=max(5, int(cfg.get("ingest_intervalo_min", 1440))),
                   id="ingest", replace_existing=True)
    _sched.start()


def reconfigurar():
    """Reaplica os intervalos após o usuário salvar a config na UI."""
    if not _sched:
        return
    cfg = db.get_config()
    _sched.reschedule_job("retrain", trigger="interval",
                          minutes=max(5, int(cfg.get("retrain_intervalo_min", 1440))))
    _sched.reschedule_job("ingest", trigger="interval",
                          minutes=max(5, int(cfg.get("ingest_intervalo_min", 1440))))
