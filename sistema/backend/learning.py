"""Loop de aprendizado — aprende com a TRIAGEM do objeto (não com o rito).

  eh_eng=1 (revisor confirmou engenharia) -> rótulo positivo (1)
  eh_eng=0 (revisor disse serviço comum)  -> rótulo negativo (0)

Robusto: nunca lança; troca atômica do modelo em model.incrementa()."""
from . import db, model, ingest


def precisa_retreinar():
    cfg = db.get_config()
    if cfg.get("retrain_modo") == "por_feedbacks":
        try:
            return db.n_triagens_novas() >= int(cfg.get("retrain_n_feedbacks", 20))
        except Exception:
            return False
    return False


def retreinar(motivo="manual"):
    tri = db.triagens_nao_aplicadas()
    if not tri:
        db.evento("aprendizado", f"{motivo}: sem triagem nova")
        return {"ok": True, "aplicados": 0, "msg": "Sem triagem nova."}
    textos = [t["objeto"] for t in tri]
    labels = [int(t["eh_eng"]) for t in tri]
    peso = float(db.get_config().get("peso_feedback", 3))
    n = model.incrementa(textos, labels, peso)
    if n < 0:
        db.evento("aprendizado", f"{motivo}: falha no re-treino")
        return {"ok": False, "aplicados": 0, "msg": "Falha no re-treino (modelo anterior mantido)."}
    db.marca_triagens_aplicadas([t["id"] for t in tri])
    rep = ingest.repontuar()
    db.evento("aprendizado", f"{motivo}: {n} triagens; ranking repontuado ({rep}).")
    return {"ok": True, "aplicados": n, "msg": f"Modelo atualizado com {n} triagens; ranking repontuado."}
