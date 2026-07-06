"""Loop de aprendizado: aplica os feedbacks pendentes ao modelo online.

Regra de rótulo: o funcionário revisa um caso SUSPEITO de subenquadramento.
  - concorda=1  → é engenharia mal enquadrada  → rótulo positivo (1)
  - concorda=0  → é serviço comum de verdade    → rótulo negativo (0)

Robusto: grava no histórico o resultado; nunca lança para o chamador.
"""
from . import db, model, ingest


def precisa_retreinar() -> bool:
    cfg = db.get_config()
    if cfg.get("retrain_modo") == "por_feedbacks":
        try:
            return db.n_feedbacks_novos() >= int(cfg.get("retrain_n_feedbacks", 25))
        except Exception:
            return False
    return False   # modo "por_tempo" é disparado pelo agendador


def retreinar(motivo="manual") -> dict:
    fbs = db.feedbacks_nao_aplicados()
    if not fbs:
        db.log_retrain(0, "vazio", f"{motivo}: sem feedback novo")
        return {"ok": True, "aplicados": 0, "msg": "Sem feedback novo."}
    textos = [f["objeto"] for f in fbs]
    labels = [int(f["concorda"]) for f in fbs]
    peso = float(db.get_config().get("peso_feedback", 3))
    n = model.incrementa(textos, labels, peso)
    if n < 0:
        db.log_retrain(len(fbs), "erro", f"{motivo}: falha no incrementa()")
        return {"ok": False, "aplicados": 0, "msg": "Falha no re-treino (modelo anterior mantido)."}
    db.marca_feedbacks_aplicados([f["id"] for f in fbs])
    repont = ingest.repontuar_fila()
    db.log_retrain(n, "ok", f"{motivo}: {n} feedbacks; fila repontuada ({repont})")
    return {"ok": True, "aplicados": n, "repontuados": repont,
            "msg": f"Modelo atualizado com {n} feedbacks; fila repontuada."}
