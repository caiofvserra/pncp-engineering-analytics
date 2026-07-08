"""Aprendizado com o humano no loop: a cada N triagens, RE-TREINA o classificador
do sistema (positivos do órgão + amostra de gerais + rótulos humanos com peso) e
RE-PONTUA a fila. Nunca lança para o chamador."""
from . import db, classifier, pipeline

_APLICADAS = "_triagens_aplicadas"   # marca simples no config


def _n_desde_ultimo():
    cfg = db.get_config()
    return db.n_triagens() - int(cfg.get(_APLICADAS, "0") or 0)


def precisa_retreinar():
    cfg = db.get_config()
    if cfg.get("retrain_modo") == "por_feedbacks":
        try:
            return _n_desde_ultimo() >= int(cfg.get("retrain_n_feedbacks", 20))
        except Exception:
            return False
    return False


def retreinar(motivo="manual"):
    met = classifier.treinar()
    if not met.get("ok"):
        return {"ok": False, "msg": met.get("msg", "Falha no treino.")}
    n_susp = pipeline.pontuar_gerais()
    db.set_config({_APLICADAS: str(db.n_triagens())})
    db.evento("aprendizado", f"{motivo}: modelo re-treinado; {n_susp} suspeitos na fila.")
    return {"ok": True, "aplicados": met["n_humano"],
            "msg": f"Modelo re-treinado ({met['n_pos']} pos / {met['n_neg']} neg, "
                   f"+{met['n_humano']} humanos); fila repontuada."}
