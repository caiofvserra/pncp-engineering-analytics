"""Camada de dados em SQLite (arquivo único, sem servidor). Barato e robusto.
Cada função abre e fecha sua própria conexão (thread-safe para o uso do app)."""
import sqlite3
from contextlib import contextmanager
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS contratos (
    id TEXT PRIMARY KEY,                 -- numeroControlePNCP
    objeto TEXT NOT NULL,
    orgao TEXT, valor REAL,
    tipo_eng TEXT,
    prob_base REAL,                      -- probabilidade vinda do notebook
    score REAL,                          -- score final (base + modelo online)
    status TEXT DEFAULT 'pendente',      -- pendente | revisado
    origem TEXT DEFAULT 'notebook',      -- notebook | pncp | demo
    criado_em TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_status_score ON contratos(status, score DESC);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contrato_id TEXT NOT NULL,
    usuario TEXT,
    concorda INTEGER NOT NULL,           -- 1 = é subenquadramento | 0 = não é
    rito_ok INTEGER,                     -- rito seguido? (opcional)
    justificativa TEXT,
    aplicado_no_modelo INTEGER DEFAULT 0,
    criado_em TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (contrato_id) REFERENCES contratos(id)
);

CREATE TABLE IF NOT EXISTS config (
    chave TEXT PRIMARY KEY, valor TEXT
);

CREATE TABLE IF NOT EXISTS retrain_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quando TEXT DEFAULT (datetime('now')),
    n_feedbacks INTEGER, status TEXT, detalhe TEXT
);
"""


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL;")   # concorrência leitura/escrita robusta
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)
        for k, v in config.CONFIG_PADRAO.items():
            c.execute("INSERT OR IGNORE INTO config(chave, valor) VALUES (?,?)", (k, v))


# ── Config ──────────────────────────────────────────────────────────────
def get_config():
    with conn() as c:
        return {r["chave"]: r["valor"] for r in c.execute("SELECT * FROM config")}


def set_config(d: dict):
    with conn() as c:
        for k, v in d.items():
            c.execute("UPDATE config SET valor=? WHERE chave=?", (str(v), k))


# ── Contratos ───────────────────────────────────────────────────────────
def upsert_contratos(rows: list[dict]):
    with conn() as c:
        for r in rows:
            c.execute(
                """INSERT INTO contratos(id,objeto,orgao,valor,tipo_eng,prob_base,score,origem)
                   VALUES(:id,:objeto,:orgao,:valor,:tipo_eng,:prob_base,:score,:origem)
                   ON CONFLICT(id) DO UPDATE SET
                     score=excluded.score, prob_base=excluded.prob_base""", r)


def fila(limit=50, offset=0):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM contratos WHERE status='pendente' "
            "ORDER BY score DESC LIMIT ? OFFSET ?", (limit, offset))]


def contrato(cid):
    with conn() as c:
        r = c.execute("SELECT * FROM contratos WHERE id=?", (cid,)).fetchone()
        return dict(r) if r else None


def marca_revisado(cid):
    with conn() as c:
        c.execute("UPDATE contratos SET status='revisado' WHERE id=?", (cid,))


def n_contratos():
    with conn() as c:
        return c.execute("SELECT COUNT(*) n FROM contratos").fetchone()["n"]


# ── Feedback ────────────────────────────────────────────────────────────
def add_feedback(cid, usuario, concorda, rito_ok, justificativa):
    with conn() as c:
        c.execute(
            "INSERT INTO feedback(contrato_id,usuario,concorda,rito_ok,justificativa)"
            " VALUES(?,?,?,?,?)", (cid, usuario, concorda, rito_ok, justificativa))


def feedbacks_nao_aplicados():
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT f.*, c.objeto FROM feedback f JOIN contratos c ON c.id=f.contrato_id"
            " WHERE f.aplicado_no_modelo=0")]


def marca_feedbacks_aplicados(ids):
    if not ids:
        return
    with conn() as c:
        c.executemany("UPDATE feedback SET aplicado_no_modelo=1 WHERE id=?",
                      [(i,) for i in ids])


def n_feedbacks_novos():
    with conn() as c:
        return c.execute(
            "SELECT COUNT(*) n FROM feedback WHERE aplicado_no_modelo=0").fetchone()["n"]


# ── Estatísticas / histórico ────────────────────────────────────────────
def stats():
    with conn() as c:
        g = lambda q, *a: c.execute(q, a).fetchone()[0]
        total = g("SELECT COUNT(*) FROM contratos")
        pend = g("SELECT COUNT(*) FROM contratos WHERE status='pendente'")
        rev = total - pend
        conc = g("SELECT COUNT(*) FROM feedback WHERE concorda=1")
        disc = g("SELECT COUNT(*) FROM feedback WHERE concorda=0")
        val_conf = g("SELECT COALESCE(SUM(c.valor),0) FROM feedback f "
                     "JOIN contratos c ON c.id=f.contrato_id WHERE f.concorda=1") or 0
        por_tipo = [dict(r) for r in c.execute(
            "SELECT COALESCE(tipo_eng,'(sem tipo)') tipo, COUNT(*) n "
            "FROM contratos WHERE status='pendente' GROUP BY tipo ORDER BY n DESC LIMIT 8")]
        return {"total": total, "pendentes": pend, "revisados": rev,
                "concordancias": conc, "discordancias": disc,
                "valor_confirmado": val_conf, "por_tipo": por_tipo,
                "feedbacks_novos": n_feedbacks_novos()}


def log_retrain(n, status, detalhe=""):
    with conn() as c:
        c.execute("INSERT INTO retrain_log(n_feedbacks,status,detalhe) VALUES(?,?,?)",
                  (n, status, detalhe))


def historico(limit=30):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM retrain_log ORDER BY id DESC LIMIT ?", (limit,))]
