"""Camada de dados SQLite. Modela o ciclo de vida que espelha o notebook:

  ranking → TRIAGEM (é engenharia?) → ANÁLISE DE RITO (baixa PDFs) → veredito

status do contrato:
  novo            -> aguardando triagem do objeto (etapa 8/10 do notebook)
  triagem_nao     -> revisor disse que é serviço comum (fim; alimenta o modelo)
  aguarda_rito    -> revisor confirmou engenharia -> vai para a fila de rito (etapa 11)
  rito_seguido    -> documento tem o rito -> rótulo incorreto, mas processo correto
  subenq_real     -> documento NÃO tem o rito -> subenquadramento real (o achado!)
  rito_indeterminado -> não foi possível obter/ler o documento
"""
import sqlite3
from contextlib import contextmanager
from . import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS contratos (
    id TEXT PRIMARY KEY, objeto TEXT NOT NULL, orgao TEXT, valor REAL,
    tipo_eng TEXT, prob_base REAL, score REAL,
    status TEXT DEFAULT 'novo', origem TEXT DEFAULT 'notebook',
    criado_em TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_status_score ON contratos(status, score DESC);

CREATE TABLE IF NOT EXISTS triagem (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contrato_id TEXT NOT NULL, usuario TEXT, eh_eng INTEGER NOT NULL,
    justificativa TEXT, aplicado_no_modelo INTEGER DEFAULT 0,
    criado_em TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rito (
    contrato_id TEXT PRIMARY KEY,
    ncp_compra TEXT, n_docs INTEGER, chars INTEGER,
    marcadores TEXT, mk_score INTEGER, trecho TEXT,
    rito_seguido INTEGER, usuario TEXT,
    criado_em TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS config (chave TEXT PRIMARY KEY, valor TEXT);

CREATE TABLE IF NOT EXISTS eventos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quando TEXT DEFAULT (datetime('now')), tipo TEXT, detalhe TEXT
);
"""


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL;")
    try:
        yield c; c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)
        for k, v in config.CONFIG_PADRAO.items():
            c.execute("INSERT OR IGNORE INTO config(chave,valor) VALUES(?,?)", (k, v))


# ── config ───────────────────────────────────────────────────────────────
def get_config():
    with conn() as c:
        return {r["chave"]: r["valor"] for r in c.execute("SELECT * FROM config")}


def set_config(d):
    with conn() as c:
        for k, v in d.items():
            c.execute("UPDATE config SET valor=? WHERE chave=?", (str(v), k))


# ── contratos ──────────────────────────────────────────────────────────
def upsert_contratos(rows):
    with conn() as c:
        for r in rows:
            c.execute(
                """INSERT INTO contratos(id,objeto,orgao,valor,tipo_eng,prob_base,score,origem)
                   VALUES(:id,:objeto,:orgao,:valor,:tipo_eng,:prob_base,:score,:origem)
                   ON CONFLICT(id) DO UPDATE SET score=excluded.score,
                     prob_base=excluded.prob_base""", r)


def contrato(cid):
    with conn() as c:
        r = c.execute("SELECT * FROM contratos WHERE id=?", (cid,)).fetchone()
        return dict(r) if r else None


def set_status(cid, status):
    with conn() as c:
        c.execute("UPDATE contratos SET status=? WHERE id=?", (status, cid))


def n_contratos():
    with conn() as c:
        return c.execute("SELECT COUNT(*) n FROM contratos").fetchone()["n"]


def listar(status=None, limit=50, offset=0, ordem="score DESC"):
    q = "SELECT * FROM contratos"
    a = []
    if status:
        q += " WHERE status IN (%s)" % ",".join("?" * len(status)); a += status
    q += f" ORDER BY {ordem} LIMIT ? OFFSET ?"; a += [limit, offset]
    with conn() as c:
        return [dict(r) for r in c.execute(q, a)]


def pendentes_score():
    """Contratos ainda na fila de triagem, para repontuar após re-treino."""
    return listar(status=["novo"], limit=1000000)


# ── triagem ────────────────────────────────────────────────────────────
def add_triagem(cid, usuario, eh_eng, justificativa):
    with conn() as c:
        c.execute("INSERT INTO triagem(contrato_id,usuario,eh_eng,justificativa)"
                  " VALUES(?,?,?,?)", (cid, usuario, eh_eng, justificativa))


def triagens_nao_aplicadas():
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT t.*, c.objeto, c.valor FROM triagem t JOIN contratos c "
            "ON c.id=t.contrato_id WHERE t.aplicado_no_modelo=0")]


def marca_triagens_aplicadas(ids):
    if not ids:
        return
    with conn() as c:
        c.executemany("UPDATE triagem SET aplicado_no_modelo=1 WHERE id=?",
                      [(i,) for i in ids])


def n_triagens_novas():
    with conn() as c:
        return c.execute("SELECT COUNT(*) n FROM triagem WHERE aplicado_no_modelo=0"
                         ).fetchone()["n"]


# ── rito ────────────────────────────────────────────────────────────────
def salva_rito(cid, dados):
    with conn() as c:
        c.execute("""INSERT INTO rito(contrato_id,ncp_compra,n_docs,chars,marcadores,
                       mk_score,trecho) VALUES(?,?,?,?,?,?,?)
                     ON CONFLICT(contrato_id) DO UPDATE SET
                       ncp_compra=excluded.ncp_compra,n_docs=excluded.n_docs,
                       chars=excluded.chars,marcadores=excluded.marcadores,
                       mk_score=excluded.mk_score,trecho=excluded.trecho""",
                  (cid, dados["ncp_compra"], dados["n_docs"], dados["chars"],
                   dados["marcadores"], dados["mk_score"], dados["trecho"]))


def get_rito(cid):
    with conn() as c:
        r = c.execute("SELECT * FROM rito WHERE contrato_id=?", (cid,)).fetchone()
        return dict(r) if r else None


def veredito_rito(cid, rito_seguido, usuario):
    with conn() as c:
        c.execute("UPDATE rito SET rito_seguido=?, usuario=? WHERE contrato_id=?",
                  (rito_seguido, usuario, cid))


# ── estatísticas / eventos ───────────────────────────────────────────────
def stats():
    with conn() as c:
        g = lambda q, *a: c.execute(q, a).fetchone()[0]
        st = lambda s: g("SELECT COUNT(*) FROM contratos WHERE status=?", s)
        subenq_val = g("SELECT COALESCE(SUM(valor),0) FROM contratos WHERE status='subenq_real'") or 0
        por_tipo = [dict(r) for r in c.execute(
            "SELECT COALESCE(NULLIF(tipo_eng,''),'(sem tipo)') tipo, COUNT(*) n "
            "FROM contratos WHERE status='novo' GROUP BY tipo ORDER BY n DESC LIMIT 8")]
        return {
            "total": g("SELECT COUNT(*) FROM contratos"),
            "novos": st("novo"), "aguarda_rito": st("aguarda_rito"),
            "subenq_real": st("subenq_real"), "rito_seguido": st("rito_seguido"),
            "triagem_nao": st("triagem_nao"),
            "rito_indeterminado": st("rito_indeterminado"),
            "valor_subenq": subenq_val, "por_tipo": por_tipo,
            "triagens_novas": n_triagens_novas(),
        }


def evento(tipo, detalhe=""):
    with conn() as c:
        c.execute("INSERT INTO eventos(tipo,detalhe) VALUES(?,?)", (tipo, detalhe))


def eventos(limit=40):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM eventos ORDER BY id DESC LIMIT ?", (limit,))]
