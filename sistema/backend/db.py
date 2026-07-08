"""SQLite — sistema autossuficiente. Guarda TODOS os contratos (com a categoria
declarada pelo órgão) e conduz os 'serviços gerais' pelo fluxo:

  classificação → TRIAGEM (é engenharia?) → ANÁLISE DE RITO (baixa PDFs) → veredito

status:
  referencia          -> contrato de engenharia/obras (positivo de treino; não revisado)
  baixa               -> 'geral' com score abaixo do limiar (fora da fila)
  novo                -> 'geral' suspeito, aguardando triagem
  triagem_nao         -> revisor: serviço comum (negativo de treino)
  aguarda_rito        -> revisor: engenharia -> fila de rito
  rito_seguido        -> rito presente no documento (rótulo incorreto, processo ok)
  subenq_real         -> rito ausente (SUBENQUADRAMENTO REAL)
  rito_indeterminado  -> documento não obtido/ilegível
"""
import sqlite3
from contextlib import contextmanager
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS contratos (
    id TEXT PRIMARY KEY, objeto TEXT NOT NULL, orgao TEXT, valor REAL, uf TEXT,
    categoria TEXT DEFAULT 'geral',          -- engenharia | obras | geral
    score REAL, status TEXT DEFAULT 'novo', origem TEXT DEFAULT 'import',
    llm_classe TEXT, llm_motivo TEXT,
    criado_em TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_status_score ON contratos(status, score DESC);
CREATE INDEX IF NOT EXISTS ix_categoria ON contratos(categoria);

CREATE TABLE IF NOT EXISTS triagem (
    id INTEGER PRIMARY KEY AUTOINCREMENT, contrato_id TEXT NOT NULL, usuario TEXT,
    eh_eng INTEGER NOT NULL, justificativa TEXT,
    criado_em TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rito (
    contrato_id TEXT PRIMARY KEY, ncp_compra TEXT, n_docs INTEGER, chars INTEGER,
    marcadores TEXT, mk_score INTEGER, trecho TEXT, llm_rito TEXT,
    rito_seguido INTEGER, usuario TEXT, criado_em TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS config (chave TEXT PRIMARY KEY, valor TEXT);
CREATE TABLE IF NOT EXISTS eventos (
    id INTEGER PRIMARY KEY AUTOINCREMENT, quando TEXT DEFAULT (datetime('now')),
    tipo TEXT, detalhe TEXT
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


# ── config / eventos ──────────────────────────────────────────────────────
def get_config():
    with conn() as c:
        return {r["chave"]: r["valor"] for r in c.execute("SELECT * FROM config")}


def set_config(d):
    with conn() as c:
        for k, v in d.items():
            c.execute("UPDATE config SET valor=? WHERE chave=?", (str(v), k))


def evento(tipo, detalhe=""):
    with conn() as c:
        c.execute("INSERT INTO eventos(tipo,detalhe) VALUES(?,?)", (tipo, detalhe))


def eventos(limit=50):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM eventos ORDER BY id DESC LIMIT ?", (limit,))]


# ── contratos ──────────────────────────────────────────────────────────
def upsert(rows):
    with conn() as c:
        for r in rows:
            c.execute("""INSERT INTO contratos(id,objeto,orgao,valor,uf,categoria,
                           score,status,origem) VALUES(:id,:objeto,:orgao,:valor,
                           :uf,:categoria,:score,:status,:origem)
                         ON CONFLICT(id) DO NOTHING""", r)


def contrato(cid):
    with conn() as c:
        r = c.execute("SELECT * FROM contratos WHERE id=?", (cid,)).fetchone()
        return dict(r) if r else None


def set_status(cid, status):
    with conn() as c:
        c.execute("UPDATE contratos SET status=? WHERE id=?", (status, cid))


def set_score(cid, score, status=None):
    with conn() as c:
        if status:
            c.execute("UPDATE contratos SET score=?,status=? WHERE id=?", (score, status, cid))
        else:
            c.execute("UPDATE contratos SET score=? WHERE id=?", (score, cid))


def set_llm(cid, classe, motivo):
    with conn() as c:
        c.execute("UPDATE contratos SET llm_classe=?,llm_motivo=? WHERE id=?",
                  (classe, motivo, cid))


def n_contratos():
    with conn() as c:
        return c.execute("SELECT COUNT(*) n FROM contratos").fetchone()["n"]


def listar(status=None, categoria=None, limit=50, offset=0, ordem="score DESC"):
    q, a = "SELECT * FROM contratos WHERE 1=1", []
    if status:
        q += " AND status IN (%s)" % ",".join("?" * len(status)); a += status
    if categoria:
        q += " AND categoria IN (%s)" % ",".join("?" * len(categoria)); a += categoria
    q += f" ORDER BY {ordem} LIMIT ? OFFSET ?"; a += [limit, offset]
    with conn() as c:
        return [dict(r) for r in c.execute(q, a)]


def gerais_para_pontuar():
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id,objeto FROM contratos WHERE categoria='geral' "
            "AND status IN ('novo','baixa')")]


# ── corpus de treino (positivos órgão + gerais amostra + rótulos humanos) ──
def corpus_treino(max_neg_por_pos=3):
    with conn() as c:
        pos = [(r["objeto"], 1, 1.0) for r in c.execute(
            "SELECT objeto FROM contratos WHERE categoria IN ('engenharia','obras')")]
        n_neg = max(200, len(pos) * max_neg_por_pos)
        neg = [(r["objeto"], 0, 1.0) for r in c.execute(
            "SELECT objeto FROM contratos WHERE categoria='geral' "
            "ORDER BY RANDOM() LIMIT ?", (n_neg,))]
        peso = float(get_config().get("peso_feedback", 5))
        hum = [(r["objeto"], int(r["eh_eng"]), peso) for r in c.execute(
            "SELECT c.objeto, t.eh_eng FROM triagem t JOIN contratos c "
            "ON c.id=t.contrato_id")]
    return pos + neg + hum


def n_triagens():
    with conn() as c:
        return c.execute("SELECT COUNT(*) n FROM triagem").fetchone()["n"]


# ── triagem / rito ────────────────────────────────────────────────────────
def add_triagem(cid, usuario, eh_eng, justificativa):
    with conn() as c:
        c.execute("INSERT INTO triagem(contrato_id,usuario,eh_eng,justificativa)"
                  " VALUES(?,?,?,?)", (cid, usuario, eh_eng, justificativa))


def salva_rito(cid, d):
    with conn() as c:
        c.execute("""INSERT INTO rito(contrato_id,ncp_compra,n_docs,chars,marcadores,
                       mk_score,trecho,llm_rito) VALUES(?,?,?,?,?,?,?,?)
                     ON CONFLICT(contrato_id) DO UPDATE SET ncp_compra=excluded.ncp_compra,
                       n_docs=excluded.n_docs,chars=excluded.chars,
                       marcadores=excluded.marcadores,mk_score=excluded.mk_score,
                       trecho=excluded.trecho,llm_rito=excluded.llm_rito""",
                  (cid, d["ncp_compra"], d["n_docs"], d["chars"], d["marcadores"],
                   d["mk_score"], d["trecho"], d.get("llm_rito")))


def get_rito(cid):
    with conn() as c:
        r = c.execute("SELECT * FROM rito WHERE contrato_id=?", (cid,)).fetchone()
        return dict(r) if r else None


def veredito_rito(cid, seguido, usuario):
    with conn() as c:
        c.execute("UPDATE rito SET rito_seguido=?,usuario=? WHERE contrato_id=?",
                  (seguido, usuario, cid))


# ── estatísticas ──────────────────────────────────────────────────────────
def stats():
    with conn() as c:
        g = lambda q, *a: c.execute(q, a).fetchone()[0]
        st = lambda s: g("SELECT COUNT(*) FROM contratos WHERE status=?", s)
        por_tipo = [dict(r) for r in c.execute(
            "SELECT COALESCE(NULLIF(orgao,''),'(sem órgão)') tipo, COUNT(*) n "
            "FROM contratos WHERE status='novo' GROUP BY tipo ORDER BY n DESC LIMIT 8")]
        return {
            "total": g("SELECT COUNT(*) FROM contratos"),
            "referencia": g("SELECT COUNT(*) FROM contratos WHERE categoria IN ('engenharia','obras')"),
            "gerais": g("SELECT COUNT(*) FROM contratos WHERE categoria='geral'"),
            "novos": st("novo"), "aguarda_rito": st("aguarda_rito"),
            "subenq_real": st("subenq_real"), "rito_seguido": st("rito_seguido"),
            "triagem_nao": st("triagem_nao"),
            "valor_subenq": g("SELECT COALESCE(SUM(valor),0) FROM contratos WHERE status='subenq_real'") or 0,
            "por_tipo": por_tipo, "triagens": g("SELECT COUNT(*) FROM triagem"),
            "modelo": None,
        }
