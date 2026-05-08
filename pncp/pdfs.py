"""
Camada 2 — Análise de PDFs (Termo de Referência / Edital).

Para cada contrato suspeito, baixa documentos do PNCP, extrai texto via
PyMuPDF (rápido) ou pdfplumber/OCR (fallback), normaliza (de-hifeniza), e
detecta marcadores de engenharia (ART, RRT, ABNT, memorial descritivo,
norma técnica, etc).

Stream-friendly: cada PDF é processado e descartado, só features ficam em RAM.
"""

import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from pncp import config
from pncp.io_disco import ler_parquet, salvar_parquet, salvar_json
from pncp.ram import liberar, com_gc, monitorar_ram


# ── Decompõe número de controle PNCP ─────────────────────────────────────────
_RX_NCP = re.compile(
    r"^(?P<cnpj>\d{14})-1-(?P<seq>\d{6})/(?P<ano>\d{4})$"
)


def _decompor_ncp(num_controle):
    if not num_controle:
        return None
    m = _RX_NCP.match(str(num_controle).strip())
    if not m:
        return None
    return {"cnpj": m["cnpj"], "ano": int(m["ano"]),
            "sequencial": int(m["seq"])}


# ── Listagem e download via API PNCP ─────────────────────────────────────────
def _listar_documentos(cnpj, ano, seq):
    url = (f"{config.API_BASE}/v1/orgaos/{cnpj}/compras/{ano}/{seq}/"
           f"arquivos")
    try:
        r = requests.get(url, timeout=config.PDFS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _baixar_documento(cnpj, ano, seq, seq_doc, destino):
    url = (f"{config.API_BASE}/v1/orgaos/{cnpj}/compras/{ano}/{seq}/"
           f"arquivos/{seq_doc}")
    r = requests.get(url, timeout=config.PDFS_TIMEOUT, stream=True)
    if r.status_code != 200:
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def _path_cache_pdf(num_controle, seq_doc):
    pasta = config.caminho(config.SUB_C2, "cache_pdfs")
    nome = num_controle.replace("/", "_") + f"_{seq_doc}.pdf"
    return pasta / nome


# ── Extração de texto (3 estratégias) ────────────────────────────────────────
def _extrair_pymupdf(caminho):
    try:
        import fitz
        doc = fitz.open(caminho)
        textos = []
        for i, pag in enumerate(doc):
            if i >= config.PDFS_MAX_PAGINAS:
                break
            textos.append(pag.get_text())
        doc.close()
        return "\n".join(textos)
    except Exception:
        return ""


def _extrair_pdfplumber(caminho):
    try:
        import pdfplumber
        with pdfplumber.open(caminho) as pdf:
            textos = [(p.extract_text() or "")
                       for p in pdf.pages[: config.PDFS_MAX_PAGINAS]]
        return "\n".join(textos)
    except Exception:
        return ""


def _extrair_ocr(caminho):
    if not config.PDFS_USAR_OCR:
        return ""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io
        doc = fitz.open(caminho)
        textos = []
        for i, pag in enumerate(doc):
            if i >= config.PDFS_MAX_PAGINAS:
                break
            pix = pag.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            textos.append(pytesseract.image_to_string(img, lang="por"))
        doc.close()
        return "\n".join(textos)
    except Exception:
        return ""


def extrair_robusto(caminho):
    """Tenta PyMuPDF → pdfplumber → OCR. Retorna o primeiro que tiver >100 chars."""
    for fn in (_extrair_pymupdf, _extrair_pdfplumber, _extrair_ocr):
        txt = fn(caminho)
        if len(txt) > 100:
            return txt
    return ""


# ── Normalização (de-hifenização e similares) ────────────────────────────────
_RX_HIFEN_QUEBRA = re.compile(r"-\s*\n\s*")
_RX_QUEBRAS = re.compile(r"\s*\n\s*")
_RX_MULTI_ESPACO = re.compile(r"[ \t]+")


def _normalizar(texto):
    if not texto:
        return ""
    t = unicodedata.normalize("NFKC", texto)
    t = _RX_HIFEN_QUEBRA.sub("", t)        # palavra-\nquebrada → palavraquebrada
    t = _RX_QUEBRAS.sub(" ", t)
    t = _RX_MULTI_ESPACO.sub(" ", t)
    return t.strip().lower()


# ── Marcadores de engenharia em PDF ──────────────────────────────────────────
MARCADORES = {
    "art": r"\bart\s*(n[ºo]\s*)?\d",
    "rrt": r"\brrt\s*(n[ºo]\s*)?\d",
    "abnt_nbr": r"\babnt\s*nbr\s*\d",
    "memorial": r"\bmemorial\s+descritivo",
    "projeto_executivo": r"\bprojeto\s+executivo",
    "as_built": r"\bas\s*[- ]?built",
    "crea": r"\bcrea\b",
    "engenheiro": r"\bengenhei[rd]o\b",
    "norma_tecnica": r"\bnorma\s+t[eé]cnica",
    "anotacao_responsabilidade": r"\banotac[aã]o\s+de\s+responsabilidade",
}
MARCADORES_RX = {k: re.compile(v) for k, v in MARCADORES.items()}


def detectar_marcadores(texto):
    """Retorna dict {marcador: contagem}."""
    if not texto:
        return {k: 0 for k in MARCADORES}
    return {k: len(rx.findall(texto)) for k, rx in MARCADORES_RX.items()}


# ── Pipeline ─────────────────────────────────────────────────────────────────
@com_gc
def executar(caminho_parquet=None, max_contratos=200, ranking_path=None):
    from pncp.ram import precisa_de
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")
    if ranking_path is None:
        ranking_path = config.caminho(config.SUB_P2, "ranking.parquet")
    if not precisa_de(caminho_parquet, "pdfs",
                       "rode pncp.coleta.coletar(...) primeiro"):
        return None

    monitorar_ram("início pdfs")
    df = ler_parquet(caminho_parquet)
    if df.empty:
        print("[pdfs] parquet vazio — pulando")
        return None

    # Prioriza contratos do ranking (top suspeitos) se disponível
    col_ncp = next((c for c in ("numeroControlePNCP", "numero_controle_pncp",
                                 "numeroControlePncp")
                     if c in df.columns), None)
    if col_ncp is None:
        print("[pdfs] coluna numeroControlePNCP não encontrada")
        return None

    if Path(ranking_path).exists():
        ranking = ler_parquet(ranking_path).head(max_contratos)
        if col_ncp in ranking.columns:
            ncps = ranking[col_ncp].dropna().astype(str).tolist()
        else:
            ncps = df[df["rotulo"] == "geral"][col_ncp].dropna().astype(str) \
                    .head(max_contratos).tolist()
    else:
        ncps = df[df["rotulo"] == "geral"][col_ncp].dropna().astype(str) \
                .head(max_contratos).tolist()

    print(f"[pdfs] processando {len(ncps)} contratos...")
    registros = []
    for i, ncp in enumerate(ncps, 1):
        partes = _decompor_ncp(ncp)
        if not partes:
            continue
        docs = _listar_documentos(partes["cnpj"], partes["ano"],
                                    partes["sequencial"])
        for doc in docs[:3]:  # no máx 3 PDFs por contrato
            seq_doc = doc.get("sequencialDocumento") or doc.get("sequencial")
            if not seq_doc:
                continue
            cache = _path_cache_pdf(ncp, seq_doc)
            if not cache.exists():
                ok = _baixar_documento(partes["cnpj"], partes["ano"],
                                         partes["sequencial"], seq_doc, cache)
                if not ok:
                    continue
                time.sleep(0.3)
            texto = _normalizar(extrair_robusto(cache))
            marc = detectar_marcadores(texto)
            registros.append({
                "numeroControlePNCP": ncp,
                "seq_doc": seq_doc,
                "n_chars": len(texto),
                **marc,
            })
        if i % 20 == 0:
            print(f"[pdfs] {i}/{len(ncps)}")
            monitorar_ram(f"PDFs {i}")

    if not registros:
        print("[pdfs] nenhum PDF processado com sucesso")
        return None

    feats = pd.DataFrame(registros)
    # Agrega por contrato (soma de marcadores)
    agg = feats.groupby("numeroControlePNCP").agg(
        n_pdfs=("seq_doc", "count"),
        chars_total=("n_chars", "sum"),
        **{k: (k, "sum") for k in MARCADORES},
    ).reset_index()
    agg["score_engenharia_pdf"] = agg[list(MARCADORES)].sum(axis=1)

    saida = config.caminho(config.SUB_C2, "features_pdfs.parquet")
    salvar_parquet(agg, saida)
    salvar_json({
        "n_contratos_processados": int(len(agg)),
        "n_pdfs_extraidos": int(len(feats)),
        "media_score": float(agg["score_engenharia_pdf"].mean()),
    }, config.caminho(config.SUB_C2, "resumo.json"))
    print(f"[pdfs] {len(agg)} contratos com features → {saida}")
    liberar(df, feats)
    return saida
