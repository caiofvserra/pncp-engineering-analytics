"""
Camada 2 — Termo de Referência / Edital / Projeto Básico (PDFs anexados).

Pipeline:
  1. Para cada contrato suspeito, lista documentos via API de integração
  2. Filtra apenas tipos relevantes (TR, Projeto Básico, ETP, Edital, ...)
  3. Baixa em cache local; usa de novo se já existe
  4. Extrai texto (PyMuPDF → pdfplumber → OCR como fallbacks)
  5. Normaliza (de-hifeniza) e detecta marcadores legais (ART, CREA…)
  6. Agrega por contrato e produz score 0-9 de engenharia

Output: dados/pdfs/features_pdfs.parquet com 1 linha por contrato.
"""

import time
from pathlib import Path

import pandas as pd
import requests

from pncp import config
from pncp._marcadores import (
    MAPA_TIPO_DOCUMENTO, TIPOS_RELEVANTES_ENGENHARIA,
    detectar_marcadores, normalizar_pdf_text,
    COLS_MARCADORES, COLS_PRESENCA,
)
from pncp.io_disco import ler_parquet, salvar_parquet, salvar_json
from pncp.ram import liberar, com_gc, monitorar_ram


# ── numeroControlePNCP → componentes ─────────────────────────────────────────
import re as _re
_RX_NCP = _re.compile(r"^(?P<cnpj>\d{14})-(?P<tipo>\d+)-(?P<seq>\d+)/(?P<ano>\d{4})$")


def _decompor_ncp(num_controle):
    if not num_controle:
        return None
    m = _RX_NCP.match(str(num_controle).strip())
    if not m:
        return None
    return {"cnpj": m["cnpj"], "tipo": int(m["tipo"]),
            "ano": int(m["ano"]), "sequencial": int(m["seq"])}


# ── API de integração — listagem e download ─────────────────────────────────
def _listar_documentos(cnpj, ano, seq, tipo_recurso="compras",
                        timeout=None):
    """
    Retorna (documentos, status):
      status="ok"               → API respondeu com sucesso (lista pode ser vazia)
      status="timeout"          → não sabemos se tem doc; vale re-tentar depois
      status="erro_4xx"/"5xx"   → falha de servidor, pula
    """
    timeout = timeout or config.PDFS_TIMEOUT
    url = (f"{config.API_INTEGRACAO}/v1/orgaos/{cnpj}/{tipo_recurso}/"
           f"{ano}/{seq}/arquivos")
    for tentativa in range(2):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                return [], f"erro_{r.status_code}"
            payload = r.json()
            docs = (payload if isinstance(payload, list)
                    else payload.get("data", []))
            return docs, "ok"
        except requests.exceptions.Timeout:
            if tentativa == 0:
                continue
            return [], "timeout"
        except Exception:
            return [], "erro_conexao"
    return [], "timeout"


def _baixar(cnpj, ano, seq, seq_doc, destino, tipo_recurso="compras"):
    url = (f"{config.API_INTEGRACAO}/v1/orgaos/{cnpj}/{tipo_recurso}/"
           f"{ano}/{seq}/arquivos/{seq_doc}")
    try:
        r = requests.get(url, timeout=config.PDFS_TIMEOUT_DOWNLOAD,
                          stream=True)
        if r.status_code != 200:
            return False
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception:
        return False


def descobrir_documentos(num_controle_pncp):
    """
    Função de debug: lista os documentos de UM contrato específico.
    Use para confirmar que o endpoint está respondendo corretamente.

    Ex: pncp.pdfs.descobrir_documentos("12345678000199-1-000123/2024")
    """
    info = _decompor_ncp(num_controle_pncp)
    if not info:
        print(f"❌ numeroControlePNCP inválido: {num_controle_pncp}")
        return None
    recurso = "compras" if info["tipo"] == 1 else "contratos"
    docs, status = _listar_documentos(info["cnpj"], info["ano"],
                                        info["sequencial"], recurso)
    print(f"\n🔍 {num_controle_pncp}")
    print(f"   CNPJ={info['cnpj']} ano={info['ano']} seq={info['sequencial']}")
    print(f"   recurso={recurso} status={status}  → {len(docs)} doc(s):")
    for d in docs:
        seq_doc = d.get("sequencialDocumento") or d.get("sequencial", "?")
        tipo_id = d.get("tipoDocumentoId")
        tipo_nm = (d.get("tipoDocumentoNome")
                    or MAPA_TIPO_DOCUMENTO.get(tipo_id, "Outro"))
        titulo = (d.get("titulo") or "")[:60]
        ativo = d.get("statusAtivo", True)
        print(f"      [{seq_doc}] tipo={tipo_id} ({tipo_nm}): {titulo} "
              f"{'✓' if ativo else '✗'}")
    return docs


def _path_cache_pdf(num_controle, seq_doc):
    pasta = config.caminho(config.SUB_C2, "cache_pdfs")
    nome = num_controle.replace("/", "_") + f"_{seq_doc}.pdf"
    return pasta / nome


# ── Extração de texto (3 estratégias em cascata) ────────────────────────────
def _extrair_pymupdf(caminho):
    try:
        import fitz
        doc = fitz.open(caminho)
        textos = []
        for i, p in enumerate(doc):
            if i >= config.PDFS_MAX_PAGINAS:
                break
            textos.append(p.get_text())
        doc.close()
        return "\n".join(textos)
    except Exception:
        return ""


def _extrair_pdfplumber(caminho):
    try:
        import pdfplumber
        with pdfplumber.open(caminho) as pdf:
            return "\n".join((p.extract_text() or "")
                              for p in pdf.pages[:config.PDFS_MAX_PAGINAS])
    except Exception:
        return ""


def _extrair_ocr(caminho):
    if not config.PDFS_USAR_OCR:
        return ""
    try:
        import fitz, pytesseract, io
        from PIL import Image
        doc = fitz.open(caminho)
        textos = []
        for i, p in enumerate(doc):
            if i >= config.PDFS_MAX_PAGINAS:
                break
            pix = p.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            textos.append(pytesseract.image_to_string(img, lang="por"))
        doc.close()
        return "\n".join(textos)
    except Exception:
        return ""


def extrair_texto(caminho, min_chars=200):
    """Tenta PyMuPDF → pdfplumber → OCR; retorna o primeiro que satisfaz."""
    for fn in (_extrair_pymupdf, _extrair_pdfplumber, _extrair_ocr):
        t = fn(caminho)
        if len(t.strip()) >= min_chars:
            return t
    return ""


# ── Pipeline ─────────────────────────────────────────────────────────────────
@com_gc
def executar(caminho_parquet=None, max_contratos=1000, ranking_path=None,
             tipos_aceitos=None, apenas_geral_obvio=True):
    """
    tipos_aceitos: tipoDocumentoId a baixar. Default = TIPOS_RELEVANTES_ENGENHARIA.
    apenas_geral_obvio: se True (default), só baixa PDFs de contratos
       'geral' marcados como `eh_obvio_engenharia` na triagem (mais barato
       e mais útil — esses são os candidatos a subenquadramento).
    """
    from pncp.ram import precisa_de
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")
    if ranking_path is None:
        ranking_path = config.caminho(config.SUB_P2, "ranking.parquet")
    if not precisa_de(caminho_parquet, "pdfs",
                       "rode pncp.coleta.coletar(...) primeiro"):
        return None
    if tipos_aceitos is None:
        tipos_aceitos = TIPOS_RELEVANTES_ENGENHARIA

    monitorar_ram("início pdfs")
    df = ler_parquet(caminho_parquet)
    if df.empty:
        print("[pdfs] parquet vazio — pulando")
        return None

    col_ncp = next((c for c in ("numeroControlePNCP", "numero_controle_pncp",
                                 "numeroControlePncp")
                     if c in df.columns), None)
    if col_ncp is None:
        print("[pdfs] coluna numeroControlePNCP não encontrada")
        return None

    # Prioriza contratos óbvios da triagem (têm PDFs informativos)
    triagem_path = config.caminho("triagem", "triagem.parquet")
    if apenas_geral_obvio and triagem_path.exists():
        triagem = ler_parquet(triagem_path)
        obvios = triagem[(triagem["rotulo"] == "geral") &
                          triagem.get("eh_obvio_engenharia", False)]
        if not obvios.empty:
            ncps = obvios[col_ncp].dropna().astype(str) \
                       .head(max_contratos).tolist()
            print(f"[pdfs] alvo: {len(ncps)} 'geral' óbvios da triagem")
        else:
            ncps = []
    elif Path(ranking_path).exists():
        ranking = ler_parquet(ranking_path).head(max_contratos)
        ncps = (ranking[col_ncp].dropna().astype(str).tolist()
                if col_ncp in ranking.columns
                else df[df["rotulo"] == "geral"][col_ncp].dropna().astype(str)
                       .head(max_contratos).tolist())
    else:
        ncps = (df[df["rotulo"] == "geral"][col_ncp].dropna().astype(str)
                .head(max_contratos).tolist())

    if not ncps:
        print("[pdfs] nenhum contrato candidato — rode triagem antes")
        return None

    print(f"[pdfs] processando {len(ncps)} contratos "
          f"(timeout listagem={config.PDFS_TIMEOUT}s, download="
          f"{config.PDFS_TIMEOUT_DOWNLOAD}s)...")
    try:
        from tqdm.auto import tqdm as _tqdm
        _iter = _tqdm(enumerate(ncps, 1), total=len(ncps),
                       desc="📥 PDFs", unit="contrato")
    except ImportError:
        _iter = enumerate(ncps, 1)

    registros = []
    n_sem_doc = n_baixados = n_cache_hit = 0
    n_timeout = n_erro = 0
    # Persiste lista de timeouts p/ retentar depois com pncp.pdfs.retentar_falhas()
    diagnostico = {"confirmados_sem_doc": [], "timeout_listagem": [],
                    "erro_servidor": []}
    diag_path = config.caminho(config.SUB_C2, "pdfs_diagnostico.json")
    # Carrega lista anterior de "confirmados sem doc" — pula esses (já sabemos)
    confirmados_sem_doc_prev = set()
    if diag_path.exists():
        try:
            from pncp.io_disco import ler_json
            d_prev = ler_json(diag_path)
            confirmados_sem_doc_prev = set(d_prev.get("confirmados_sem_doc", []))
            if confirmados_sem_doc_prev:
                print(f"[pdfs] pulando {len(confirmados_sem_doc_prev)} contratos "
                      f"já confirmados sem-doc em run anterior")
        except Exception:
            pass

    for i, ncp in _iter:
        info = _decompor_ncp(ncp)
        if not info or info["tipo"] not in (1, 2):
            continue
        if ncp in confirmados_sem_doc_prev:
            n_sem_doc += 1
            continue

        recurso = "compras" if info["tipo"] == 1 else "contratos"
        docs, status = _listar_documentos(info["cnpj"], info["ano"],
                                            info["sequencial"], recurso)
        if status == "timeout":
            n_timeout += 1
            diagnostico["timeout_listagem"].append(ncp)
            continue
        if status.startswith("erro"):
            n_erro += 1
            diagnostico["erro_servidor"].append(ncp)
            continue
        if not docs:
            # status="ok" e lista vazia = confirmadamente sem doc no PNCP
            n_sem_doc += 1
            diagnostico["confirmados_sem_doc"].append(ncp)
            continue

        # Filtra por tipo (só TR/PB/ETP/Edital/etc.)
        docs_relevantes = []
        for d in docs:
            tipo_id = d.get("tipoDocumentoId")
            if tipo_id is None:
                # Fallback: tenta achar pelo nome
                nome = (d.get("tipoDocumentoNome") or "").strip().lower()
                for k, v in MAPA_TIPO_DOCUMENTO.items():
                    if v.lower() == nome:
                        tipo_id = k
                        break
            if tipo_id in tipos_aceitos or (tipo_id is None and not tipos_aceitos):
                docs_relevantes.append((d, tipo_id))

        if not docs_relevantes:
            continue

        # Prioriza pela ordem de TIPOS_RELEVANTES_ENGENHARIA:
        # TR (4), PB (6), ETP (7), Proj. Executivo (8), Anteprojeto (5),
        # Edital (2), Minuta (3), Contrato (12), Aditivo (14).
        # Marcadores aparecem MUITO mais em TR/PB/ETP do que em Contrato;
        # baixar Contrato em vez de TR desperdiça banda e degrada a Camada 2.
        ordem_prioridade = {t: i for i, t in enumerate(TIPOS_RELEVANTES_ENGENHARIA)}
        docs_relevantes.sort(
            key=lambda dt: ordem_prioridade.get(dt[1], 999),
        )

        # Limita a 3 PDFs/contrato (TR + PB + ETP normalmente bastam)
        for d, tipo_id in docs_relevantes[:3]:
            seq_doc = d.get("sequencialDocumento") or d.get("sequencial")
            if not seq_doc:
                continue
            cache = _path_cache_pdf(ncp, seq_doc)
            if cache.exists():
                n_cache_hit += 1
            else:
                if not _baixar(info["cnpj"], info["ano"], info["sequencial"],
                                seq_doc, cache, recurso):
                    continue
                n_baixados += 1
                time.sleep(0.3)

            texto = normalizar_pdf_text(extrair_texto(cache))
            marc = detectar_marcadores(texto)
            registros.append({
                "numeroControlePNCP": ncp,
                "seq_doc": seq_doc,
                "tipoDocumentoId": tipo_id,
                "tipoDocumentoNome": MAPA_TIPO_DOCUMENTO.get(tipo_id, "Outro"),
                "n_chars": len(texto),
                **marc,
            })

        # Log granular a cada 50 (tqdm já dá feedback visual contínuo)
        if i % 50 == 0:
            print(f"\n[pdfs] {i}/{len(ncps)} | "
                  f"baixados={n_baixados}, cache={n_cache_hit}, "
                  f"sem-doc={n_sem_doc}, timeout={n_timeout}, "
                  f"erro={n_erro}, com-features={len(registros)}")
            monitorar_ram(f"PDFs {i}")
            # Salva diagnóstico parcial para sobreviver a interrupções
            salvar_json(diagnostico, diag_path)

    # Salva diagnóstico final
    salvar_json(diagnostico, diag_path)

    if n_timeout > 0:
        print(f"\n[pdfs] ⚠ {n_timeout} contratos deram TIMEOUT — não sabemos "
              f"se têm doc. Rode `pncp.pdfs.retentar_falhas()` p/ tentar "
              f"de novo com timeout maior.")
    if n_erro > 0:
        print(f"[pdfs] ⚠ {n_erro} contratos com erro de servidor "
              f"(salvos em pdfs_diagnostico.json)")

    if not registros:
        print(f"[pdfs] nenhum PDF processado | sem-doc={n_sem_doc}, "
              f"cache={n_cache_hit}, baixados={n_baixados}, "
              f"timeout={n_timeout}")
        return None

    feats = pd.DataFrame(registros)
    print(f"[pdfs] {len(feats)} PDFs processados ({n_baixados} novos, "
          f"{n_cache_hit} cache, {n_sem_doc} confirmados sem doc, "
          f"{n_timeout} timeouts)")
    # Diagnóstico granular: distribuição por tipo de doc
    if "tipoDocumentoNome" in feats.columns:
        print(f"[pdfs] distribuição por tipo:")
        for nome, n in feats["tipoDocumentoNome"].value_counts().items():
            print(f"     {nome:30s}: {n:>4d}")

    # Agrega por contrato — soma marcadores, max do score, lista tipos
    agg_dict = {
        "n_pdfs": ("seq_doc", "count"),
        "chars_total": ("n_chars", "sum"),
        "tipos_doc": ("tipoDocumentoNome",
                      lambda s: " | ".join(sorted(set(s.dropna())))),
    }
    for c in COLS_MARCADORES:
        if c in feats.columns:
            agg_dict[c] = (c, "sum")
    for c in COLS_PRESENCA:
        if c in feats.columns:
            agg_dict[c] = (c, "any")
    agg = feats.groupby("numeroControlePNCP").agg(**agg_dict).reset_index()

    # Score agregado: nº de categorias presentes em qualquer PDF do contrato
    if any(c in agg.columns for c in COLS_PRESENCA):
        cols_pres_existem = [c for c in COLS_PRESENCA if c in agg.columns]
        agg["mk_score_engenharia"] = agg[cols_pres_existem].sum(axis=1)

    # ACUMULA: mescla com features_pdfs anterior (priorizando o novo)
    saida = config.caminho(config.SUB_C2, "features_pdfs.parquet")
    if Path(saida).exists():
        try:
            ant = ler_parquet(saida)
            mantidos = ant[~ant["numeroControlePNCP"]
                              .isin(agg["numeroControlePNCP"])]
            agg = pd.concat([mantidos, agg], ignore_index=True)
            print(f"[pdfs] mesclando: {len(mantidos)} antigos + {len(registros)} "
                  f"novos → {len(agg)} totais")
        except Exception as e:
            print(f"[pdfs] mesclagem falhou: {e}")

    salvar_parquet(agg, saida)
    salvar_json({
        "n_contratos_processados": int(len(agg)),
        "n_pdfs_extraidos": int(len(feats)),
        "media_score": float(agg.get("mk_score_engenharia",
                                        pd.Series([0])).mean()),
        "n_baixados_sessao": int(n_baixados),
        "n_cache_hit_sessao": int(n_cache_hit),
        "n_sem_doc": int(n_sem_doc),
    }, config.caminho(config.SUB_C2, "resumo.json"))
    print(f"[pdfs] {len(agg)} contratos com features → {saida}")

    # Gera gráfico de marcadores por rótulo — agora que temos os PDFs
    try:
        from pncp.eda import g_marcadores_por_rotulo
        p = g_marcadores_por_rotulo(config.PASTA_DADOS)
        if p:
            print(f"[pdfs] gráfico de marcadores → {p}")
    except Exception as e:
        print(f"[pdfs] gráfico marcadores falhou: {e}")

    liberar(df, feats)
    return saida


def retentar_falhas(timeout_listagem=45):
    """
    Re-processa apenas os contratos que deram TIMEOUT na sessão anterior,
    desta vez com timeout maior (default 45s vs 15s padrão).

    Útil quando você suspeita que a API estava sobrecarregada num momento
    específico e quer recuperar dados de contratos que parecem "sem doc"
    mas na verdade são "não-sabemos-se-tem-doc".

    Os contratos confirmados como "sem doc" (API respondeu 200 com lista
    vazia) NÃO são re-tentados — sabemos definitivamente que não têm.
    """
    from pncp.io_disco import ler_json
    diag_path = config.caminho(config.SUB_C2, "pdfs_diagnostico.json")
    if not Path(diag_path).exists():
        print("[retentar] sem pdfs_diagnostico.json — rode pncp.pdfs.executar() antes")
        return None
    diag = ler_json(diag_path)
    pendentes = diag.get("timeout_listagem", [])
    if not pendentes:
        print("[retentar] nenhum timeout pendente")
        return None

    print(f"[retentar] {len(pendentes)} contratos com timeout — re-tentando "
          f"com timeout={timeout_listagem}s")

    try:
        from tqdm.auto import tqdm as _tqdm
        _iter = _tqdm(pendentes, desc="🔄 Retentar", unit="contrato")
    except ImportError:
        _iter = pendentes

    novos_sucesso = 0
    novos_timeout = 0
    novos_sem_doc = 0
    timeout_restante = []

    registros = []
    for ncp in _iter:
        info = _decompor_ncp(ncp)
        if not info or info["tipo"] not in (1, 2):
            continue
        recurso = "compras" if info["tipo"] == 1 else "contratos"
        docs, status = _listar_documentos(info["cnpj"], info["ano"],
                                            info["sequencial"], recurso,
                                            timeout=timeout_listagem)
        if status == "timeout":
            novos_timeout += 1
            timeout_restante.append(ncp)
            continue
        if not docs:
            novos_sem_doc += 1
            diag.setdefault("confirmados_sem_doc", []).append(ncp)
            continue
        novos_sucesso += 1
        # Baixa e processa (mesma lógica de executar)
        for d in docs[:3]:
            seq_doc = d.get("sequencialDocumento") or d.get("sequencial")
            if not seq_doc:
                continue
            cache = _path_cache_pdf(ncp, seq_doc)
            if not cache.exists():
                if not _baixar(info["cnpj"], info["ano"], info["sequencial"],
                                 seq_doc, cache, recurso):
                    continue
                time.sleep(0.3)
            texto = normalizar_pdf_text(extrair_texto(cache))
            marc = detectar_marcadores(texto)
            registros.append({
                "numeroControlePNCP": ncp,
                "seq_doc": seq_doc,
                "tipoDocumentoId": d.get("tipoDocumentoId"),
                "tipoDocumentoNome": d.get("tipoDocumentoNome",
                                            MAPA_TIPO_DOCUMENTO.get(
                                                d.get("tipoDocumentoId"),
                                                "Outro")),
                "n_chars": len(texto), **marc,
            })

    diag["timeout_listagem"] = timeout_restante
    salvar_json(diag, diag_path)
    print(f"[retentar] sucesso={novos_sucesso}, sem-doc={novos_sem_doc}, "
          f"ainda em timeout={novos_timeout}")

    # Mescla os novos registros com features_pdfs existente
    if registros:
        feats = pd.DataFrame(registros)
        agg_dict = {
            "n_pdfs": ("seq_doc", "count"),
            "chars_total": ("n_chars", "sum"),
            "tipos_doc": ("tipoDocumentoNome",
                          lambda s: " | ".join(sorted(set(s.dropna())))),
        }
        for c in COLS_MARCADORES:
            if c in feats.columns:
                agg_dict[c] = (c, "sum")
        for c in COLS_PRESENCA:
            if c in feats.columns:
                agg_dict[c] = (c, "any")
        agg = feats.groupby("numeroControlePNCP").agg(**agg_dict).reset_index()
        cols_pres_existem = [c for c in COLS_PRESENCA if c in agg.columns]
        if cols_pres_existem:
            agg["mk_score_engenharia"] = agg[cols_pres_existem].sum(axis=1)

        saida = config.caminho(config.SUB_C2, "features_pdfs.parquet")
        if Path(saida).exists():
            ant = ler_parquet(saida)
            mantidos = ant[~ant["numeroControlePNCP"]
                              .isin(agg["numeroControlePNCP"])]
            agg = pd.concat([mantidos, agg], ignore_index=True)
        salvar_parquet(agg, saida)
        print(f"[retentar] features_pdfs atualizado: {len(agg)} contratos")

    return {"sucesso": novos_sucesso, "sem_doc": novos_sem_doc,
            "timeout_restante": novos_timeout}
