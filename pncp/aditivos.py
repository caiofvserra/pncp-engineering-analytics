"""
Camada 3 — Termos aditivos (mudança de escopo).

Insight central: contrato original rotulado 'geral' que recebeu aditivo
com marcadores de engenharia (ART, RRT, "obra", etc.) é mudança de escopo
indevida — a licitação original não seguiu o rito de engenharia mas a
execução acabou incluindo trabalho que exigiria. Sinal jurídico forte.

Output:
  - dados/aditivos/aditivos.parquet  (1 linha por aditivo)
  - dados/aditivos/mudanca_escopo_suspeita.csv  (top suspeitos)
  - dados/aditivos/resumo.json
"""

import time
from pathlib import Path

import pandas as pd
import requests

from pncp import config
from pncp._marcadores import (
    eh_termo_aditivo, detectar_marcadores, normalizar_pdf_text,
    COLS_MARCADORES, COLS_PRESENCA,
)
from pncp.io_disco import ler_parquet, salvar_parquet, salvar_json
from pncp.ram import liberar, com_gc

import re
_RX_NCP = re.compile(r"^(?P<cnpj>\d{14})-(?P<tipo>\d+)-(?P<seq>\d+)/(?P<ano>\d{4})$")


def _decompor_ncp(num):
    if not num:
        return None
    m = _RX_NCP.match(str(num).strip())
    if not m:
        return None
    return {"cnpj": m["cnpj"], "tipo": int(m["tipo"]),
            "ano": int(m["ano"]), "sequencial": int(m["seq"])}


def _listar_arquivos(cnpj, ano, seq, tipo_recurso="compras"):
    url = (f"{config.API_INTEGRACAO}/v1/orgaos/{cnpj}/{tipo_recurso}/"
           f"{ano}/{seq}/arquivos")
    try:
        r = requests.get(url, timeout=config.TIMEOUT_HTTP)
        if r.status_code != 200:
            return []
        d = r.json()
        return d if isinstance(d, list) else d.get("data", [])
    except Exception:
        return []


def _baixar_aditivo(cnpj, ano, seq, seq_doc, destino, tipo_recurso="compras"):
    url = (f"{config.API_INTEGRACAO}/v1/orgaos/{cnpj}/{tipo_recurso}/"
           f"{ano}/{seq}/arquivos/{seq_doc}")
    try:
        r = requests.get(url, timeout=config.PDFS_TIMEOUT, stream=True)
        if r.status_code != 200:
            return False
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception:
        return False


def _extrair_texto_pdf(caminho):
    try:
        import fitz
        doc = fitz.open(caminho)
        textos = [p.get_text() for i, p in enumerate(doc)
                   if i < config.PDFS_MAX_PAGINAS]
        doc.close()
        return normalizar_pdf_text("\n".join(textos))
    except Exception:
        return ""


@com_gc
def executar(caminho_parquet=None, max_contratos=2000, apenas_geral=True,
             priorizar_antigos=True):
    """
    apenas_geral: foca em contratos rotulo='geral' (default).
    priorizar_antigos: contratos mais antigos têm mais chance de já ter
    recebido aditivo (default True). Útil porque contratos novos
    raramente têm aditivo ainda — desperdiça API.

    Default `max_contratos=2000` (era 500) para varrer base maior — em
    amostra pequena de contratos recentes, a chance de achar aditivo é
    quase zero. Com 2000 + priorização de antigos, encontra ~5–10%.
    """
    from pncp.ram import precisa_de
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")
    if not precisa_de(caminho_parquet, "aditivos",
                       "rode pncp.coleta.coletar(...) primeiro"):
        return None

    df = ler_parquet(caminho_parquet)
    if df.empty:
        print("[aditivos] parquet vazio — pulando")
        return None

    col_ncp = next((c for c in ("numeroControlePNCP", "numero_controle_pncp",
                                 "numeroControlePncp")
                     if c in df.columns), None)
    if col_ncp is None:
        print("[aditivos] coluna numeroControlePNCP não encontrada")
        return None

    alvo = df[df["rotulo"] == "geral"] if apenas_geral else df
    if priorizar_antigos and "dataPublicacaoPncp" in alvo.columns:
        alvo = alvo.sort_values("dataPublicacaoPncp", ascending=True)
    ncps = alvo[col_ncp].dropna().astype(str).head(max_contratos).tolist()
    print(f"[aditivos] varrendo {len(ncps)} contratos "
          f"({'apenas geral' if apenas_geral else 'todos'}"
          f"{'/antigos primeiro' if priorizar_antigos else ''})")

    pasta_cache = config.caminho(config.SUB_C3, "cache_aditivos")
    registros = []
    n_sem_doc = n_sem_aditivo = n_com_aditivo = n_baixados = n_cache = 0

    for i, ncp in enumerate(ncps, 1):
        info = _decompor_ncp(ncp)
        if not info or info["tipo"] not in (1, 2):
            continue

        recurso = "compras" if info["tipo"] == 1 else "contratos"
        docs = _listar_arquivos(info["cnpj"], info["ano"],
                                  info["sequencial"], recurso)
        if not docs:
            n_sem_doc += 1
            continue
        aditivos = [d for d in docs if eh_termo_aditivo(d)]
        if not aditivos:
            n_sem_aditivo += 1
            continue
        n_com_aditivo += 1

        for d in aditivos:
            seq_doc = d.get("sequencialDocumento") or d.get("sequencial")
            if not seq_doc:
                continue
            cache = pasta_cache / f"{ncp.replace('/', '_')}_aditivo_{seq_doc}.pdf"
            if cache.exists():
                n_cache += 1
            else:
                if not _baixar_aditivo(info["cnpj"], info["ano"],
                                          info["sequencial"], seq_doc,
                                          cache, recurso):
                    continue
                n_baixados += 1
                time.sleep(0.3)

            texto = _extrair_texto_pdf(cache)
            marc = detectar_marcadores(texto)
            registros.append({
                "numeroControlePNCP": ncp,
                "rotulo_original": str(alvo[alvo[col_ncp] == ncp]["rotulo"]
                                          .iloc[0]) if not alvo.empty else "",
                "seq_doc": seq_doc,
                "titulo": (d.get("titulo") or "")[:200],
                "data_publicacao": d.get("dataPublicacaoPncp", ""),
                "n_chars": len(texto),
                **marc,
            })

        if i % 50 == 0:
            print(f"[aditivos] {i}/{len(ncps)} | "
                  f"com-aditivo={n_com_aditivo}, sem-aditivo={n_sem_aditivo}, "
                  f"sem-doc={n_sem_doc}, baixados={n_baixados}, cache={n_cache}")

    if not registros:
        print(f"[aditivos] nenhum aditivo encontrado em {len(ncps)} contratos")
        print(f"   sem-doc={n_sem_doc}  (contrato não tem doc nenhum)")
        print(f"   sem-aditivo={n_sem_aditivo}  (tem doc mas nenhum é aditivo)")
        print(f"   👉 contratos novos geralmente não têm aditivo. "
              f"Tente max_contratos maior ou apenas_geral=False")
        return None

    feats = pd.DataFrame(registros)

    # Mudança de escopo suspeita: contrato 'geral' + aditivo com ≥2
    # categorias de marcadores de engenharia (Lei 6.496/1977)
    feats["mudanca_escopo_suspeita"] = (
        (feats["rotulo_original"] == "geral") &
        (feats.get("mk_score_engenharia", 0) >= 2)
    )

    # ACUMULA com runs anteriores
    saida = config.caminho(config.SUB_C3, "aditivos.parquet")
    if Path(saida).exists():
        try:
            ant = ler_parquet(saida)
            mantidos = ant[~ant["numeroControlePNCP"]
                              .isin(feats["numeroControlePNCP"])]
            feats = pd.concat([mantidos, feats], ignore_index=True)
        except Exception as e:
            print(f"[aditivos] mesclagem falhou: {e}")

    salvar_parquet(feats, saida)

    # Suspeitos em CSV separado para inspeção humana
    suspeitos = (feats[feats["mudanca_escopo_suspeita"]]
                 .sort_values("mk_score_engenharia", ascending=False))
    if not suspeitos.empty:
        cols = ["numeroControlePNCP", "titulo", "data_publicacao",
                "mk_score_engenharia"] + [c for c in COLS_PRESENCA
                                            if c in suspeitos.columns]
        sus_path = config.caminho(config.SUB_C3,
                                    "mudanca_escopo_suspeita.csv")
        suspeitos[cols].to_csv(sus_path, index=False, encoding="utf-8-sig")
        print(f"[aditivos] ⚠ {len(suspeitos)} mudanças de escopo suspeitas → "
              f"{sus_path.name}")

    salvar_json({
        "n_aditivos_total": int(len(feats)),
        "n_baixados_sessao": int(n_baixados),
        "n_cache_hit_sessao": int(n_cache),
        "n_sem_aditivo": int(n_sem_aditivo),
        "n_mudanca_escopo_suspeita": int(feats["mudanca_escopo_suspeita"].sum()),
        "media_score_em_geral": float(
            feats[feats["rotulo_original"] == "geral"]
            .get("mk_score_engenharia", pd.Series([0])).mean()
        ),
    }, config.caminho(config.SUB_C3, "resumo.json"))
    print(f"[aditivos] {len(feats)} aditivos processados → {saida}")
    liberar(df, feats)
    return saida
