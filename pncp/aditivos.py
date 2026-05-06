"""
Camada 3 — Termos aditivos.

Aditivos que mudam o objeto, prazo ou valor podem indicar mudança de escopo
não declarada inicialmente. Para contratos rotulados 'geral' que receberam
aditivo com objeto de engenharia, é forte indício de subenquadramento.
"""

import time
import re
from typing import List

import pandas as pd
import requests

from pncp import config
from pncp.io_disco import ler_parquet, salvar_parquet, salvar_json
from pncp.ram import liberar, com_gc


_RX_NCP = re.compile(r"^(?P<cnpj>\d{14})-1-(?P<seq>\d{6})/(?P<ano>\d{4})$")


def _decompor_ncp(num_controle):
    if not num_controle:
        return None
    m = _RX_NCP.match(str(num_controle).strip())
    if not m:
        return None
    return {"cnpj": m["cnpj"], "ano": int(m["ano"]),
            "sequencial": int(m["seq"])}


def _listar_aditivos(cnpj, ano, seq):
    """Lista termos aditivos de um contrato via API PNCP."""
    url = (f"{config.API_BASE}/v1/orgaos/{cnpj}/contratos/{ano}/{seq}/"
           f"termos-aditivos")
    try:
        r = requests.get(url, timeout=config.TIMEOUT_HTTP)
        r.raise_for_status()
        return r.json() or []
    except Exception:
        return []


def _eh_mudanca_escopo(aditivo):
    """Heurística: tipo do aditivo sugere mudança de escopo/objeto."""
    txt = " ".join(str(v) for v in aditivo.values() if v).lower()
    return any(p in txt for p in (
        "objeto", "escopo", "acréscimo", "acrescimo", "supressão", "supressao",
        "alteração quantitativa", "alteração qualitativa",
    ))


@com_gc
def executar(caminho_parquet=None, max_contratos=200, apenas_geral=True):
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")

    df = ler_parquet(caminho_parquet)
    col_ncp = next((c for c in ("numeroControlePNCP", "numero_controle_pncp",
                                 "numeroControlePncp")
                     if c in df.columns), None)
    if col_ncp is None:
        print("[aditivos] coluna numeroControlePNCP não encontrada")
        return None

    alvo = df[df["rotulo"] == "geral"] if apenas_geral else df
    ncps = (alvo[col_ncp].dropna().astype(str).head(max_contratos).tolist())
    print(f"[aditivos] consultando {len(ncps)} contratos...")

    registros = []
    for i, ncp in enumerate(ncps, 1):
        partes = _decompor_ncp(ncp)
        if not partes:
            continue
        aditivos = _listar_aditivos(partes["cnpj"], partes["ano"],
                                       partes["sequencial"])
        if not aditivos:
            continue
        n_total = len(aditivos)
        n_escopo = sum(1 for a in aditivos if _eh_mudanca_escopo(a))
        registros.append({
            "numeroControlePNCP": ncp,
            "n_aditivos": n_total,
            "n_aditivos_escopo": n_escopo,
            "tem_mudanca_escopo": n_escopo > 0,
        })
        if i % 50 == 0:
            print(f"[aditivos] {i}/{len(ncps)}")
        time.sleep(0.3)

    if not registros:
        print("[aditivos] nenhum aditivo encontrado")
        return None

    out = pd.DataFrame(registros)
    saida = config.caminho(config.SUB_C3, "aditivos.parquet")
    salvar_parquet(out, saida)
    salvar_json({
        "n_contratos": int(len(out)),
        "n_com_aditivo": int(out["n_aditivos"].gt(0).sum()),
        "n_com_mudanca_escopo": int(out["tem_mudanca_escopo"].sum()),
    }, config.caminho(config.SUB_C3, "resumo.json"))
    print(f"[aditivos] {out['tem_mudanca_escopo'].sum()} contratos com "
          f"mudança de escopo")
    liberar(df, out)
    return saida
