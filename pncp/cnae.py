"""
Enriquecimento via CNAE (Classificação Nacional de Atividades Econômicas).

Para cada CNPJ de fornecedor, consulta a BrasilAPI / OpenCNPJ e cruza com
a lista oficial de 702 CNAEs do CONFEA (cnaes_crea.xlsx). Se o fornecedor
tem CNAE de engenharia mas o contrato foi rotulado 'geral', é forte indício
de subenquadramento.

Cache em disco — não consulta o mesmo CNPJ duas vezes.
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests

from pncp import config
from pncp.io_disco import (
    ler_parquet, salvar_parquet, ler_json, salvar_json,
)
from pncp.ram import liberar, com_gc


_RX_DIGITOS = re.compile(r"\D+")


def _so_digitos(s):
    return _RX_DIGITOS.sub("", str(s)) if s else ""


# ── CNAEs do CONFEA (engenharia regulamentada) ───────────────────────────────
def carregar_cnaes_crea(caminho_excel="cnaes_crea.xlsx"):
    """Lê a planilha do CONFEA e devolve um set com códigos limpos."""
    if not Path(caminho_excel).exists():
        print(f"[cnae] {caminho_excel} não encontrado — sem lista CREA")
        return set()
    df = pd.read_excel(caminho_excel, dtype=str)
    # Tenta detectar a coluna que tem códigos CNAE
    col = next((c for c in df.columns
                 if "cnae" in c.lower() or "código" in c.lower()), df.columns[0])
    codigos = df[col].dropna().map(_so_digitos)
    codigos = codigos[codigos.str.len() >= 5].str.zfill(7)
    return set(codigos)


# ── Consulta CNPJ (cache em disco) ───────────────────────────────────────────
def _path_cache_cnpj():
    return config.caminho(config.SUB_P8, "cache_cnpj.json")


def _carregar_cache():
    p = _path_cache_cnpj()
    return ler_json(p) if Path(p).exists() else {}


def _gravar_cache(cache):
    salvar_json(cache, _path_cache_cnpj())


def _consultar_brasilapi(cnpj):
    url = f"https://brasilapi.com.br/api/cnpj/v1/{cnpj}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def _consultar_opencnpj(cnpj):
    url = f"https://opencnpj.org/api/cnpj/{cnpj}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def consultar_cnpj(cnpj):
    """Consulta com fallback (BrasilAPI → OpenCNPJ) e cache local."""
    cnpj = _so_digitos(cnpj).zfill(14)
    cache = _carregar_cache()
    if cnpj in cache:
        return cache[cnpj]
    dados = None
    for tentar in (_consultar_brasilapi, _consultar_opencnpj):
        try:
            dados = tentar(cnpj)
            break
        except Exception:
            continue
    cache[cnpj] = dados or {}
    _gravar_cache(cache)
    return cache[cnpj]


def _extrair_cnaes(dados_cnpj):
    """Tira os códigos CNAE (principal + secundários) do payload."""
    if not dados_cnpj:
        return []
    codigos = []
    p = dados_cnpj.get("cnae_fiscal") or dados_cnpj.get("cnae_principal")
    if p:
        codigos.append(_so_digitos(p).zfill(7))
    for sec in dados_cnpj.get("cnaes_secundarios", []) or []:
        c = sec.get("codigo") if isinstance(sec, dict) else sec
        if c:
            codigos.append(_so_digitos(c).zfill(7))
    return codigos


# ── Pipeline completo ────────────────────────────────────────────────────────
@com_gc
def executar(caminho_parquet=None,
             max_consultas=200,
             caminho_excel_crea="cnaes_crea.xlsx"):
    from pncp.ram import precisa_de
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")
    if not precisa_de(caminho_parquet, "cnae",
                       "rode pncp.coleta.coletar(...) primeiro"):
        return None

    df = ler_parquet(caminho_parquet)
    if df.empty:
        print("[cnae] parquet vazio — pulando")
        return None
    col_cnpj = next((c for c in ("niFornecedor", "cnpjFornecedor", "cnpj")
                      if c in df.columns), None)
    if col_cnpj is None:
        print("[cnae] coluna de CNPJ não encontrada — sem enriquecimento")
        return None

    cnaes_crea = carregar_cnaes_crea(caminho_excel_crea)
    print(f"[cnae] {len(cnaes_crea)} CNAEs CONFEA carregados")

    # Pega CNPJs únicos dos suspeitos (rótulo 'geral' com termos de eng)
    suspeitos = df.copy()
    if "n_termos_eng" in suspeitos.columns:
        suspeitos = suspeitos[(suspeitos["rotulo"] == "geral") &
                              (suspeitos["n_termos_eng"] > 0)]
    cnpjs = (suspeitos[col_cnpj].dropna().astype(str)
             .map(_so_digitos).unique())[:max_consultas]
    print(f"[cnae] consultando {len(cnpjs)} CNPJs únicos...")

    registros = []
    for i, cnpj in enumerate(cnpjs, 1):
        info = consultar_cnpj(cnpj)
        codigos = _extrair_cnaes(info)
        match_crea = [c for c in codigos if c in cnaes_crea]
        registros.append({
            "cnpj": cnpj,
            "razao_social": info.get("razao_social", "") if info else "",
            "cnaes": codigos,
            "tem_cnae_eng": len(match_crea) > 0,
            "cnaes_eng": match_crea,
        })
        if i % 50 == 0:
            print(f"[cnae] {i}/{len(cnpjs)}")
        time.sleep(0.2)  # rate limit cordial

    enriq = pd.DataFrame(registros)
    saida = config.caminho(config.SUB_P8, "fornecedores_cnae.parquet")

    # ACUMULA: mescla com fornecedores_cnae anterior, priorizando o novo.
    if Path(saida).exists():
        try:
            anterior = ler_parquet(saida)
            n_antes = len(anterior)
            mantidos = anterior[~anterior["cnpj"].isin(enriq["cnpj"])]
            enriq = pd.concat([mantidos, enriq], ignore_index=True)
            print(f"[cnae] mesclando: {n_antes} antigos + {len(registros)} novos "
                  f"→ {len(enriq)} totais")
        except Exception as e:
            print(f"[cnae] não foi possível mesclar: {e}")

    salvar_parquet(enriq, saida)

    # Cruza com df original e gera amostra para revisão manual
    df_join = df.merge(enriq, left_on=col_cnpj, right_on="cnpj", how="left")
    fortes = df_join[(df_join["rotulo"] == "geral") &
                      (df_join["tem_cnae_eng"] == True)]
    salvar_parquet(fortes,
                   config.caminho(config.SUB_P8, "suspeitos_fortes.parquet"))

    # Amostra para revisão manual — PRESERVA revisões já preenchidas
    saida_csv = config.caminho(config.SUB_P8, "amostra_revisao_manual.csv")
    nova_amostra = fortes.head(50).copy()
    nova_amostra["revisao_manual"] = ""

    if Path(saida_csv).exists():
        # Se já existe, preserva linhas com revisao_manual preenchida.
        # Adiciona NCPs novos que ainda não foram revisados.
        try:
            anterior_csv = pd.read_csv(saida_csv)
            if "revisao_manual" in anterior_csv.columns:
                ja_revisados = anterior_csv[
                    anterior_csv["revisao_manual"]
                    .fillna("").astype(str).ne("")
                ]
                # Mantém os já revisados + adiciona novos não vistos
                novos_ncp = set(nova_amostra["numeroControlePNCP"]) - \
                            set(ja_revisados["numeroControlePNCP"])
                a_adicionar = nova_amostra[
                    nova_amostra["numeroControlePNCP"].isin(novos_ncp)
                ]
                amostra_final = pd.concat([ja_revisados, a_adicionar],
                                            ignore_index=True)
                amostra_final.to_csv(saida_csv, index=False)
                print(f"[cnae] revisão manual preservada: "
                      f"{len(ja_revisados)} já revisados + "
                      f"{len(a_adicionar)} novos = {len(amostra_final)}")
            else:
                nova_amostra.to_csv(saida_csv, index=False)
        except Exception as e:
            print(f"[cnae] erro ao preservar revisões: {e}")
            # Backup do CSV antigo antes de sobrescrever
            try:
                from datetime import datetime as _dt
                bak = saida_csv.with_suffix(
                    f".bak.{_dt.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                Path(saida_csv).rename(bak)
                print(f"[cnae] backup salvo em {bak.name}")
            except Exception:
                pass
            nova_amostra.to_csv(saida_csv, index=False)
    else:
        nova_amostra.to_csv(saida_csv, index=False)

    metricas = {
        "n_cnpjs_consultados": int(len(cnpjs)),
        "n_com_cnae_eng": int(enriq["tem_cnae_eng"].sum()),
        "n_suspeitos_fortes": int(len(fortes)),
        "amostra_revisao": str(saida_csv),
    }
    salvar_json(metricas, config.caminho(config.SUB_P8, "resumo.json"))
    print(f"[cnae] {metricas['n_suspeitos_fortes']} suspeitos fortes "
          f"(geral + CNAE engenharia)")
    liberar(df, df_join, enriq)
    return metricas
