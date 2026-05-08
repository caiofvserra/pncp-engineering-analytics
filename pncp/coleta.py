"""
Coleta de contratos via API PNCP — endpoint /v1/contratos.

Por que /v1/contratos e não /v1/contratacoes/publicacao?
  Em testes anteriores, /v1/contratacoes/publicacao com filtro `uf` direto
  na URL dava timeout sistemático (>60s por chamada). O endpoint
  /v1/contratos responde rápido, traz `categoriaProcessoId` (rótulo do TCC)
  e `objetoContrato`. Filtramos UF em Python depois do download.

Estratégia anti-timeout:
  1. Itera mês a mês (janela curta = sem timeout)
  2. Backoff exponencial: 4s, 8s, 16s, 32s, 64s — até 6 tentativas por página
  3. Timeout 60s por requisição
  4. 5 falhas consecutivas só na MESMA página → abandona o mês
  5. Checkpoint mensal: se Colab cair, retoma de onde parou

Funções principais:
  - coletar(uf, anos, ...)         — coleta em modo programático
  - coletar_interativo()           — perguntas no terminal
  - carregar_checkpoint(uf)        — recupera após reiniciar kernel
  - combinar_parquets(uf)          — junta coletas separadas
  - status(uf, anos)               — mostra estado da coleta
  - verificar_duplicatas(uf)       — checa integridade
  - deduplicar(uf)                 — remove duplicatas
"""

import datetime
import os
import time
from pathlib import Path

import pandas as pd
import requests

from pncp import config
from pncp.io_disco import salvar_parquet, ler_parquet
from pncp.ram import liberar, monitorar_ram

# tqdm é opcional (ambiente sem ele cai num shim transparente)
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, *a, **kw):
        return it


# ── Mapas auxiliares (ID → nome humano) ──────────────────────────────────────
MAPA_CATEGORIA = {
    1: "Cessão",
    2: "Compras",
    3: "Informática (TIC)",
    4: "Internacional",
    5: "Locação Imóveis",
    6: "Mão de Obra",
    7: "Obras",
    8: "Serviços",
    9: "Serviços de Engenharia",
    10: "Serviços de Saúde",
    11: "Alienação de bens móveis/imóveis",
}

MAPA_ESFERA = {"F": "Federal", "E": "Estadual", "M": "Municipal", "D": "Distrital"}
MAPA_PODER = {"E": "Executivo", "L": "Legislativo", "J": "Judiciário"}


# ── HTTP com retry exponencial ───────────────────────────────────────────────
def _get_com_retry(url, params, tentativas=6, espera_base=4.0, timeout=60):
    """
    GET resiliente: até 6 tentativas, backoff exponencial (4, 8, 16, 32, 64s).
    Retorna a resposta ou None (após esgotar tentativas).
    """
    for i in range(1, tentativas + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (200, 204):
                return r
            if r.status_code in (400, 422):
                # erro do cliente — não adianta tentar de novo
                return None
            if r.status_code >= 500:
                espera = espera_base * (2 ** (i - 1))
                print(f"   ⚠ HTTP {r.status_code} — tentativa {i}/{tentativas}, "
                      f"aguardando {espera:.0f}s")
                time.sleep(espera)
            else:
                print(f"   ⚠ HTTP {r.status_code} inesperado.")
                return None
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            espera = espera_base * (2 ** (i - 1))
            print(f"   ⚠ {type(e).__name__} — tentativa {i}/{tentativas}, "
                  f"aguardando {espera:.0f}s")
            time.sleep(espera)
    return None


# ── Achata 1 contrato da API em dict plano ───────────────────────────────────
def _aplanar(r):
    """Extrai colunas relevantes de um registro de /v1/contratos."""
    cat = r.get("categoriaProcesso") or {}
    unidade = r.get("unidadeOrgao") or {}
    orgao = r.get("orgaoEntidade") or {}
    tipo = r.get("tipoContrato") or {}
    return {
        "numeroControlePNCP":        r.get("numeroControlePNCP", ""),
        "categoriaProcessoId":       cat.get("id"),
        "categoriaProcessoNome":     cat.get("nome", ""),
        "objetoContrato":            r.get("objetoContrato", ""),
        "informacaoComplementar":    r.get("informacaoComplementar", ""),
        "valorInicial":              r.get("valorInicial"),
        "valorGlobal":               r.get("valorGlobal"),
        "valorAcumulado":            r.get("valorAcumulado"),
        "dataPublicacaoPncp":        r.get("dataPublicacaoPncp"),
        "dataAssinatura":            r.get("dataAssinatura"),
        "dataVigenciaInicio":        r.get("dataVigenciaInicio"),
        "dataVigenciaFim":           r.get("dataVigenciaFim"),
        "ufSigla":                   unidade.get("ufSigla", ""),
        "municipioNome":             unidade.get("municipioNome", ""),
        "nomeUnidade":               unidade.get("nomeUnidade", ""),
        "cnpjOrgao":                 orgao.get("cnpj", ""),
        "razaoSocialOrgao":          orgao.get("razaoSocial", ""),
        "esferaId":                  orgao.get("esferaId", ""),
        "poderId":                   orgao.get("poderId", ""),
        "niFornecedor":              r.get("niFornecedor", ""),
        "tipoPessoa":                r.get("tipoPessoa", ""),
        "nomeRazaoSocialFornecedor": r.get("nomeRazaoSocialFornecedor", ""),
        "tipoContratoId":            tipo.get("id"),
        "tipoContratoNome":          tipo.get("nome", ""),
    }


# ── Limpeza pós-download (gera coluna `rotulo` e `objeto`) ──────────────────
def limpar(df_raw):
    """
    Aplica limpeza, deriva colunas usadas pelo restante do pipeline:
      - `rotulo` (engenharia / obras / geral) a partir de categoriaProcessoId
      - `objeto` = objetoContrato + informacaoComplementar
      - `anoPublicacao`, `valor`, dtypes leves
    """
    df = df_raw.copy()
    if "categoriaProcessoId" not in df.columns:
        raise ValueError("coluna 'categoriaProcessoId' ausente — coleta inválida")

    # Filtra só categorias do escopo do TCC
    df = df[df["categoriaProcessoId"].isin(config.CATEGORIAS_INTERESSE)].copy()
    if df.empty:
        print("[coleta] nenhum registro nas categorias 7/8/9 — devolvendo vazio")
        return df

    df["rotulo"] = (df["categoriaProcessoId"].map(config.rotular)
                      .astype("category"))
    df["categoriaProcessoNome"] = df["categoriaProcessoId"].map(MAPA_CATEGORIA)

    # Texto principal — concatena objeto + complementar
    if "objetoContrato" in df.columns:
        obj = df["objetoContrato"].fillna("").astype(str)
        comp = df.get("informacaoComplementar",
                      pd.Series([""] * len(df))).fillna("").astype(str)
        df["objeto"] = (obj + " | " + comp).str.strip(" |")

    # Datas
    if "dataPublicacaoPncp" in df.columns:
        dt = pd.to_datetime(df["dataPublicacaoPncp"], errors="coerce")
        df["anoPublicacao"] = dt.dt.year.astype("Int16")
        df["mesPublicacao"] = dt.dt.month.astype("Int8")

    # Valor
    for col in ("valorInicial", "valorGlobal"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df["valor"] = df.get("valorGlobal").fillna(df.get("valorInicial"))

    # Esfera/poder legíveis
    if "esferaId" in df.columns:
        df["esferaNome"] = df["esferaId"].map(MAPA_ESFERA).fillna(df["esferaId"])
    if "poderId" in df.columns:
        df["poderNome"] = df["poderId"].map(MAPA_PODER).fillna(df["poderId"])

    df = df.reset_index(drop=True)
    return df


def _salvar_checkpoint(chk_path, todos):
    """Salva os registros coletados (em RAM) como parquet checkpoint."""
    if not todos:
        return
    try:
        df = pd.DataFrame(todos).drop_duplicates(subset=["numeroControlePNCP"])
        salvar_parquet(df, chk_path)
        print(f"   💾 checkpoint: {chk_path.name} ({len(df):,} regs)")
    except Exception as e:
        print(f"   ⚠ falha no checkpoint: {e}")


# ── Path do checkpoint e progresso ──────────────────────────────────────────
# Por que chavear por UF e não por range:
#   Se você roda coletar(uf='SP', anos=range(2024, 2026)) e depois
#   coletar(uf='SP', anos=range(2024, 2027)), o estado deve ser PRESERVADO.
#   Antes, era chaveado por range — mudou range, perdia progresso.
def _path_checkpoint(uf):
    return config.caminho(config.SUB_COLETA, f"checkpoint_{uf}.parquet")


def _path_progresso(uf):
    return config.caminho(config.SUB_COLETA, f"progresso_{uf}.json")


def _path_consolidado(uf=None):
    nome = f"contratos_{uf}.parquet" if uf else "contratos.parquet"
    return config.caminho(config.SUB_COLETA, nome)


# ── Frequência de checkpoint ─────────────────────────────────────────────────
# A cada N páginas grava o parquet em disco. Quanto menor, menos furo de
# dados em interrupção, mas mais I/O no Drive.
# 5 = compromisso: re-baixa no máx 4 páginas (~2000 regs) por interrupção.
CHECKPOINT_A_CADA = 5


def _ler_progresso(path, uf=None, ano_ini=None, ano_fim=None):
    """Estado de progresso por UF. Estrutura:

        {
          "meses_completos": ["2024-01", ...],      # baixados inteiros
          "meses_parciais": {                        # iniciados mas não terminados
            "2024-06": {"ultima_pagina_salva": 15, "registros": 7500}
          }
        }

    Faz migração transparente do formato antigo (que tinha "ultimo_parcial"
    único e era chaveado por range).
    """
    import json as _json

    if path.exists():
        try:
            d = _json.loads(path.read_text(encoding="utf-8"))
            # Migra: formato antigo tinha "ultimo_parcial" único
            if "meses_parciais" not in d and "ultimo_parcial" in d:
                ultimo = d.get("ultimo_parcial")
                d["meses_parciais"] = {}
                if ultimo:
                    d["meses_parciais"][ultimo["mes"]] = {
                        "ultima_pagina_salva": int(ultimo.get("ultima_pagina", 0)),
                        "registros": int(ultimo.get("registros", 0)),
                    }
                d.pop("ultimo_parcial", None)
            d.setdefault("meses_completos", [])
            d.setdefault("meses_parciais", {})
            return d
        except Exception:
            pass

    # Tenta migrar progresso de range antigo (chaveado por ano_ini/ano_fim)
    if uf and ano_ini and ano_fim:
        antigo = config.caminho(
            config.SUB_COLETA,
            f"progresso_{uf}_{ano_ini}_{ano_fim}.json",
        )
        if antigo.exists():
            try:
                d = _json.loads(antigo.read_text(encoding="utf-8"))
                if "meses_parciais" not in d and "ultimo_parcial" in d:
                    ultimo = d.get("ultimo_parcial")
                    d["meses_parciais"] = {}
                    if ultimo:
                        d["meses_parciais"][ultimo["mes"]] = {
                            "ultima_pagina_salva": int(ultimo.get("ultima_pagina", 0)),
                            "registros": int(ultimo.get("registros", 0)),
                        }
                    d.pop("ultimo_parcial", None)
                d.setdefault("meses_completos", [])
                d.setdefault("meses_parciais", {})
                print(f"   ↻ migrando progresso antigo {antigo.name} → {path.name}")
                return d
            except Exception:
                pass

    return {"meses_completos": [], "meses_parciais": {}}


def _salvar_progresso(path, prog):
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    prog_out = {
        "meses_completos": sorted(prog.get("meses_completos", [])),
        "meses_parciais": dict(sorted(
            (prog.get("meses_parciais") or {}).items())),
    }
    path.write_text(_json.dumps(prog_out, ensure_ascii=False, indent=2),
                     encoding="utf-8")


def status(uf, anos=None):
    """Mostra o estado atual da coleta para a UF."""
    prog = _ler_progresso(_path_progresso(uf), uf=uf)
    chk = _path_checkpoint(uf)

    print(f"\n📊 Status coleta {uf}")
    print(f"   meses completos: {len(prog['meses_completos'])}")
    if prog["meses_completos"]:
        for m in prog["meses_completos"]:
            print(f"     ✓ {m}")
    print(f"   meses parciais: {len(prog['meses_parciais'])}")
    for m, info in prog["meses_parciais"].items():
        print(f"     ↻ {m}: até pág {info['ultima_pagina_salva']} "
              f"({info.get('registros', 0):,} regs)")
    if chk.exists():
        df = pd.read_parquet(chk)
        print(f"   checkpoint: {len(df):,} registros em {chk.name}")
    else:
        print(f"   checkpoint: ainda não existe")

    if anos is not None:
        if isinstance(anos, int):
            anos = [anos]
        anos = sorted(list(anos))
        ano_ini, ano_fim = anos[0], anos[-1]
        pares = [(a, m) for a in range(ano_ini, ano_fim + 1) for m in range(1, 13)]
        chaves = [f"{a}-{m:02d}" for a, m in pares]
        completos = set(prog["meses_completos"])
        parciais = set(prog["meses_parciais"].keys())
        pendentes = [c for c in chaves if c not in completos]
        novos = [c for c in pendentes if c not in parciais]
        print(f"\n   No range {ano_ini}-{ano_fim}:")
        print(f"     • {len(chaves)} meses no total")
        print(f"     • {len(chaves) - len(pendentes)} já completos")
        print(f"     • {len([c for c in pendentes if c in parciais])} parciais")
        print(f"     • {len(novos)} a baixar do zero")
    return prog


def verificar_duplicatas(uf=None):
    """Conta duplicatas no checkpoint por numeroControlePNCP."""
    chk = _path_checkpoint(uf) if uf else _path_consolidado()
    if not chk.exists():
        cand = _path_consolidado(uf)
        if cand.exists():
            chk = cand
        else:
            print(f"[duplicatas] {chk} não existe")
            return None

    df = pd.read_parquet(chk)
    n_total = len(df)
    if "numeroControlePNCP" not in df.columns:
        print("[duplicatas] coluna numeroControlePNCP ausente")
        return None
    n_unico = df["numeroControlePNCP"].nunique()
    n_dup = n_total - n_unico
    print(f"\n🔍 Verificação de duplicatas em {chk.name}")
    print(f"   total de linhas:        {n_total:,}")
    print(f"   numeroControlePNCP únicos: {n_unico:,}")
    print(f"   duplicatas:             {n_dup:,}")
    if n_dup > 0:
        top = (df["numeroControlePNCP"].value_counts().head(5))
        print(f"   top NCPs duplicados:")
        for ncp, n in top.items():
            if n > 1:
                print(f"     {n}× {ncp}")
        print(f"   👉 use `pncp.coleta.deduplicar(uf='{uf}')` para limpar")
    return {"n_total": n_total, "n_unico": n_unico, "n_duplicado": n_dup}


def deduplicar(uf=None):
    """Remove duplicatas do checkpoint e do consolidado, mantendo a 1ª."""
    for cand in [_path_checkpoint(uf) if uf else None,
                 _path_consolidado(uf) if uf else None,
                 _path_consolidado()]:
        if cand and cand.exists():
            df = pd.read_parquet(cand)
            antes = len(df)
            df = df.drop_duplicates(subset=["numeroControlePNCP"], keep="first")
            depois = len(df)
            if depois < antes:
                salvar_parquet(df, cand)
                print(f"   🧹 {cand.name}: {antes:,} → {depois:,} "
                      f"(removidas {antes - depois:,} duplicatas)")


# ── Coleta principal ─────────────────────────────────────────────────────────
def coletar(uf, anos, mes_inicio=1, mes_fim=12, max_paginas=200, tamanho=500):
    """
    Coleta contratos via /v1/contratos, mês a mês, com retomada perfeita.

    Estado por UF (não por range): pode chamar com `range(2024, 2026)` e
    depois `range(2024, 2027)` que o progresso é preservado.

    Retomada por mês independente: cada mês tem seu próprio estado
    (completo / parcial / não iniciado). Mês com falhas é re-tentado
    na próxima execução, sem duplicar nem furar dados.
    """
    monitorar_ram("início coleta")

    if isinstance(anos, int):
        anos = [anos]
    anos = sorted(list(anos))
    ano_ini, ano_fim = anos[0], anos[-1]

    pares = []
    for a in range(ano_ini, ano_fim + 1):
        m_ini = mes_inicio if a == ano_ini else 1
        m_fim = mes_fim if a == ano_fim else 12
        for m in range(m_ini, m_fim + 1):
            pares.append((a, m))

    print(f"\n🔎 Coleta PNCP — {uf} | {ano_ini}-{ano_fim}  ({len(pares)} meses)")

    chk_path = _path_checkpoint(uf)
    prog_path = _path_progresso(uf)
    prog = _ler_progresso(prog_path, uf=uf, ano_ini=ano_ini, ano_fim=ano_fim)
    meses_completos = set(prog["meses_completos"])
    meses_parciais = dict(prog["meses_parciais"])

    todos = []
    if chk_path.exists():
        try:
            todos = pd.read_parquet(chk_path).to_dict("records")
            print(f"   ↻ checkpoint: {len(todos):,} registros já em disco")
        except Exception:
            todos = []

    chaves_range = [f"{a}-{m:02d}" for a, m in pares]
    a_baixar = [c for c in chaves_range if c not in meses_completos]
    a_retomar = [c for c in a_baixar if c in meses_parciais]
    a_novo = [c for c in a_baixar if c not in meses_parciais]
    print(f"   ✓ {len(chaves_range) - len(a_baixar)} mês(es) já completos — pulando")
    if a_retomar:
        print(f"   ↻ {len(a_retomar)} mês(es) parciais a retomar:")
        for c in a_retomar:
            info = meses_parciais[c]
            print(f"     {c}: retomar da pág {info['ultima_pagina_salva'] + 1} "
                  f"(já tem {info.get('registros', 0):,} regs em disco)")
    if a_novo:
        print(f"   → {len(a_novo)} mês(es) novos para baixar do zero")

    total_meses_a_processar = len(a_baixar)
    idx_processado = 0

    for (a, mes) in pares:
        chave = f"{a}-{mes:02d}"
        if chave in meses_completos:
            continue
        idx_processado += 1

        d_ini = datetime.date(a, mes, 1).strftime("%Y%m%d")
        if mes < 12:
            d_fim = datetime.date(a, mes + 1, 1) - datetime.timedelta(days=1)
        else:
            d_fim = datetime.date(a, 12, 31)
        d_fim = d_fim.strftime("%Y%m%d")

        info_parcial = meses_parciais.get(chave)
        pag_inicio = (info_parcial["ultima_pagina_salva"] + 1
                      if info_parcial else 1)
        reg_anteriores = (info_parcial.get("registros", 0)
                          if info_parcial else 0)

        cabecalho = (f"\n── [{idx_processado}/{total_meses_a_processar}] "
                     f"{chave} ({d_ini} → {d_fim})")
        if info_parcial:
            print(f"{cabecalho} RETOMANDO da pág {pag_inicio} "
                  f"({reg_anteriores:,} regs já em disco) ──")
        else:
            print(f"{cabecalho} ──")

        reg_mes_novo = 0
        ultima_pag_ok = pag_inicio - 1
        ultima_pag_salva = info_parcial["ultima_pagina_salva"] if info_parcial else 0
        falhas_seguidas = 0
        mes_terminou_natural = False

        for pag in range(pag_inicio, max_paginas + 1):
            params = {
                "dataInicial": d_ini,
                "dataFinal": d_fim,
                "pagina": pag,
                "tamanhoPagina": tamanho,
            }
            r = _get_com_retry(f"{config.API_BASE}/v1/contratos", params)

            if r is None:
                falhas_seguidas += 1
                print(f"   ✗ pág {pag:3d}: falha — pulando")
                if falhas_seguidas >= 5:
                    print(f"   ⚠ 5 falhas seguidas — abandonando o mês "
                          f"(retomado na próxima sessão)")
                    break
                continue

            if r.status_code == 204:
                print(f"   ✓ pág {pag} 204 — fim do mês")
                mes_terminou_natural = True
                break

            try:
                payload = r.json()
            except ValueError:
                falhas_seguidas += 1
                continue

            registros = (payload.get("data", []) if isinstance(payload, dict)
                          else payload if isinstance(payload, list) else [])
            if not registros:
                print(f"   ✓ pág {pag} vazia — fim do mês")
                mes_terminou_natural = True
                break

            todos.extend(_aplanar(r) for r in registros)
            reg_mes_novo += len(registros)
            ultima_pag_ok = pag
            falhas_seguidas = 0
            print(f"   → pág {pag:3d}: +{len(registros):4d} "
                  f"(mês: {reg_anteriores + reg_mes_novo:,} | "
                  f"global: {len(todos):,})")

            if pag % CHECKPOINT_A_CADA == 0:
                _salvar_checkpoint(chk_path, todos)
                ultima_pag_salva = pag
                meses_parciais[chave] = {
                    "ultima_pagina_salva": ultima_pag_salva,
                    "registros": reg_anteriores + reg_mes_novo,
                }
                _salvar_progresso(prog_path, {
                    "meses_completos": sorted(meses_completos),
                    "meses_parciais": meses_parciais,
                })

            if len(registros) < tamanho:
                print(f"   ✓ pág {pag} parcial — fim do mês")
                mes_terminou_natural = True
                break
            time.sleep(config.PAUSA_PAGINA)

        _salvar_checkpoint(chk_path, todos)
        ultima_pag_salva = ultima_pag_ok

        if mes_terminou_natural:
            meses_completos.add(chave)
            meses_parciais.pop(chave, None)
            print(f"   ✅ {chave} COMPLETO ({reg_anteriores + reg_mes_novo:,} regs)")
        else:
            meses_parciais[chave] = {
                "ultima_pagina_salva": ultima_pag_salva,
                "registros": reg_anteriores + reg_mes_novo,
            }
            print(f"   ⏸ {chave} PARCIAL (até pág {ultima_pag_salva}, "
                  f"{reg_anteriores + reg_mes_novo:,} regs) — "
                  f"retomar próxima sessão")

        _salvar_progresso(prog_path, {
            "meses_completos": sorted(meses_completos),
            "meses_parciais": meses_parciais,
        })

    if not todos:
        print("[coleta] nada baixado — verifique conexão e parâmetros")
        return None

    df = pd.DataFrame(todos).drop_duplicates(subset=["numeroControlePNCP"])
    print(f"\n[coleta] bruto: {len(df):,} registros antes do filtro UF")

    if uf and "ufSigla" in df.columns:
        antes = len(df)
        df = df[df["ufSigla"].astype(str).str.upper() == uf.upper()].copy()
        print(f"[coleta] filtro UF='{uf}': {antes:,} → {len(df):,}")

    df = limpar(df)
    if df.empty:
        return None

    saida_uf = _path_consolidado(uf)
    salvar_parquet(df, saida_uf)
    saida_geral = _path_consolidado()
    salvar_parquet(df, saida_geral)
    print(f"\n✅ consolidado em {saida_geral} ({len(df):,} contratos limpos)")
    print(f"   distribuição: {df['rotulo'].value_counts().to_dict()}")

    liberar(todos, df)
    monitorar_ram("fim coleta")
    return saida_geral


# ── Modo interativo ──────────────────────────────────────────────────────────
def _pedir_int(msg, padrao, mi, ma):
    while True:
        r = input(f"  {msg} [{padrao}]: ").strip()
        if r == "":
            return padrao
        try:
            v = int(r)
            if mi <= v <= ma:
                return v
            print(f"    ⚠ entre {mi} e {ma}")
        except ValueError:
            print("    ⚠ inteiros apenas")


def _pedir_texto(msg, padrao=""):
    r = input(f"  {msg} [{padrao or 'Enter = sem filtro'}]: ").strip()
    return r if r else padrao


def coletar_interativo():
    """Pergunta os parâmetros no terminal e dispara a coleta."""
    print("\n" + "═" * 62)
    print("  CONFIGURAÇÃO — API PNCP (/v1/contratos)")
    print("═" * 62)
    ano_ini = _pedir_int("Ano inicial (2022-2026)", 2024, 2022, 2026)
    ano_fim = _pedir_int(f"Ano final (≥ {ano_ini})", ano_ini, ano_ini, 2026)
    mi = _pedir_int("Mês inicial do primeiro ano (1-12)", 1, 1, 12)
    mf = _pedir_int("Mês final do último ano (1-12)", 12, 1, 12)
    mp = _pedir_int("Máx. páginas por mês", 200, 1, 500)
    ta = _pedir_int("Registros por página (1-500)", 500, 1, 500)
    uf = _pedir_texto("UF (obrigatório)", "SP").upper()

    n_meses = (ano_fim - ano_ini - 1) * 12 + (12 - mi + 1) + mf \
              if ano_fim > ano_ini else (mf - mi + 1)
    print(f"\n  → {n_meses} meses serão coletados ({uf})")
    if input("\n  Iniciar? [S/n]: ").strip().lower() == "n":
        print("Cancelado.")
        return None

    return coletar(uf=uf, anos=range(ano_ini, ano_fim + 1),
                   mes_inicio=mi, mes_fim=mf,
                   max_paginas=mp, tamanho=ta)


# ── Recuperação após queda do kernel ─────────────────────────────────────────
def carregar_checkpoint(uf=None, caminho=None):
    """Carrega o checkpoint mais recente (ou um path específico) e devolve
    o DataFrame já limpo."""
    if caminho is None:
        if uf:
            cand = _path_consolidado(uf)
            if cand.exists():
                caminho = cand
        if caminho is None:
            geral = _path_consolidado()
            if geral.exists():
                caminho = geral
        if caminho is None and uf:
            chk_uf = _path_checkpoint(uf)
            if chk_uf.exists():
                caminho = chk_uf
        if caminho is None:
            pasta = config.PASTA_DADOS / config.SUB_COLETA
            if pasta.exists():
                checkpoints = sorted(
                    pasta.glob("checkpoint_*.parquet"),
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                if checkpoints:
                    caminho = checkpoints[0]

    if caminho is None or not Path(caminho).exists():
        raise FileNotFoundError(
            "Nenhum parquet de coleta encontrado. "
            "Rode `pncp.coleta.coletar(...)` primeiro."
        )

    print(f"📥 carregando: {caminho}")
    df = pd.read_parquet(caminho)
    if uf and "ufSigla" in df.columns:
        antes = len(df)
        df = df[df["ufSigla"].astype(str).str.upper() == uf.upper()].copy()
        print(f"   filtro UF='{uf}': {antes:,} → {len(df):,}")

    if "rotulo" not in df.columns:
        df = limpar(df)
    print(f"   ✓ {len(df):,} contratos prontos")
    return df


# ── Combina parquets parciais ────────────────────────────────────────────────
def combinar_parquets(uf=None, padrao="*.parquet", salvar_em=None):
    """Junta todos os parquets da pasta de coleta num único DataFrame
    deduplicado e limpo."""
    pasta = config.PASTA_DADOS / config.SUB_COLETA
    if not pasta.exists():
        raise FileNotFoundError(f"pasta não existe: {pasta}")

    arquivos = sorted(pasta.glob(padrao))
    arquivos = [a for a in arquivos
                 if a.name not in ("contratos.parquet",
                                    f"contratos_{uf}.parquet" if uf else "")]
    if not arquivos:
        raise FileNotFoundError(f"nada casando com '{padrao}' em {pasta}")

    print(f"\n📂 combinando {len(arquivos)} parquet(s):")
    for a in arquivos:
        print(f"   • {a.name}")

    pedacos = []
    for a in arquivos:
        try:
            d = pd.read_parquet(a)
            pedacos.append(d)
        except Exception as e:
            print(f"   ⚠ pulando {a.name}: {e}")

    if not pedacos:
        raise RuntimeError("nenhum arquivo pôde ser lido")

    df = pd.concat(pedacos, ignore_index=True)
    print(f"   📊 concatenado: {len(df):,} linhas (com duplicatas)")

    if uf and "ufSigla" in df.columns:
        antes = len(df)
        df = df[df["ufSigla"].astype(str).str.upper() == uf.upper()].copy()
        print(f"   🗺  filtro UF='{uf}': {antes:,} → {len(df):,}")

    if "numeroControlePNCP" in df.columns:
        antes = len(df)
        df = df.drop_duplicates(subset=["numeroControlePNCP"], keep="first")
        print(f"   🧹 dedup: {antes:,} → {len(df):,}")

    if "rotulo" not in df.columns:
        df = limpar(df)

    if salvar_em is None:
        salvar_em = _path_consolidado(uf) if uf else _path_consolidado()
    salvar_parquet(df, salvar_em)

    if uf:
        salvar_parquet(df, _path_consolidado())
    print(f"   ✅ salvo em {salvar_em}")
    return df


# ── Filtro temporal (útil após EDA detectar viés) ────────────────────────────
def filtrar_anos(caminho_parquet, ano_minimo=None, ano_maximo=None):
    """Reescreve o parquet aplicando filtro de anos (útil após EDA)."""
    df = ler_parquet(caminho_parquet)
    n_antes = len(df)
    if ano_minimo is not None:
        df = df[df["anoPublicacao"] >= ano_minimo]
    if ano_maximo is not None:
        df = df[df["anoPublicacao"] <= ano_maximo]
    salvar_parquet(df, caminho_parquet)
    print(f"[coleta] filtro temporal: {n_antes:,} → {len(df):,} contratos")
    liberar(df)
    return caminho_parquet
