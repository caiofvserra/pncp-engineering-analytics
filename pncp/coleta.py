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


# ── Path do checkpoint ───────────────────────────────────────────────────────
def _path_checkpoint(uf, ano_ini, ano_fim):
    nome = f"checkpoint_{uf}_{ano_ini}_{ano_fim}.parquet"
    return config.caminho(config.SUB_COLETA, nome)


def _path_consolidado(uf=None):
    nome = f"contratos_{uf}.parquet" if uf else "contratos.parquet"
    return config.caminho(config.SUB_COLETA, nome)


# ── Coleta principal ─────────────────────────────────────────────────────────
def coletar(uf, anos, mes_inicio=1, mes_fim=12, max_paginas=200, tamanho=500):
    """
    Coleta contratos via /v1/contratos, mês a mês, com checkpoint mensal.

    Args:
        uf: sigla da UF para filtro pós-download (ex: 'SP')
        anos: int (ano único) ou iterável de anos (ex: range(2023, 2026))
        mes_inicio, mes_fim: limites mensais (aplicados aos anos das pontas)
        max_paginas: limite de páginas por mês (200 = até 100k registros/mês)
        tamanho: registros por página (máx 500 — limite da API)

    Returns:
        Path do parquet final consolidado e limpo.
        Em caso de falha total na coleta, retorna None.
    """
    monitorar_ram("início coleta")

    if isinstance(anos, int):
        anos = [anos]
    anos = sorted(list(anos))
    ano_ini, ano_fim = anos[0], anos[-1]

    # Constrói lista (ano, mês)
    pares = []
    for a in range(ano_ini, ano_fim + 1):
        m_ini = mes_inicio if a == ano_ini else 1
        m_fim = mes_fim if a == ano_fim else 12
        for m in range(m_ini, m_fim + 1):
            pares.append((a, m))

    print(f"\n🔎 Coleta PNCP — {uf} | {ano_ini}-{ano_fim}  "
          f"({len(pares)} meses)")

    todos = []
    chk_path = _path_checkpoint(uf, ano_ini, ano_fim)

    # Retoma de checkpoint se existir
    if chk_path.exists():
        try:
            anteriores = pd.read_parquet(chk_path)
            todos = anteriores.to_dict("records")
            print(f"   ↻ retomando de checkpoint ({len(todos):,} registros já)")
        except Exception:
            todos = []

    for (a, mes) in tqdm(pares, desc="📅 Meses", unit="mês"):
        d_ini = datetime.date(a, mes, 1).strftime("%Y%m%d")
        if mes < 12:
            d_fim = (datetime.date(a, mes + 1, 1) - datetime.timedelta(days=1))
        else:
            d_fim = datetime.date(a, 12, 31)
        d_fim = d_fim.strftime("%Y%m%d")

        print(f"\n── {a}-{mes:02d} ({d_ini} → {d_fim}) ──")
        reg_mes = 0
        falhas_seguidas = 0

        for pag in range(1, max_paginas + 1):
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
                    print(f"   ⚠ 5 falhas seguidas — abandonando o mês")
                    break
                continue

            if r.status_code == 204:
                print(f"   ✓ pág {pag} 204 — fim do mês")
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
                break

            todos.extend(_aplanar(r) for r in registros)
            reg_mes += len(registros)
            falhas_seguidas = 0
            print(f"   → pág {pag:3d}: +{len(registros):4d} (acum: {len(todos):,})")
            time.sleep(config.PAUSA_PAGINA)

        print(f"   {a}-{mes:02d}: {reg_mes:,} registros")

        # Checkpoint mensal
        if todos:
            df_chk = (pd.DataFrame(todos)
                      .drop_duplicates(subset=["numeroControlePNCP"]))
            try:
                salvar_parquet(df_chk, chk_path)
                print(f"   💾 checkpoint: {chk_path.name} "
                      f"({len(df_chk):,} regs)")
            except Exception as e:
                print(f"   ⚠ falha no checkpoint: {e}")

    if not todos:
        print("[coleta] nada baixado — verifique conexão e parâmetros")
        return None

    # ── Pós-processamento: dedup, filtro UF, limpeza ──────────────────────
    df = pd.DataFrame(todos).drop_duplicates(subset=["numeroControlePNCP"])
    print(f"\n[coleta] bruto: {len(df):,} registros antes do filtro UF")

    if uf and "ufSigla" in df.columns:
        antes = len(df)
        df = df[df["ufSigla"].astype(str).str.upper() == uf.upper()].copy()
        print(f"[coleta] filtro UF='{uf}': {antes:,} → {len(df):,}")

    df = limpar(df)
    if df.empty:
        return None

    # Salva consolidado (com UF no nome — facilita combinar várias depois)
    saida_uf = _path_consolidado(uf)
    salvar_parquet(df, saida_uf)
    # E também como contratos.parquet (esperado pelos demais módulos)
    saida_geral = _path_consolidado()
    salvar_parquet(df, saida_geral)
    print(f"\n✅ consolidado em {saida_geral} ({len(df):,} contratos limpos)")
    print(f"   distribuição: {df['rotulo'].value_counts().to_dict()}")

    liberar(todos, df)
    monitorar_ram("fim coleta")
    return saida_geral


# ── Modo interativo (para uso no Colab) ──────────────────────────────────────
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
    """
    Carrega o checkpoint mais recente (ou um path específico) e devolve
    o DataFrame já limpo.

    Args:
        uf: filtra por UF se houver mistura no arquivo
        caminho: path específico (sobrescreve auto-detecção)
    """
    if caminho is None:
        # 1º tenta o consolidado da UF
        if uf:
            cand = _path_consolidado(uf)
            if cand.exists():
                caminho = cand
        # 2º o consolidado geral
        if caminho is None:
            geral = _path_consolidado()
            if geral.exists():
                caminho = geral
        # 3º o checkpoint mais recente
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


# ── Combina parquets parciais (útil quando se coleta ano a ano) ─────────────
def combinar_parquets(uf=None, padrao="*.parquet", salvar_em=None):
    """
    Junta todos os parquets da pasta de coleta num único DataFrame
    deduplicado e limpo. Útil quando se rodou `coletar` várias vezes
    (ex: 1 ano por sessão).

    Args:
        uf: filtra por UF (recomendado)
        padrao: glob para os arquivos a juntar
        salvar_em: path do parquet final (default = contratos.parquet)
    """
    pasta = config.PASTA_DADOS / config.SUB_COLETA
    if not pasta.exists():
        raise FileNotFoundError(f"pasta não existe: {pasta}")

    arquivos = sorted(pasta.glob(padrao))
    # Não inclui o consolidado final no merge para evitar duplicação
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

    # Também salva como contratos.parquet (esperado pelos outros módulos)
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
