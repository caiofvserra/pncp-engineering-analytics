"""
pncp_coleta.py — Módulo de coleta da API do PNCP

Este módulo concentra TUDO relacionado à coleta de dados:
  • Configurações da API e parâmetros de download
  • Tabelas de domínio (categorias, modalidades, critérios)
  • Tokenização e stopwords (necessárias para a limpeza)
  • Função principal: baixar_contratacoes_pncp_por_uf()
  • Limpeza/normalização: carregar_e_limpar()
  • Persistência: Drive, checkpoints, retomada de coleta
  • Combinação de coletas multi-sessão: combinar_parquets()

SEPARAÇÃO INTENCIONAL: Este módulo NÃO depende de pncp_analise.py.
Você pode usar este módulo isoladamente para baixar dados, e só depois
carregar pncp_analise.py para fazer análises. Isso evita recarregar 10k
linhas quando você está iterando apenas em análises.

USO TÍPICO NO COLAB
───────────────────
    %run pncp_coleta.py
    montar_drive()
    keep_alive_javascript()
    df = executar_apenas_coleta(modo_interativo=True)
    # Em outras sessões, pode juntar coletas separadas:
    df = combinar_parquets(uf_filtro="SP")

E SÓ ENTÃO carregar o módulo de análise pesado:
    %run pncp_analise.py
    eda_res = executar_apenas_eda(df)
    p2 = executar_parte2(df)
    # ...

INTEGRAÇÃO COM pncp_analise.py
──────────────────────────────
O pncp_analise.py importa este módulo no início para reutilizar
constantes e funções de coleta sem duplicar código:
    from pncp_coleta import (
        carregar_e_limpar, montar_drive, executar_apenas_coleta,
        combinar_parquets, carregar_checkpoint,
        STOPWORDS_CONTRATOS, MAPA_CATEGORIA, ...
    )
"""

# ════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════

import os
import re
import sys
import time
import json
import datetime
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Detecta ambiente Colab para ajustar comportamento (Drive, keep-alive)
try:
    import google.colab  # noqa: F401
    EM_COLAB = True
except ImportError:
    EM_COLAB = False

# tqdm com fallback (caso não esteja instalado, mostra apenas mensagens)
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return it

# RSLP Stemmer (NLTK) — usado em tokenizar() quando stem=True
try:
    import nltk
    try:
        nltk.data.find("stemmers/rslp")
    except LookupError:
        nltk.download("rslp", quiet=True)
    from nltk.stem import RSLPStemmer
    _STEMMER_PT = RSLPStemmer()
except Exception:
    _STEMMER_PT = None


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES PADRÃO (sobrescreva via parâmetros das funções)
# ════════════════════════════════════════════════════════════════════════════

ANO_INICIO  = 2024
ANO_FIM     = 2024
MES_INICIO  = 1
MES_FIM     = 12
MAX_PAGINAS = 100      # páginas máximas por mês (cada página = TAMANHO contratos)
TAMANHO     = 500      # registros por página (limite da API)
UF          = "SP"

BASE_URL = "https://pncp.gov.br/api/consulta"


# ════════════════════════════════════════════════════════════════════════════
# TABELAS DE DOMÍNIO (Manual PNCP v1.0)
# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — Tabelas de domínio (Manual PNCP v1.0)
# ════════════════════════════════════════════════════════════════════════════
# Estas tabelas mapeiam os IDs numéricos retornados pela API para nomes
# legíveis. Os IDs são fixos no Portal — não mudam.

# Categoria do processo (= rótulo do TCC).
# As três categorias relevantes para o problema:
#   • 7 (Obras)            → engenharia (positivo)  — privativa CREA/CAU
#   • 8 (Serviços)         → geral (negativo)       — domínio do subenquadramento
#   • 9 (Serv. Engenharia) → engenharia (positivo)  — engenharia "pura"
MAPA_CATEGORIA = {
    1: "Cessão", 2: "Compras", 3: "Informática/TIC", 4: "Internacional",
    5: "Locação Imóveis", 6: "Mão de Obra", 7: "Obras", 8: "Serviços",
    9: "Serv. Engenharia", 10: "Serv. Saúde", 11: "Alienação",
}

# Modalidade (Lei 14.133/2021 art. 28). Para engenharia:
#   • Concorrência (4, 5):  preferencial p/ obras e serv. eng. especiais
#   • Pregão (6, 7):        OK só p/ serv. comuns de eng. (art. 6º XXI)
#                           VEDADO p/ obras (art. 29 § único)
#   • Diálogo Competit. (2): casos complexos
MAPA_MODALIDADE = {
    1: "Leilão - Eletr.", 2: "Diálogo Competit.", 3: "Concurso",
    4: "Concorrência - Eletr.", 5: "Concorrência - Pres.",
    6: "Pregão - Eletr.", 7: "Pregão - Pres.", 8: "Dispensa",
    9: "Inexigibilidade", 10: "Manif. Interesse",
    11: "Pré-qualificação", 12: "Credenciamento", 13: "Leilão - Pres.",
}

# Critério de julgamento (Lei 14.133/2021 art. 33).
# Concorrência admite: 1, 2, 4, 6, 8 (cinco critérios)
# Pregão admite:        1, 2          (dois critérios)
MAPA_CRITERIO = {
    1: "Menor preço", 2: "Maior desconto", 4: "Técnica e preço",
    5: "Maior lance", 6: "Maior ret. econôm.", 7: "Não se aplica",
    8: "Melhor técnica", 9: "Conteúdo artístico",
}
MAPA_ESFERA = {"F": "Federal", "E": "Estadual", "M": "Municipal", "D": "Distrital"}
MAPA_PODER  = {"E": "Executivo", "L": "Legislativo", "J": "Judiciário"}

# Mapeamento dos rótulos do classificador (binário engenharia × geral).
# Lei 14.133/2021 art. 6º XII define OBRA como "toda atividade estabelecida,
# por força de lei, como privativa das profissões de arquiteto e engenheiro".
# Por isso, a categoria 7 (Obras) é tratada como ENGENHARIA junto com a 9
# (Serviços de Engenharia). A categoria 8 (Serviços) permanece como GERAL.
MAPA_ROTULO = {7: "engenharia", 8: "geral", 9: "engenharia"}
CATEGORIAS_CONSIDERADAS = list(MAPA_ROTULO.keys())   # [7, 8, 9]

# Paleta de cores usada nos gráficos (consistente entre módulos)
PALETA = {"engenharia": "#1a6faf", "geral": "#e07b39"}


# ════════════════════════════════════════════════════════════════════════════
# STOPWORDS + TOKENIZAÇÃO
# ════════════════════════════════════════════════════════════════════════════

STOPWORDS_PT = set("""
a ao aos aquela aquelas aquele aqueles aquilo as até
com como da das de dela delas dele deles depois
do dos e ela elas ele eles em entre era eram essa
essas esse esses esta estar estas este estou estes
eu foi for forma foi há isso isto já lhe lhes lo
mais mas me mesmo meu minha minha muito na não nas
nos nossa nossas nosso nossos num numa
o os ou para pela pelas pelo pelos por qual quando
que quem se sem ser seu seus sim sob sobre também
te tem têm teu tua tuas tudo um uma uns umas você vocês à às
""".split())

STOPWORDS_CONTRATOS = STOPWORDS_PT | {
    # Genéricos do domínio de contratações
    "contratação","contratacao","empresa","serviço","serviços",
    "servico","servicos","prestação","prestacao","execução","execucao",
    "objeto","conforme","referência","referencia","termo","edital",
    "contrato","mediante","visando","atender","demanda","necessidade",
    "órgão","orgao","especializada","especializado","pessoa","juridica",
    "fornecimento","aquisição","aquisicao","compra","compras",
    # Adicionados: termos geográficos/administrativos sem poder discriminativo
    # (aparecem em geral E em engenharia, então não ajudam a separar as classes)
    "sao","paulo","municipal","municipio","municipios",
    "prefeitura","secretaria","distrito","estadual","federal",
    "dia","dias","mes","mes","ano","anos",
}

def _normalizar(t: str) -> str:
    """Lowercase + remove acentos via NFD + filtro de combining marks."""
    t = t.lower()
    return "".join(c for c in unicodedata.normalize("NFD", t)
                   if unicodedata.category(c) != "Mn")


def tokenizar(texto: str, min_len: int = 3, stem: bool = False) -> list:
    """
    Tokeniza texto em português: normaliza, remove stopwords, opcionalmente stemmiza.

    O resultado é uma lista de palavras já minúsculas, sem acentos, sem
    pontuação, sem stopwords genéricas e do domínio (STOPWORDS_CONTRATOS),
    e com pelo menos `min_len` caracteres.

    Parâmetros
    ──────────
    stem : se True, aplica RSLP Stemmer (NLTK) — reduz "manutenção", "manutenções"
           e "manutencoes" para o radical "manuten" comum (Aula quinzena 01).
           Útil para gráficos de frequência e reduzir esparsidade no vocabulário
           do TF-IDF em datasets pequenos. NÃO use em comparação com KEYWORDS_ENG
           (lista fixa de palavras inteiras).
    """
    # _normalizar já removeu acentos: regex captura apenas [a-z]+
    tokens = re.findall(r"[a-z]+", _normalizar(str(texto)))
    tokens = [t for t in tokens if len(t) >= min_len and t not in STOPWORDS_CONTRATOS]
    if stem and _STEMMER_PT is not None:
        tokens = [_STEMMER_PT.stem(t) for t in tokens]
    return tokens


def bigramas(tokens: list) -> list:
    """Gera bigramas como strings 'palavra1 palavra2' a partir de uma lista."""
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

# ════════════════════════════════════════════════════════════════════════════
# API PNCP — endpoint /v1/contratos com filtro UF em Python
# ════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# ████████████████████████   PARTE 1 — COLETA + EDA   ███████████████████████
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — API PNCP — endpoint /v1/contratos com filtro UF em Python
# ════════════════════════════════════════════════════════════════════════════
#
# HISTÓRICO DA ESCOLHA DO ENDPOINT
# ────────────────────────────────
# A versão anterior usava /v1/contratacoes/publicacao com filtro `uf` direto
# na API. Em testes no Colab, o servidor PNCP entregava timeouts sistemáticos
# quando combinávamos `uf` + `codigoModalidadeContratacao` — cada chamada que
# devia levar 3s passava de 60s e dava ReadTimeoutError.
#
# Solução: voltar à estratégia original com /v1/contratos, que:
#   • NÃO exige iterar por modalidade (1 chamada por mês × página)
#   • É rápido (servidor PNCP responde em segundos)
#   • Traz `categoriaProcessoId` (rótulo do TCC) e `objetoContrato`
#   • Não traz UF — filtramos em Python depois do download
#
# Trade-off honesto: /v1/contratos NÃO traz `modalidadeId` nem
# `criterioJulgamentoId`. O classificador funciona normalmente sem eles
# (objetoContrato é a feature principal de NLP), mas alguns gráficos
# secundários da EDA mostrarão "modalidade desconhecida" quando ausente.
# ────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://pncp.gov.br/api/consulta"
MODALIDADES_DISPUTA = list(range(1, 14))  # mantido para compatibilidade


def _get_com_retry(url: str, params: dict, tentativas: int = 6, espera: float = 4.0,
                    timeout: int = 60):
    """
    GET com retentativas exponenciais para timeouts e erros 5xx.

    Configuração ajustada após observação dos logs reais:
      • 6 tentativas (antes 3) — servidor PNCP fica intermitentemente lento
      • timeout 60s (antes 30s) — algumas páginas demoram mesmo
      • Backoff exponencial: 4s, 8s, 16s, 32s, 64s (entre tentativas)
        Total potencial de espera: até ~2min antes de desistir da página.
    """
    for i in range(1, tentativas + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200: return resp
            if resp.status_code == 204: return resp
            if resp.status_code in (400, 422): return None
            if resp.status_code >= 500:
                # Backoff exponencial em servidor sobrecarregado
                espera_atual = espera * (2 ** (i - 1))
                print(f"   ⚠ HTTP {resp.status_code} — tentativa {i}/{tentativas}, "
                      f"aguardando {espera_atual:.0f}s")
                time.sleep(espera_atual)
            else:
                print(f"   ⚠ HTTP {resp.status_code} inesperado.")
                return None
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            espera_atual = espera * (2 ** (i - 1))
            print(f"   ⚠ {type(e).__name__} — tentativa {i}/{tentativas}, "
                  f"aguardando {espera_atual:.0f}s")
            time.sleep(espera_atual)
    return None


def _aplanar_contrato(r: dict) -> dict:
    """Achata um registro de /v1/contratos para um dict plano."""
    cat     = r.get("categoriaProcesso") or {}
    unidade = r.get("unidadeOrgao")      or {}
    orgao   = r.get("orgaoEntidade")     or {}
    tipo    = r.get("tipoContrato")      or {}

    return {
        "numeroControlePNCP":      r.get("numeroControlePNCP", ""),
        # Rótulo (chave do problema de classificação)
        "categoriaProcessoId":     cat.get("id"),
        "categoriaProcessoNome":   cat.get("nome", ""),
        # Texto principal para NLP
        "objetoContrato":          r.get("objetoContrato", ""),
        "informacaoComplementar":  r.get("informacaoComplementar", ""),
        # Valores
        "valorInicial":            r.get("valorInicial"),
        "valorGlobal":             r.get("valorGlobal"),
        "valorAcumulado":          r.get("valorAcumulado"),
        "valorTotalEstimado":      r.get("valorInicial"),    # alias
        "valorTotalHomologado":    r.get("valorGlobal"),     # alias
        # Datas
        "dataPublicacaoPncp":      r.get("dataPublicacaoPncp"),
        "dataAssinatura":          r.get("dataAssinatura"),
        "dataVigenciaInicio":      r.get("dataVigenciaInicio"),
        "dataVigenciaFim":         r.get("dataVigenciaFim"),
        # Localização
        "ufSigla":                 unidade.get("ufSigla", ""),
        "municipioNome":           unidade.get("municipioNome", ""),
        "nomeUnidade":             unidade.get("nomeUnidade", ""),
        # Órgão (CONTRATANTE — quem compra)
        "cnpjOrgao":               orgao.get("cnpj", ""),
        "razaoSocialOrgao":        orgao.get("razaoSocial", ""),
        "esferaId":                orgao.get("esferaId", ""),
        "poderId":                 orgao.get("poderId", ""),
        # ── Fornecedor (CONTRATADO — quem vende) ──────────────────────────
        # Necessário para análise de grafos órgão↔fornecedor (Parte 7)
        "niFornecedor":             r.get("niFornecedor", ""),
        "tipoPessoa":               r.get("tipoPessoa", ""),
        "nomeRazaoSocialFornecedor": r.get("nomeRazaoSocialFornecedor", ""),
        # Subcontratação (raro mas existe)
        "niFornecedorSubContratado":   r.get("niFornecedorSubContratado", ""),
        "nomeFornecedorSubContratado": r.get("nomeFornecedorSubContratado", ""),
        # Tipo de instrumento
        "tipoContratoId":          tipo.get("id"),
        "tipoContratoNome":        tipo.get("nome", ""),
    }


# Alias para compatibilidade com código antigo
_aplanar_contratacao = _aplanar_contrato


def baixar_contratacoes_pncp_por_uf(ano, uf, mes_inicio=1, mes_fim=12,
                                     max_paginas=100, tamanho=500,
                                     modalidades=None, ano_fim=None):
    """
    Baixa contratos do PNCP usando o endpoint estável /v1/contratos.

    Estratégia anti-timeout:
      1. Itera mês a mês (janelas curtas evitam timeouts)
      2. Para cada mês, pagina até max_paginas ou até retornar vazio
      3. Filtro por UF aplicado em Python após o download

    Parâmetros
    ──────────
    ano         : ano de publicação inicial (ex.: 2022)
    uf          : UF para filtro pós-download (ex.: 'SP')
    mes_inicio  : mês inicial (1-12, aplicado ao primeiro ano)
    mes_fim     : mês final (1-12, aplicado ao último ano)
    max_paginas : limite de páginas por mês (default 100 = até 50.000 reg/mês)
    tamanho     : registros por página (máx 500 — limite da API)
    modalidades : ignorado (mantido p/ compatibilidade)
    ano_fim     : se fornecido, coleta de `ano` a `ano_fim` inclusive.
                   Quando None, coleta apenas o ano único `ano`.
    """
    if ano_fim is None:
        ano_fim = ano

    if ano_fim < ano:
        print(f"⚠ ano_fim ({ano_fim}) < ano_inicio ({ano}) — invertendo.")
        ano, ano_fim = ano_fim, ano

    # Constrói lista de (ano, mes) considerando o intervalo
    pares_ano_mes = []
    for a in range(ano, ano_fim + 1):
        m_ini = mes_inicio if a == ano     else 1
        m_fim = mes_fim    if a == ano_fim else 12
        for m in range(m_ini, m_fim + 1):
            pares_ano_mes.append((a, m))

    todos = []
    total_reqs = 0
    print(f"\n🔎 Baixando contratos do PNCP")
    if ano == ano_fim:
        print(f"   Ano: {ano} | meses {mes_inicio}-{mes_fim}")
    else:
        print(f"   Anos: {ano}–{ano_fim}  (total: {len(pares_ano_mes)} meses)")
    print(f"   Endpoint: /v1/contratos | filtro UF='{uf}' aplicado em Python")
    print(f"   Configuração: max_paginas={max_paginas}, tamanho={tamanho}")

    for (a, mes) in tqdm(pares_ano_mes, desc="📅 Meses", unit="mês"):
        d_ini = datetime.date(a, mes, 1).strftime("%Y%m%d")
        d_fim = (datetime.date(a, mes + 1, 1) - datetime.timedelta(days=1)
                 if mes < 12 else datetime.date(a, 12, 31)).strftime("%Y%m%d")
        print(f"\n── {datetime.date(a, mes, 1).strftime('%B/%Y')} "
              f"({d_ini} → {d_fim}) ──")

        reg_mes = 0
        falhas_consecutivas = 0
        for pag in range(1, max_paginas + 1):
            params = {
                "dataInicial":   d_ini,
                "dataFinal":     d_fim,
                "pagina":        pag,
                "tamanhoPagina": tamanho,
            }
            resp = _get_com_retry(f"{BASE_URL}/v1/contratos", params)
            total_reqs += 1

            # Tratamento granular de cada cenário:
            if resp is None:
                # Esgotou as retentativas — pular ESTA página, NÃO o mês inteiro.
                # (Antes este caso causava `break` e abandonava todo o mês,
                # perdendo até centenas de páginas seguintes que podiam estar OK.)
                falhas_consecutivas += 1
                print(f"   ✗ Pág. {pag:3d}: falha — pulando para próxima")
                if falhas_consecutivas >= 5:
                    # 5 páginas seguidas falhando → servidor provavelmente
                    # com problema crônico, aí sim abandonamos o mês
                    print(f"   ⚠ {falhas_consecutivas} falhas consecutivas — "
                          f"abandonando o mês")
                    break
                continue

            if resp.status_code == 204:
                # Sem conteúdo: realmente fim dos dados
                print(f"   ✓ Página {pag} sem conteúdo (204) — fim do mês.")
                break

            try:
                payload = resp.json()
            except ValueError:
                falhas_consecutivas += 1
                print(f"   ✗ Pág. {pag:3d}: resposta não-JSON — pulando")
                continue

            registros = (payload.get("data", []) if isinstance(payload, dict)
                         else payload if isinstance(payload, list) else [])
            if not registros:
                # Página retornou JSON válido mas vazia → fim natural do mês
                print(f"   ✓ Página {pag} vazia — fim do mês.")
                break

            todos.extend(_aplanar_contrato(r) for r in registros)
            reg_mes += len(registros)
            falhas_consecutivas = 0   # sucesso: zera contador de falhas
            print(f"   → Pág. {pag:3d} | {len(registros):4d} registros | "
                  f"acumulado total: {len(todos):,}")
            time.sleep(0.3)  # pausa educada com o servidor

        print(f"   {datetime.date(a, mes, 1).strftime('%b/%Y')}: {reg_mes:,} registros")

        # Checkpoint mensal — salva progresso parcial em disco/Drive.
        # Se o Colab desconectar agora, você não perde o que já baixou.
        if todos:
            sufixo_chk = f"{uf}_{ano}_{ano_fim}_chk"
            df_parcial = pd.DataFrame(todos).drop_duplicates(subset=["numeroControlePNCP"])
            arq_chk = _salvar_checkpoint_coleta(df_parcial, sufixo_chk)
            if arq_chk:
                print(f"   💾 Checkpoint: {arq_chk} ({len(df_parcial):,} registros)")

    print(f"\n✅ Download concluído: {len(todos):,} registros brutos "
          f"em {total_reqs} requisições")

    if not todos:
        print("⚠ Nenhum dado.")
        return pd.DataFrame()

    df = pd.DataFrame(todos)
    antes = len(df)
    df = df.drop_duplicates(subset=["numeroControlePNCP"])
    if len(df) < antes:
        print(f"[info] {antes-len(df):,} duplicatas removidas.")

    # Filtro UF pós-download
    if uf and "ufSigla" in df.columns:
        antes_uf = len(df)
        df = df[df["ufSigla"].str.upper() == uf.upper()].copy()
        print(f"🗺  Filtro UF='{uf}': {antes_uf:,} → {len(df):,} registros "
              f"({len(df)/max(antes_uf,1)*100:.1f}%)")

    print(f"\n✅ {len(df):,} contratos únicos do estado de {uf}.")
    return df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — Limpeza
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# LIMPEZA E NORMALIZAÇÃO DO DATAFRAME
# ════════════════════════════════════════════════════════════════════════════

def carregar_e_limpar(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "categoriaProcessoId" not in df.columns:
        raise ValueError("Coluna 'categoriaProcessoId' ausente.")

    # Filtra somente as categorias relevantes para o problema:
    #   7=Obras, 8=Serviços, 9=Serv. Engenharia (ver MAPA_ROTULO acima)
    df = df[df["categoriaProcessoId"].isin(CATEGORIAS_CONSIDERADAS)].copy()
    if df.empty:
        raise ValueError(
            f"Nenhum registro com categoria em {CATEGORIAS_CONSIDERADAS}."
        )

    df["rotulo"] = df["categoriaProcessoId"].map(MAPA_ROTULO)
    # Categoria original (informativa — preservada como subclasse)
    # Útil para análises posteriores: distinguir "Obra (7)" de "Serv.Eng (9)"
    # dentro da classe "engenharia".
    df["subclasse"] = df["categoriaProcessoId"].map(
        {7: "obra", 9: "serv_engenharia", 8: "serv_geral"}
    )

    print(f"\n   Distribuição por categoria_processo (rótulo final):")
    print(df.groupby(["categoriaProcessoId", "rotulo"]).size()
            .to_frame("registros").to_string())

    # Verificação de UFs presentes (alerta de filtro suspeito)
    if "ufSigla" in df.columns:
        ufs = df["ufSigla"].astype(str).str.upper().value_counts()
        ufs = ufs[ufs.index.str.len() == 2]   # ignora "" e "Outros"
        if len(ufs) > 1:
            print(f"\n   ⚠ ATENÇÃO: dataset contém múltiplas UFs:")
            print(f"      {ufs.to_dict()}")
            print(f"      Se você quer apenas uma UF, use filtrar_uf(df, 'XX')")
            print(f"      ou recarregue com `carregar_checkpoint(uf_filtro='XX')`.")
        elif len(ufs) == 1:
            print(f"   ✓ UF única: {ufs.index[0]} ({ufs.iloc[0]:,} contratos)")

    if "modalidadeId" in df.columns:
        df["modalidadeNome"] = df["modalidadeId"].map(MAPA_MODALIDADE).fillna(
            df.get("modalidadeNome", "Desconhecida"))
    if "criterioJulgamentoId" in df.columns:
        df["criterioJulgamentoNome"] = df["criterioJulgamentoId"].map(MAPA_CRITERIO).fillna(
            df.get("criterioJulgamentoNome", "Desconhecido"))
    df["categoriaProcessoNome"] = df["categoriaProcessoId"].map(MAPA_CATEGORIA)
    if "esferaId" in df.columns:
        df["esferaNome"] = df["esferaId"].map(MAPA_ESFERA).fillna(df["esferaId"])
    if "poderId" in df.columns:
        df["poderNome"] = df["poderId"].map(MAPA_PODER).fillna(df["poderId"])

    for col in ["dataPublicacaoPncp", "dataAberturaProposta", "dataEncerramentoProposta"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "dataPublicacaoPncp" in df.columns:
        df["anoPublicacao"] = df["dataPublicacaoPncp"].dt.year
        df["mesPublicacao"] = df["dataPublicacaoPncp"].dt.month

    for col in ["valorTotalEstimado", "valorTotalHomologado"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan

    for cand in ["objetoCompra", "objetoContrato", "objeto"]:
        if cand in df.columns:
            df["objeto"] = df[cand].astype(str).str.strip()
            df = df[df["objeto"].str.len() > 5].copy()
            break

    # Combina objeto + informação complementar.
    # A informacaoComplementar frequentemente tem detalhamento ESSENCIAL
    # que o objeto omite. Exemplo real:
    #   objeto                  : "Serviço de manutenção predial"
    #   informacaoComplementar  : "Reforma com substituição de cobertura,
    #                              troca de telhado, impermeabilização..."
    # Ao concatenar, capturamos sinais que o classificador precisa para
    # detectar o subenquadramento (presença de "telhado", "cobertura").
    if "informacaoComplementar" in df.columns:
        info = df["informacaoComplementar"].astype(str).fillna("").str.strip()
        df["objeto_completo"] = (
            df["objeto"].astype(str) + " | " + info
        ).str.strip(" |").str.replace(r"\s*\|\s*$", "", regex=True)
        n_com_info = (info.str.len() > 0).sum()
        print(f"   {n_com_info:,} contratos com 'informacaoComplementar' "
              f"({n_com_info/max(len(df),1)*100:.1f}%) — concatenado em 'objeto_completo'")
    else:
        df["objeto_completo"] = df["objeto"]

    if ("dataAberturaProposta" in df.columns and
            "dataEncerramentoProposta" in df.columns):
        df["duracaoPropostaDias"] = (
            df["dataEncerramentoProposta"] - df["dataAberturaProposta"]).dt.days
        df.loc[df["duracaoPropostaDias"] < 0, "duracaoPropostaDias"] = np.nan

    df = df.reset_index(drop=True)
    print(f"[OK] Limpo: {len(df):,} | {df['rotulo'].value_counts().to_dict()}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — Análises descritivas
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# FILTROS PÓS-COLETA
# ════════════════════════════════════════════════════════════════════════════

def filtrar_anos(df: pd.DataFrame, ano_minimo: int) -> pd.DataFrame:
    """
    Filtra o DataFrame removendo contratos publicados antes de `ano_minimo`.
    Útil para descartar anos com viés de classificação detectado pela
    `analisar_vies_temporal`.
    """
    if "anoPublicacao" not in df.columns:
        if "dataPublicacaoPncp" in df.columns:
            df = df.copy()
            df["anoPublicacao"] = pd.to_datetime(
                df["dataPublicacaoPncp"], errors="coerce"
            ).dt.year
        else:
            print("   [aviso] sem coluna de ano — retornando df original.")
            return df

    antes = len(df)
    df_f = df[df["anoPublicacao"].fillna(0).astype(int) >= ano_minimo].copy()
    print(f"   Filtro ano ≥ {ano_minimo}: {antes:,} → {len(df_f):,} "
          f"({len(df_f)/max(antes,1)*100:.1f}% mantidos)")
    return df_f


def _derivar_sufixo_uf_ano(df: pd.DataFrame) -> str:
    """
    Deriva sufixo padrão 'UF_ANO' ou 'UF_ANOMIN_ANOMAX' a partir do df.
    Usado para nomear pastas e arquivos de saída de forma consistente.
    Exemplo:  'SP_2024'  ou  'SP_2022_2024'
    """
    uf = (df["ufSigla"].mode()[0]
          if "ufSigla" in df.columns and not df["ufSigla"].isna().all()
          else "xx")
    if "anoPublicacao" in df.columns:
        anos = df["anoPublicacao"].dropna().astype(int)
        if len(anos) > 0:
            a_min, a_max = int(anos.min()), int(anos.max())
            periodo = f"{a_min}_{a_max}" if a_min != a_max else str(a_min)
            return f"{uf}_{periodo}"
    return f"{uf}_xxxx"


def _pasta_saida_padrao(df: pd.DataFrame) -> str:
    """Pasta padrão para gráficos: 'graficos_pncp_<UF>_<periodo>/'."""
    return f"graficos_pncp_{_derivar_sufixo_uf_ano(df)}"


def filtrar_uf(df: pd.DataFrame, uf: str) -> pd.DataFrame:
    """
    Filtra DataFrame mantendo apenas contratos da UF especificada.
    Útil quando o parquet tem múltiplos estados misturados (caso o
    download original tenha sido feito sem filtro UF).
    """
    if "ufSigla" not in df.columns:
        print("   [aviso] sem coluna ufSigla — retornando df original.")
        return df
    antes = len(df)
    ufs_antes = df["ufSigla"].value_counts().to_dict()
    df_f = df[df["ufSigla"].astype(str).str.upper() == uf.upper()].copy()
    print(f"   Filtro UF='{uf}': {antes:,} → {len(df_f):,}")
    if len(ufs_antes) > 1:
        print(f"   UFs antes: {ufs_antes}")
    return df_f



# ════════════════════════════════════════════════════════════════════════════
# PROMPTS INTERATIVOS (coleta de parâmetros)
# ════════════════════════════════════════════════════════════════════════════

def _pedir_int(msg, pad, mi, ma):
    while True:
        r = input(f"  {msg} [{pad}]: ").strip()
        if r == "": return pad
        try:
            v = int(r)
            if mi <= v <= ma: return v
            print(f"    ⚠ Entre {mi} e {ma}.")
        except ValueError:
            print("    ⚠ Inteiros apenas.")

def _pedir_texto(msg, pad=""):
    r = input(f"  {msg} [{pad or 'Enter = sem filtro'}]: ").strip()
    return r if r else pad

def coletar_parametros_interativo():
    print("\n" + "═"*62 + "\n  CONFIGURAÇÃO — API PNCP (/v1/contratos)\n" + "═"*62)
    print("\n  Ano único (Enter no segundo prompt) ou intervalo (ano_fim > ano_ini)")
    ano_ini = _pedir_int("Ano inicial (2022-2026)", 2024, 2022, 2026)
    ano_fim = _pedir_int(f"Ano final   (≥ {ano_ini}, mesmo ano = um único ano)",
                            ano_ini, ano_ini, 2026)
    mi  = _pedir_int("Mês inicial do primeiro ano (1-12)", 1, 1, 12)
    mf  = _pedir_int("Mês final do último ano (1-12)", 12, 1, 12)

    n_meses = (ano_fim - ano_ini) * 12 + (mf - mi + 1) - (mi - 1) - (12 - mf)
    # cálculo robusto: ajusta para a janela primeiro/último ano
    if ano_fim == ano_ini:
        n_meses_total = mf - mi + 1
    else:
        n_meses_total = (12 - mi + 1) + (ano_fim - ano_ini - 1) * 12 + mf
    print(f"\n  Total estimado: {n_meses_total} meses de coleta")

    print(f"\n  PARÂMETROS DE COLETA:")
    print(f"    • max_paginas: páginas por mês (cada página = até 500 registros)")
    print(f"    • tamanho: registros por página (máx 500)")
    print(f"    Para coleta COMPLETA (sem perder nada): max_paginas=200, tamanho=500")
    mp = _pedir_int("Máx. páginas por mês", 100, 1, 500)
    ta = _pedir_int("Registros por página (1-500)", 500, 1, 500)
    uf = _pedir_texto("UF (obrigatório — ex.: SP, RJ)", "SP").upper()
    print("\n── Parâmetros ──")
    p = dict(ano=ano_ini, ano_fim=ano_fim, mes_inicio=mi, mes_fim=mf,
             max_paginas=mp, tamanho=ta, uf=uf)
    for k, v in p.items(): print(f"   {k:12s}: {v}")
    ok = input("\n  Iniciar? [S/n]: ").strip().lower()
    if ok == "n": raise SystemExit("Cancelado.")
    return p


# ════════════════════════════════════════════════════════════════════════════
# PERSISTÊNCIA (Google Drive + checkpoints)
# ════════════════════════════════════════════════════════════════════════════
#
# Problema 1 (perda de dados ao desconectar): /content é efêmero. Para
# preservar entre sessões, montamos o Drive e salvamos lá.
#
# Problema 2 (Colab desconecta à noite): salvar checkpoints incrementais
# durante a coleta — se desconectar, retoma de onde parou.

DRIVE_MONTADO = False
PASTA_DRIVE   = None  # ex.: "/content/drive/MyDrive/PNCP_TCC"

# Arquivo "marker" que sobrevive a `%run` e nos diz onde o Drive está montado.
_DRIVE_MARKER_FILE = "/tmp/pncp_drive_path.txt"
def _salvar_marker_drive(pasta: str) -> None:
    """Salva o caminho do Drive num arquivo /tmp para sobreviver a %run."""
    try:
        with open(_DRIVE_MARKER_FILE, "w") as f:
            f.write(pasta)
    except Exception:
        pass


def _ler_marker_drive() -> str:
    """Lê o caminho do Drive do arquivo /tmp se existir."""
    try:
        if os.path.exists(_DRIVE_MARKER_FILE):
            with open(_DRIVE_MARKER_FILE) as f:
                pasta = f.read().strip()
            if pasta and os.path.isdir(pasta):
                return pasta
    except Exception:
        pass
    return ""


# Tenta restaurar PASTA_DRIVE do marker logo no carregamento do módulo.
# Isso garante que se o usuário fez montar_drive() em uma sessão antes
# do %run, o Drive continua sendo usado.
_marker_inicial = _ler_marker_drive()
if _marker_inicial:
    PASTA_DRIVE = _marker_inicial
    print(f"   ✓ Drive já montado em sessão anterior: {PASTA_DRIVE}")


def montar_drive(pasta_drive: str = "/content/drive/MyDrive/PNCP_TCC") -> str:
    """
    Monta o Google Drive no Colab e cria pasta de trabalho.

    Use ESTA pasta para salvar parquets, modelos e relatórios — os arquivos
    persistem entre sessões e você não perde dados quando o Colab desconectar.

    No Colab:
        from pncp_analise import montar_drive
        pasta = montar_drive()   # vai pedir autorização no Drive
        # Agora os parquets vão para a pasta retornada

    Em outros ambientes (não Colab): cria a pasta localmente e usa ela.
    """
    global DRIVE_MONTADO, PASTA_DRIVE
    if EM_COLAB:
        try:
            from google.colab import drive
            if not DRIVE_MONTADO:
                drive.mount("/content/drive", force_remount=False)
                DRIVE_MONTADO = True
            os.makedirs(pasta_drive, exist_ok=True)
            PASTA_DRIVE = pasta_drive
            _salvar_marker_drive(pasta_drive)   # PERSISTE o caminho
            print(f"✅ Drive montado. Pasta de trabalho: {pasta_drive}")
            print(f"   (caminho salvo em {_DRIVE_MARKER_FILE} —")
            print(f"    sobrevive a %run pncp_analise.py durante esta sessão)")
            return pasta_drive
        except Exception as e:
            print(f"⚠ Falha ao montar Drive: {e}")
            print(f"   Usando pasta local: {os.getcwd()}")
            PASTA_DRIVE = os.getcwd()
            return PASTA_DRIVE
    else:
        # Fora do Colab — usa pasta local
        os.makedirs(pasta_drive, exist_ok=True)
        PASTA_DRIVE = pasta_drive
        _salvar_marker_drive(pasta_drive)
        print(f"✅ (Fora do Colab) Pasta de trabalho: {pasta_drive}")
        return pasta_drive


def monitorar_ram(rotulo: str = "") -> float:
    """
    Imprime e retorna a RAM em uso pelo processo (em MB).
    Use para identificar onde a memória explode no pipeline:
        monitorar_ram("antes do TF-IDF")
        X = construir_features(df)
        monitorar_ram("depois do TF-IDF")

    Retorna o valor em MB (float). Funciona em qualquer ambiente
    (Linux/Mac/Windows) usando psutil quando disponível, ou /proc/self/status.
    """
    try:
        import psutil
        proc = psutil.Process()
        mb = proc.memory_info().rss / 1024 / 1024
    except ImportError:
        try:
            with open("/proc/self/status") as f:
                for linha in f:
                    if linha.startswith("VmRSS:"):
                        mb = int(linha.split()[1]) / 1024
                        break
                else:
                    mb = -1
        except Exception:
            mb = -1

    if mb >= 0:
        marcador = "⚠️" if mb > 8000 else ("🟡" if mb > 4000 else "🟢")
        print(f"   {marcador} RAM: {mb:>7,.0f} MB" + (f"  ({rotulo})" if rotulo else ""))
    return mb


def liberar_memoria(*objetos) -> None:
    """
    Libera memória de objetos Python explicitamente. Útil entre etapas
    para evitar acúmulo, especialmente matrizes esparsas grandes do TF-IDF.

    Uso:
        # ao final da Parte 2, antes da Parte 3
        liberar_memoria(p2["X_dense_temporario"], lista_grande_temporaria)
        # equivalente a: del + gc.collect()

    Não passe variáveis que você ainda vai usar! Esta função apenas
    força gc.collect() e printa o ganho.
    """
    import gc
    antes = monitorar_ram("antes do gc")
    # Os objetos passados são apenas referências locais a esta função;
    # o gc.collect() vai limpá-los se eles não tiverem outras referências
    # no escopo do chamador. A LIBERAÇÃO REAL acontece quando você faz
    # `del nome_variavel` no escopo onde ela vive.
    del objetos
    gc.collect()
    depois = monitorar_ram("depois do gc")
    if antes > 0 and depois > 0:
        ganho = antes - depois
        print(f"   ✅ liberados {ganho:,.0f} MB" if ganho > 50 else
              f"   (gc.collect liberou pouco — use `del nome_var` antes)")


def diagnostico_drive() -> dict:
    """
    Diagnostica em detalhes o estado do Drive e onde os arquivos estão sendo
    salvos. Útil quando você suspeita que checkpoints não estão indo para o
    Drive como esperado.

    Use:
        from pncp_analise import diagnostico_drive
        diagnostico_drive()
    """
    import glob
    print("\n" + "═"*62)
    print("  DIAGNÓSTICO DA PERSISTÊNCIA")
    print("═"*62)

    # 1. Variáveis globais
    print(f"\n[1] Variáveis globais:")
    print(f"   DRIVE_MONTADO  = {DRIVE_MONTADO}")
    print(f"   PASTA_DRIVE    = {PASTA_DRIVE}")

    # 2. Marker file
    print(f"\n[2] Arquivo marker ({_DRIVE_MARKER_FILE}):")
    if os.path.exists(_DRIVE_MARKER_FILE):
        with open(_DRIVE_MARKER_FILE) as f:
            print(f"   Conteúdo: {f.read().strip()}")
    else:
        print(f"   ⚠ Arquivo NÃO existe (montar_drive nunca rodou nesta sessão)")

    # 3. Existência de /content/drive
    print(f"\n[3] Drive montado no Colab?")
    drive_dir = "/content/drive/MyDrive"
    if os.path.isdir(drive_dir):
        print(f"   ✓ {drive_dir} existe (Drive ESTÁ montado)")
        pncp_dir = "/content/drive/MyDrive/PNCP_TCC"
        if os.path.isdir(pncp_dir):
            print(f"   ✓ {pncp_dir} existe")
        else:
            print(f"   ⚠ {pncp_dir} NÃO existe (rode montar_drive())")
    else:
        print(f"   ✗ {drive_dir} NÃO existe (Drive NÃO está montado)")
        print(f"      → Faça: from pncp_analise import montar_drive; montar_drive()")

    # 4. Onde um checkpoint seria salvo agora
    print(f"\n[4] Teste: para onde um checkpoint iria agora?")
    teste = _path_persistente("teste_diagnostico.parquet")
    print(f"   {teste}")
    if teste.startswith("/content/drive/"):
        print(f"   ✓ Iria para o Drive (PERSISTE)")
    else:
        print(f"   ⚠ Iria para diretório local (NÃO persiste se desconectar)")

    # 5. Arquivos já salvos no Drive
    print(f"\n[5] Parquets já salvos:")
    locais_busca = []
    if PASTA_DRIVE and os.path.isdir(PASTA_DRIVE):
        locais_busca.append(PASTA_DRIVE)
    if os.path.isdir("/content/drive/MyDrive/PNCP_TCC"):
        locais_busca.append("/content/drive/MyDrive/PNCP_TCC")
    locais_busca.append(os.getcwd())
    for loc in set(locais_busca):
        parquets = sorted(glob.glob(os.path.join(loc, "contratacoes_*.parquet")))
        if parquets:
            print(f"   📂 {loc}:")
            for p in parquets:
                tam_mb = os.path.getsize(p) / 1024 / 1024
                print(f"      • {os.path.basename(p)} ({tam_mb:.1f} MB)")
        else:
            print(f"   📂 {loc}: (vazio)")

    print("\n" + "═"*62)
    return {
        "drive_montado":  DRIVE_MONTADO,
        "pasta_drive":    PASTA_DRIVE,
        "marker_existe":  os.path.exists(_DRIVE_MARKER_FILE),
        "drive_disponivel": os.path.isdir("/content/drive/MyDrive"),
        "destino_atual":  teste,
    }


def keep_alive_javascript():
    """
    Injeta JavaScript no navegador para manter a sessão Colab viva.

    Estratégia tripla (resiliente a mudanças do Colab):
      1. Clica no botão Connect a cada 60s (estrutura nova)
      2. Simula movimento de mouse (fallback)
      3. Dispara evento de scroll (mantém aba "ativa")

    Limitações honestas:
    • NÃO ultrapassa o limite duro de 12h (Colab grátis) ou 24h (Pro)
    • NÃO funciona se o Chrome/Firefox descartar a aba inativa
      (Chrome agressivamente economiza recursos em abas em background;
      mantenha a aba do Colab visível ou use 'Configurações > Sistema >
      Continuar executando aplicativos em segundo plano')
    • NÃO resolve CAPTCHAs aleatórios do Colab — só intervenção humana

    Para sessões longas (5+ horas), recomendo FORTEMENTE:
    • Manter a aba do Colab VISÍVEL na tela durante a noite (não minimizada)
    • Desativar economia de bateria do Windows
    • Usar Colab Pro se possível
    • Quebrar a coleta em pedaços de ≤4 horas, retomando do checkpoint
    """
    if not EM_COLAB:
        print("⚠ keep_alive_javascript funciona apenas no Colab.")
        return
    try:
        from IPython.display import Javascript, display
        js = """
        // Limpa intervalos anteriores se existirem
        if (window._pncpKeepAlive) clearInterval(window._pncpKeepAlive);
        if (window._pncpScrollAlive) clearInterval(window._pncpScrollAlive);

        // Estratégia 1: Clica no botão de conexão (estratégia principal)
        window._pncpKeepAlive = setInterval(function() {
            try {
                console.log('PNCP keep-alive [' + new Date().toISOString() + ']');
                // Tenta múltiplos seletores conforme Colab muda
                var btn = document.querySelector('colab-connect-button');
                if (btn && btn.shadowRoot) {
                    var inner = btn.shadowRoot.querySelector('#connect');
                    if (inner) { inner.click(); console.log('  → clicked connect'); }
                }
                // Fallback: barra de toolbar
                var btn2 = document.querySelector('colab-toolbar-button#connect');
                if (btn2) btn2.click();
                // Simula mouse move para registrar atividade
                document.dispatchEvent(new MouseEvent('mousemove', {
                    bubbles: true, clientX: Math.random()*200, clientY: Math.random()*200
                }));
            } catch(e) { console.warn('keep-alive falhou:', e); }
        }, 60000);

        // Estratégia 2: Scroll mínimo para manter a aba marcada como ativa
        window._pncpScrollAlive = setInterval(function() {
            try {
                window.scrollBy(0, 1);
                window.scrollBy(0, -1);
            } catch(e) {}
        }, 30000);

        console.log('PNCP keep-alive ATIVO (clique a cada 60s + scroll a cada 30s)');
        """
        display(Javascript(js))
        print("✅ Keep-alive ATIVO no navegador.")
        print("   • Clica no botão Connect a cada 60s")
        print("   • Scroll mínimo a cada 30s (mantém aba ativa)")
        print()
        print("⚠ AVISOS IMPORTANTES:")
        print("   1. NÃO funciona se a aba ficar minimizada/descartada.")
        print("   2. Chrome 'descongela' abas em background — desabilite isso:")
        print("      chrome://discards/  →  Disable")
        print("   3. NÃO ultrapassa o limite de 12h do Colab grátis.")
        print("   4. Para coletas >4h, use checkpoints (já implementado).")
    except Exception as e:
        print(f"⚠ Falha ao injetar keep-alive: {e}")


def _path_persistente(nome_arquivo: str) -> str:
    """
    Retorna o caminho onde um arquivo deve ser salvo:
    • Se Drive montado, salva no Drive (persiste entre sessões)
    • Caso contrário, salva no diretório atual (efêmero no Colab)

    Hierarquia de detecção do Drive (em ordem):
      1. Variável global PASTA_DRIVE (setada por montar_drive nesta sessão)
      2. Marker em /tmp/pncp_drive_path.txt (montar_drive de sessão anterior)
      3. Auto-detecção em /content/drive/MyDrive/PNCP_TCC
    """
    global PASTA_DRIVE
    # Estratégia 1: variável global atual
    if PASTA_DRIVE and os.path.isdir(PASTA_DRIVE):
        return os.path.join(PASTA_DRIVE, nome_arquivo)
    # Estratégia 2: arquivo marker (sobrevive %run)
    marker = _ler_marker_drive()
    if marker:
        PASTA_DRIVE = marker
        return os.path.join(PASTA_DRIVE, nome_arquivo)
    # Estratégia 3: auto-detect padrão
    if os.path.isdir("/content/drive/MyDrive"):
        PASTA_DRIVE = "/content/drive/MyDrive/PNCP_TCC"
        os.makedirs(PASTA_DRIVE, exist_ok=True)
        _salvar_marker_drive(PASTA_DRIVE)
        return os.path.join(PASTA_DRIVE, nome_arquivo)
    # Fallback: dir atual
    return nome_arquivo


def _salvar_checkpoint_coleta(df_parcial: pd.DataFrame, sufixo: str) -> str:
    """
    Salva checkpoint da coleta (chamado a cada N páginas durante o download).
    Se a coleta falhar, basta carregar este parquet e continuar de onde parou.
    """
    if df_parcial.empty:
        return ""
    nome = f"contratacoes_checkpoint_{sufixo}.parquet"
    caminho = _path_persistente(nome)
    try:
        df_parcial.to_parquet(caminho, index=False)
        # Confirmação visual: o caminho começa com /content/drive ou não?
        eh_drive = caminho.startswith("/content/drive/")
        local_str = "Drive ✓" if eh_drive else "DIR LOCAL ⚠ (não persiste!)"
        return f"{caminho} [{local_str}]"
    except Exception as e:
        print(f"   [aviso] checkpoint falhou: {e}")
        return ""



# ════════════════════════════════════════════════════════════════════════════
# RECARGA DE DADOS (checkpoint + combinação)
# ════════════════════════════════════════════════════════════════════════════

def carregar_checkpoint(caminho: str = None,
                          uf_filtro: str = None,
                          aplicar_limpeza: bool = True) -> pd.DataFrame:
    """
    Carrega um parquet (checkpoint OU contratacoes_limpas) e retorna o
    df pronto para usar nas próximas etapas.

    Funciona com 3 modos:
      1. caminho fornecido → carrega esse arquivo específico
      2. caminho=None → procura automaticamente:
         a) PASTA_DRIVE/contratacoes_limpas_*.parquet (mais recente)
         b) PASTA_DRIVE/contratacoes_checkpoint_*.parquet (mais recente)
         c) ./contratacoes_limpas_*.parquet (no dir atual)

    Parâmetros
    ──────────
    caminho         : path do parquet (opcional)
    uf_filtro       : UF a filtrar (caso o arquivo tenha vários estados)
                       Se None, não filtra — assume que o parquet já está OK.
    aplicar_limpeza : se True e o arquivo for "brutas" ou "checkpoint",
                       aplica `carregar_e_limpar` para gerar o df limpo.

    Exemplo
    ───────
        # Carrega o checkpoint mais recente automaticamente
        df = carregar_checkpoint()

        # Carrega path específico, filtra apenas SP
        df = carregar_checkpoint(
            "/content/drive/MyDrive/PNCP_TCC/contratacoes_checkpoint_SP_2022_2025_chk.parquet",
            uf_filtro="SP"
        )
    """
    import glob

    # Localiza o arquivo
    if caminho is None:
        # Procura no Drive primeiro (mais recente)
        candidatos = []
        if PASTA_DRIVE and os.path.isdir(PASTA_DRIVE):
            candidatos += sorted(glob.glob(
                os.path.join(PASTA_DRIVE, "contratacoes_limpas_*.parquet")
            ), key=os.path.getmtime, reverse=True)
            candidatos += sorted(glob.glob(
                os.path.join(PASTA_DRIVE, "contratacoes_checkpoint_*.parquet")
            ), key=os.path.getmtime, reverse=True)
        # Fallback: dir atual
        candidatos += sorted(glob.glob("contratacoes_limpas_*.parquet"),
                              key=os.path.getmtime, reverse=True)
        candidatos += sorted(glob.glob("contratacoes_checkpoint_*.parquet"),
                              key=os.path.getmtime, reverse=True)
        # E em /content/drive (caso PASTA_DRIVE não esteja setado)
        for p in [
            "/content/drive/MyDrive/PNCP_TCC",
            "/content/drive/MyDrive",
        ]:
            if os.path.isdir(p):
                candidatos += sorted(glob.glob(
                    os.path.join(p, "contratacoes_*.parquet")
                ), key=os.path.getmtime, reverse=True)

        if not candidatos:
            raise FileNotFoundError(
                "Nenhum parquet 'contratacoes_*.parquet' encontrado.\n"
                "Rode primeiro `df = executar_apenas_coleta(modo_interativo=True)`."
            )
        caminho = candidatos[0]
        print(f"   📂 Auto-detectado: {caminho}")

    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    print(f"   📥 Carregando: {caminho}")
    df = pd.read_parquet(caminho)
    print(f"   ✓ {len(df):,} linhas × {len(df.columns)} colunas")

    # Filtro UF (se solicitado)
    if uf_filtro and "ufSigla" in df.columns:
        antes = len(df)
        ufs_presentes = df["ufSigla"].value_counts()
        if len(ufs_presentes) > 1:
            print(f"   ⚠ Múltiplas UFs no arquivo: {ufs_presentes.to_dict()}")
        df = df[df["ufSigla"].astype(str).str.upper() == uf_filtro.upper()].copy()
        print(f"   🗺  Filtro UF='{uf_filtro}': {antes:,} → {len(df):,}")

    # Decide se aplica limpeza
    eh_bruto = ("brutas" in caminho.lower() or "checkpoint" in caminho.lower())
    ja_limpo = "rotulo" in df.columns
    if aplicar_limpeza and eh_bruto and not ja_limpo:
        print(f"   🧹 Aplicando carregar_e_limpar (parquet bruto detectado)...")
        df = carregar_e_limpar(df)
    elif ja_limpo:
        print(f"   ✓ Parquet já limpo (coluna 'rotulo' presente).")

    return df


def combinar_parquets(caminhos: list = None,
                        uf_filtro: str = None,
                        salvar_em: str = None,
                        aplicar_limpeza: bool = True,
                        ignorar_outras_ufs: bool = True) -> pd.DataFrame:
    """
    Combina múltiplos parquets de coletas separadas em um único DataFrame.

    Caso de uso típico: você fez 3 coletas separadas (uma por ano) para
    evitar desconexão no Colab. Esta função junta tudo em um parquet só.

    Parâmetros
    ──────────
    caminhos : lista de paths (opcional). Se None, busca automaticamente
                em PASTA_DRIVE, /content/drive/MyDrive/PNCP_TCC e dir atual:
                contratacoes_brutas_*.parquet, contratacoes_checkpoint_*.parquet,
                contratacoes_limpas_*.parquet.
    uf_filtro : se especificado, mantém somente contratos dessa UF.
    salvar_em : caminho do parquet consolidado a gerar. Se None, gera
                 nome automático: contratacoes_limpas_<UF>_<min>-<max>_consolidado.parquet
                 e salva no Drive (se montado) ou dir atual.
    aplicar_limpeza : se True, aplica `carregar_e_limpar` no resultado final
                       (só roda se o df não tiver coluna 'rotulo' ainda).
    ignorar_outras_ufs : avisa quando arquivos têm UFs diferentes mas
                          tenta consolidar mesmo assim. Se False, aborta.

    Retorna
    ───────
    DataFrame consolidado, deduplicado por numeroControlePNCP.

    Exemplo
    ───────
        # Auto: junta tudo que estiver no Drive
        df = combinar_parquets(uf_filtro="SP")

        # Manual: caminhos específicos
        df = combinar_parquets(
            caminhos=[
                "/content/drive/MyDrive/PNCP_TCC/contratacoes_checkpoint_SP_2022_2022_chk.parquet",
                "/content/drive/MyDrive/PNCP_TCC/contratacoes_checkpoint_SP_2023_2023_chk.parquet",
                "/content/drive/MyDrive/PNCP_TCC/contratacoes_checkpoint_SP_2024_2024_chk.parquet",
            ],
            uf_filtro="SP",
            salvar_em="/content/drive/MyDrive/PNCP_TCC/contratacoes_limpas_SP_2022_2024.parquet",
        )
    """
    import glob

    print("\n" + "█"*62)
    print("  COMBINAR PARQUETS DE COLETAS SEPARADAS")
    print("█"*62)

    # ── 1. Resolve lista de caminhos ────────────────────────────────────────
    if caminhos is None:
        candidatos = []
        # Busca em PASTA_DRIVE (se setada) + caminhos padrão
        locais = []
        if PASTA_DRIVE and os.path.isdir(PASTA_DRIVE):
            locais.append(PASTA_DRIVE)
        for p in ["/content/drive/MyDrive/PNCP_TCC",
                   "/content/drive/MyDrive", os.getcwd()]:
            if os.path.isdir(p) and p not in locais:
                locais.append(p)

        for loc in locais:
            for padrao in ["contratacoes_brutas_*.parquet",
                              "contratacoes_checkpoint_*.parquet",
                              "contratacoes_limpas_*.parquet"]:
                candidatos += glob.glob(os.path.join(loc, padrao))

        # Dedup por nome de arquivo (mesmo arquivo em locais diferentes)
        vistos = set()
        caminhos = []
        for c in sorted(candidatos):
            base = os.path.basename(c)
            if base not in vistos:
                vistos.add(base)
                caminhos.append(c)

        if not caminhos:
            raise FileNotFoundError(
                "Nenhum parquet encontrado. Especifique `caminhos=[...]` "
                "ou rode `executar_apenas_coleta()` primeiro."
            )
        print(f"\n   Auto-detectados {len(caminhos)} parquet(s):")
        for c in caminhos:
            print(f"   • {c}")

    # ── 2. Carrega cada um e empilha ────────────────────────────────────────
    print(f"\n   Carregando e empilhando...")
    pedacos = []
    sumario = []
    for c in caminhos:
        if not os.path.exists(c):
            print(f"   ⚠ Pulando (não existe): {c}")
            continue
        try:
            d = pd.read_parquet(c)
        except Exception as e:
            print(f"   ⚠ Falha em {c}: {e}")
            continue
        n = len(d)
        ufs_no_arq = (d["ufSigla"].dropna().astype(str).str.upper().value_counts()
                       if "ufSigla" in d.columns else pd.Series(dtype=int))
        anos_arq = (sorted(d["dataPublicacaoPncp"].dropna().astype(str).str[:4].unique())
                     if "dataPublicacaoPncp" in d.columns else [])
        sumario.append({
            "arquivo":       os.path.basename(c),
            "registros":     n,
            "colunas":       d.shape[1],
            "ja_limpo":      "rotulo" in d.columns,
            "ufs":           dict(ufs_no_arq.head(5)),
            "anos":          anos_arq[:6],
        })
        pedacos.append(d)

    if not pedacos:
        raise RuntimeError("Nenhum parquet pôde ser carregado.")

    # Sumário ANTES da concatenação
    print(f"\n   ── Sumário dos arquivos carregados ──")
    df_sumario = pd.DataFrame(sumario)
    print(df_sumario.to_string(index=False))

    # ── 3. Verificação de UFs (alerta) ──────────────────────────────────────
    todas_ufs = set()
    for s in sumario:
        todas_ufs.update(s["ufs"].keys())
    todas_ufs = {u for u in todas_ufs if isinstance(u, str) and len(u) == 2}
    if len(todas_ufs) > 1:
        print(f"\n   ⚠ Os arquivos contêm múltiplas UFs: {sorted(todas_ufs)}")
        if uf_filtro:
            print(f"   → Filtro uf_filtro='{uf_filtro}' será aplicado.")
        elif ignorar_outras_ufs:
            print(f"   → Vai consolidar TODAS (use uf_filtro='SP' p/ filtrar)")
        else:
            raise ValueError(
                f"Múltiplas UFs detectadas {sorted(todas_ufs)} e "
                f"ignorar_outras_ufs=False. Especifique uf_filtro."
            )

    # ── 4. Concatena ────────────────────────────────────────────────────────
    df = pd.concat(pedacos, ignore_index=True)
    n_concat = len(df)
    print(f"\n   📊 Concatenado: {n_concat:,} linhas")

    # ── 5. Filtro UF ────────────────────────────────────────────────────────
    if uf_filtro and "ufSigla" in df.columns:
        antes = len(df)
        df = df[df["ufSigla"].astype(str).str.upper() == uf_filtro.upper()].copy()
        print(f"   🗺  Filtro UF='{uf_filtro}': {antes:,} → {len(df):,}")

    # ── 6. Deduplicação ─────────────────────────────────────────────────────
    if "numeroControlePNCP" in df.columns:
        antes = len(df)
        df = df.drop_duplicates(subset=["numeroControlePNCP"], keep="first")
        n_dups = antes - len(df)
        if n_dups > 0:
            print(f"   🧹 Deduplicação: {antes:,} → {len(df):,} "
                  f"(removidas {n_dups:,} duplicatas)")
        else:
            print(f"   ✓ Sem duplicatas (todas linhas têm numeroControlePNCP único)")

    # ── 7. Aplica limpeza ───────────────────────────────────────────────────
    ja_limpo = "rotulo" in df.columns
    if aplicar_limpeza and not ja_limpo:
        print(f"\n   🧹 Aplicando carregar_e_limpar (parquet sem coluna 'rotulo')...")
        df = carregar_e_limpar(df)
    elif ja_limpo:
        # Se ALGUNS arquivos eram limpos e outros brutos, pode haver linhas
        # sem 'rotulo'. Refiltra se houver NaN.
        n_sem_rotulo = df["rotulo"].isna().sum()
        if n_sem_rotulo > 0:
            print(f"   ⚠ {n_sem_rotulo} linhas sem rótulo. Reaplicando limpeza...")
            df = carregar_e_limpar(df)

    # ── 8. Distribuição por ano ─────────────────────────────────────────────
    if "anoPublicacao" in df.columns:
        print(f"\n   📅 Distribuição por ano:")
        for ano, cnt in df["anoPublicacao"].value_counts().sort_index().items():
            if pd.notna(ano):
                print(f"      {int(ano)}: {cnt:,}")

    # ── 9. Salva resultado ──────────────────────────────────────────────────
    if salvar_em is None:
        # Gera nome automático
        if "ufSigla" in df.columns and not df["ufSigla"].isna().all():
            uf = df["ufSigla"].mode()[0]
        else:
            uf = uf_filtro or "XX"
        if "anoPublicacao" in df.columns:
            anos = df["anoPublicacao"].dropna().astype(int)
            if len(anos) > 0:
                ano_min, ano_max = int(anos.min()), int(anos.max())
                periodo = f"{ano_min}_{ano_max}" if ano_min != ano_max else str(ano_min)
            else:
                periodo = "periodo"
        else:
            periodo = "periodo"
        nome = f"contratacoes_limpas_{uf}_{periodo}_consolidado.parquet"
        salvar_em = _path_persistente(nome)

    try:
        df.to_parquet(salvar_em, index=False)
        eh_drive = "/content/drive/" in salvar_em
        local_str = "Drive ✓" if eh_drive else "DIR LOCAL ⚠"
        print(f"\n   💾 Consolidado salvo: {salvar_em}  [{local_str}]")
        print(f"      Total: {len(df):,} contratos")
    except Exception as e:
        print(f"   ⚠ Falha ao salvar: {e}")

    print("\n" + "█"*62)
    print(f"  CONSOLIDAÇÃO ✅  {len(df):,} registros únicos")
    print("█"*62)
    return df


def executar_apenas_coleta(modo_interativo=True, forcar_redownload=False):
    """
    ETAPA A1 — apenas baixa da API e salva o parquet limpo.
    NÃO roda a EDA. Útil quando você quer baixar uma vez e iterar
    sobre análises sem precisar reconectar com a API.

    Uso típico no Colab:
        df = executar_apenas_coleta(modo_interativo=True)
        # depois, em outra célula:
        executar_apenas_eda(df)

    Parâmetros
    ──────────
    modo_interativo   : se True, pergunta os parâmetros (ano, UF, etc.)
    forcar_redownload : se True, ignora parquets existentes e baixa de novo.
                         Default False — se um parquet com o mesmo
                         <UF>_<período> já existe, carrega ele e pula a coleta.

    Retorna
    ───────
    df_limpo (DataFrame) ou None se falhou.
    """
    if modo_interativo:
        params = coletar_parametros_interativo()
    else:
        params = dict(ano=ANO_INICIO, ano_fim=ANO_FIM,
                       mes_inicio=MES_INICIO, mes_fim=MES_FIM,
                       max_paginas=MAX_PAGINAS, tamanho=TAMANHO, uf=UF)

    ano_ini = params["ano"]
    ano_fim = params.get("ano_fim", ano_ini)
    sufixo = (f"{params['uf']}_{ano_ini}" if ano_ini == ano_fim
              else f"{params['uf']}_{ano_ini}_{ano_fim}")

    # Verifica se já há um parquet limpo desse exato período no Drive.
    # Útil quando o usuário re-roda a célula sem querer (Colab desconectou e
    # ele não lembra se já baixou, ou está iterando análises).
    arq_limpo_existente = _path_persistente(f"contratacoes_limpas_{sufixo}.parquet")
    if os.path.exists(arq_limpo_existente) and not forcar_redownload:
        print(f"\n⚠ Já existe coleta deste período: {arq_limpo_existente}")
        try:
            df_pronto = pd.read_parquet(arq_limpo_existente)
            print(f"   {len(df_pronto):,} contratos já coletados.")
            print(f"   Para baixar de novo, passe `forcar_redownload=True`.")
            return df_pronto
        except Exception as e:
            print(f"   [aviso] arquivo existe mas falhou ao carregar: {e}")
            print(f"   Vai re-baixar.")

    print("\n" + "═"*62 + "\n  ETAPA A1 — COLETA DA API (sem EDA)\n" + "═"*62)
    df_raw = baixar_contratacoes_pncp_por_uf(**params)
    if df_raw.empty:
        print("❌ Sem dados.")
        return None

    arq = _path_persistente(f"contratacoes_brutas_{sufixo}.parquet")
    df_raw.to_parquet(arq, index=False)
    print(f"💾 Bruto: {arq}  {df_raw.shape}")
    if "categoriaProcessoId" in df_raw.columns:
        print("\n── Categorias encontradas ──")
        print(df_raw["categoriaProcessoId"].value_counts().to_string())

    print("\n── Limpeza ──")
    try:
        df = carregar_e_limpar(df_raw)
    except ValueError as e:
        print(f"❌ {e}")
        return None

    arq2 = _path_persistente(f"contratacoes_limpas_{sufixo}.parquet")
    df.to_parquet(arq2, index=False)
    print(f"\n💾 Limpo: {arq2}")
    print(f"\n✅ Etapa A1 concluída. Para rodar EDA:")
    print(f"   executar_apenas_eda(df)")
    return df




# ════════════════════════════════════════════════════════════════════════════
# RESTAURAÇÃO AUTOMÁTICA DO DRIVE (ao carregar o módulo)
# ════════════════════════════════════════════════════════════════════════════
# Se o usuário fez montar_drive() em uma sessão antes de %run, recupera o
# caminho do arquivo marker — assim os checkpoints continuam indo para o
# Drive sem precisar chamar montar_drive() de novo.
_marker_inicial = _ler_marker_drive()
if _marker_inicial:
    PASTA_DRIVE = _marker_inicial
    print(f"   ✓ Drive já montado em sessão anterior: {PASTA_DRIVE}")


print("✅ pncp_coleta.py carregado.")
