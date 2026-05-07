"""
Triagem determinística — antes do ML.

Sua reflexão metodológica em 3 etapas:

  1. Pré-filtro lexical (esta etapa)
     Termos que SEM AMBIGUIDADE indicam obra/engenharia ("construção de
     ponte", "pavimentação asfáltica", "reforma estrutural"). Contratos
     rotulados 'geral' que casam com esses padrões são `obvio_engenharia`.

  2. Verificação de rito (esta etapa)
     Para cada `obvio_engenharia`, checa se o processo seguiu o rito de
     engenharia da Lei 14.133/2021 (ART/RRT, memorial descritivo, projeto
     básico/executivo, engenheiro responsável, norma ABNT NBR).
     - Se ≥2 sinais de rito → `rotulacao_incorreta_processo_ok`
       (rótulo errado, mas processo correto — não viola a lei)
     - Se < 2 sinais → `subenquadramento_real`
       (violação Lei 14.133: fugiu do rito de engenharia)

  3. ML para os ambíguos
     Contratos não-óbvios entram no pipeline TF-IDF/BERTimbau normal.

Esta separação prévia aumenta precisão do ML (treina em casos ambíguos),
reduz custo (não roda PDFs em todos), e produz um relatório jurídico mais
útil (separa "rotulação errada" de "violação da lei").
"""

import re
from pathlib import Path

import pandas as pd

from pncp import config
from pncp.io_disco import (
    ler_parquet, salvar_parquet, salvar_json,
)
from pncp.ram import liberar, com_gc


# ── Padrões "óbvios" de engenharia ───────────────────────────────────────────
# Construídos com cuidado para alta PRECISÃO (poucos falsos positivos).
# Cada padrão é uma frase/expressão que, se aparece no objeto, é
# praticamente certo que o contrato é de obra/engenharia.
PADROES_OBVIOS = [
    # Construção e reforma
    r"\bconstruc[aã]o\s+(de|da|do|dos|das)\b",
    r"\breforma\s+(de|da|do|dos|das|estrutural|predial|civil)\b",
    r"\bampliac[aã]o\s+(de|da|do|dos|das)\b.*(predio|escola|hospital|ponte|ed[ií]ficio)",
    # Infraestrutura viária
    r"\bpavimentac[aã]o\s+(asf[aá]ltica|com\s+asfalto|de\s+via|de\s+rua)\b",
    r"\brecapeamento\s+asf[aá]ltico\b",
    r"\bdrenagem\s+(pluvial|urbana|de\s+via)\b",
    r"\bterraplenagem\b",
    # Obras de arte
    r"\bconstruc[aã]o\s+de\s+(ponte|viaduto|passarela|bueiro)\b",
    r"\bobra\s+civil\b",
    r"\bobra\s+de\s+arte\s+especial\b",
    # Edificações
    r"\bedificac[aã]o\s+(nova|escolar|hospitalar|p[uú]blica)\b",
    r"\bconstruc[aã]o\s+de\s+(escola|creche|posto|ubs|upa|hospital)\b",
    # Saneamento e infraestrutura
    r"\brede\s+de\s+(esgoto|[aá]gua|distribuic[aã]o)\b",
    r"\bestac[aã]o\s+de\s+(tratamento|elevac[aã]o|bombeamento)\b",
    r"\bsubestac[aã]o\s+el[eé]trica\b",
    # Materiais de obra
    r"\bestrutura\s+(met[aá]lica|de\s+concreto)\b",
    r"\bconcreto\s+armado\b",
    r"\balvenaria\s+estrutural\b",
    # Documentos técnicos no objeto
    r"\bprojeto\s+(b[aá]sico|executivo|arquitet[oô]nico|estrutural)\b",
    r"\bmemorial\s+descritivo\b",
    # Profissionais (se citados no objeto, é eng)
    r"\bengenharia\s+civil\b",
    r"\barquitet[oô]nico\s+e\s+complementares\b",
]
PADROES_OBVIOS_RX = [re.compile(p, flags=re.IGNORECASE) for p in PADROES_OBVIOS]


def _texto_para_matching(texto):
    """Normaliza para matching dos padrões (mantém acentos, lowercase)."""
    if not isinstance(texto, str):
        return ""
    return texto.lower()


def eh_obvio_engenharia(texto):
    """Retorna True se o texto casa com algum padrão óbvio + qual padrão."""
    t = _texto_para_matching(texto)
    if not t:
        return False, None
    for rx in PADROES_OBVIOS_RX:
        if rx.search(t):
            return True, rx.pattern
    return False, None


# ── Sinais de rito de engenharia (Lei 14.133/2021) ───────────────────────────
# Estes sinais aparecem nos PDFs (Camada 2) ou em campos específicos do
# contrato. A presença de 2+ sinais indica que o processo seguiu o rito
# formal de engenharia, mesmo se o rótulo veio errado.
SINAIS_RITO = (
    "art",                         # Anotação de Responsabilidade Técnica
    "rrt",                         # Registro de Responsabilidade Técnica
    "memorial",                    # Memorial descritivo
    "projeto_executivo",           # Projeto executivo
    "as_built",                    # As-built
    "crea",                        # CREA citado
    "engenheiro",                  # Profissional eng. citado
    "norma_tecnica",               # Norma técnica
    "abnt_nbr",                    # ABNT NBR
    "anotacao_responsabilidade",   # ART por extenso
)
LIMIAR_RITO = 2  # ≥2 sinais = rito seguido


def contar_sinais_rito(linha_features_pdf):
    """
    Recebe uma linha do parquet de features de PDF (Camada 2) e
    conta quantos sinais de rito apareceram.
    """
    if linha_features_pdf is None:
        return 0
    return sum(int(linha_features_pdf.get(s, 0) or 0) > 0 for s in SINAIS_RITO)


# ── Pipeline ─────────────────────────────────────────────────────────────────
@com_gc
def executar(caminho_parquet=None):
    """
    Marca cada contrato com `eh_obvio_engenharia` e (se PDFs já processados)
    `seguiu_rito` + `classificacao_triagem`.

    Saída: dados/triagem/triagem.parquet com colunas extras.
    Também salva resumo.json com contagens.
    """
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")
    df = ler_parquet(caminho_parquet)

    # 1. Pré-filtro lexical sobre o objeto
    print("[triagem] aplicando pré-filtro lexical...")
    flags = df["objeto"].fillna("").map(eh_obvio_engenharia)
    df["eh_obvio_engenharia"] = flags.map(lambda t: t[0]).astype(bool)
    df["padrao_obvio"] = flags.map(lambda t: t[1])

    # 2. Verificação de rito (se Camada 2 já rodou)
    pdfs_path = config.caminho(config.SUB_C2, "features_pdfs.parquet")
    if Path(pdfs_path).exists():
        print("[triagem] cruzando com features de PDF (Camada 2)...")
        feats = ler_parquet(pdfs_path)
        df = df.merge(
            feats[["numeroControlePNCP"] + list(SINAIS_RITO)],
            on="numeroControlePNCP", how="left",
        )
        df["n_sinais_rito"] = (
            df[list(SINAIS_RITO)].fillna(0).gt(0).sum(axis=1).astype("int8")
        )
        df["seguiu_rito"] = df["n_sinais_rito"] >= LIMIAR_RITO
    else:
        print("[triagem] Camada 2 ainda não rodou — sem verificação de rito")
        df["n_sinais_rito"] = 0
        df["seguiu_rito"] = False

    # 3. Classificação determinística
    df["classificacao_triagem"] = "ambiguo"
    df.loc[df["rotulo"] != "geral", "classificacao_triagem"] = "fora_escopo"

    obvio_geral = (df["rotulo"] == "geral") & df["eh_obvio_engenharia"]
    df.loc[obvio_geral & df["seguiu_rito"],
           "classificacao_triagem"] = "rotulacao_incorreta_processo_ok"
    df.loc[obvio_geral & ~df["seguiu_rito"],
           "classificacao_triagem"] = "subenquadramento_real"

    # 4. Persiste e resumo
    saida = config.caminho("triagem", "triagem.parquet")
    salvar_parquet(df, saida)

    resumo = {
        "n_total": int(len(df)),
        "n_geral_obvio": int(obvio_geral.sum()),
        "distribuicao_triagem": df["classificacao_triagem"]
                                  .value_counts().to_dict(),
        "padrao_obvio_top": (
            df.loc[obvio_geral, "padrao_obvio"]
              .value_counts().head(10).to_dict()
        ),
        "limiar_rito": LIMIAR_RITO,
        "sinais_rito_considerados": list(SINAIS_RITO),
    }
    salvar_json(resumo, config.caminho("triagem", "resumo.json"))
    print(f"[triagem] {resumo['distribuicao_triagem']}")
    liberar(df)
    mostrar()
    return saida


def mostrar():
    """Resumo da triagem: óbvios, sinais de rito, distribuição final."""
    from pncp.io_disco import ler_json
    p = config.caminho("triagem", "resumo.json")
    if not p.exists():
        print("[triagem.mostrar] rode pncp.triagem.executar() primeiro")
        return
    r = ler_json(p)
    print(f"\n🔎 Triagem — {r['n_total']:,} contratos")
    print(f"   óbvios de engenharia (rotulo='geral'): {r['n_geral_obvio']}")
    print(f"   classificação:")
    for k, v in r["distribuicao_triagem"].items():
        print(f"     {k}: {v:,}")
    if r.get("padrao_obvio_top"):
        print(f"\n   padrões mais frequentes:")
        for padrao, n in list(r["padrao_obvio_top"].items())[:5]:
            print(f"     {n:4d}× {padrao[:60]}")


def listar_para_ml(caminho_triagem=None):
    """
    Devolve apenas os contratos que precisam de ML (classificacao='ambiguo').
    O pipeline ML deve treinar nos rotulados (engenharia/obras/geral não-óbvio)
    e prever para os 'geral' ambíguos.
    """
    if caminho_triagem is None:
        caminho_triagem = config.caminho("triagem", "triagem.parquet")
    df = ler_parquet(caminho_triagem)
    return df[df["classificacao_triagem"] == "ambiguo"]
