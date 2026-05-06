"""
Consolidação final — junta sinais de todas as camadas e gera o relatório TCC.

Sinais (cada um vale um voto/score):
  - prob_engenharia (Camada 1, classificação TF-IDF)
  - score_engenharia_pdf (Camada 2, marcadores em PDF)
  - tem_mudanca_escopo (Camada 3, aditivos)
  - tem_cnae_eng (CNAE do fornecedor)
  - red_flag_grafo (concentração suspeita)

O ranking final ordena 'geral' por nº de sinais positivos + prob_engenharia.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pncp import config
from pncp.io_disco import (
    ler_parquet, salvar_parquet, ler_json, salvar_json,
)
from pncp.ram import liberar, com_gc


def _ler_se_existe(caminho, eh_json=False):
    p = Path(caminho)
    if not p.exists():
        return None
    return ler_json(p) if eh_json else ler_parquet(p)


# ── Consolidação ─────────────────────────────────────────────────────────────
@com_gc
def consolidar():
    """Junta sinais de todas as etapas num único parquet de suspeitos."""
    base = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"))

    # Camada 1 — ranking de probabilidades
    rank = _ler_se_existe(config.caminho(config.SUB_P2, "ranking.parquet"))
    if rank is not None and "numeroControlePNCP" in rank.columns:
        base = base.merge(rank[["numeroControlePNCP", "prob_engenharia"]],
                           on="numeroControlePNCP", how="left")

    # Camada 2 — features de PDF
    pdfs = _ler_se_existe(config.caminho(config.SUB_C2, "features_pdfs.parquet"))
    if pdfs is not None:
        base = base.merge(
            pdfs[["numeroControlePNCP", "score_engenharia_pdf"]],
            on="numeroControlePNCP", how="left",
        )

    # Camada 3 — aditivos
    adit = _ler_se_existe(config.caminho(config.SUB_C3, "aditivos.parquet"))
    if adit is not None:
        base = base.merge(
            adit[["numeroControlePNCP", "tem_mudanca_escopo"]],
            on="numeroControlePNCP", how="left",
        )

    # CNAE
    fortes = _ler_se_existe(config.caminho(config.SUB_P8, "suspeitos_fortes.parquet"))
    if fortes is not None and "numeroControlePNCP" in fortes.columns:
        ids = set(fortes["numeroControlePNCP"].astype(str))
        base["tem_cnae_eng"] = base["numeroControlePNCP"].astype(str).isin(ids)

    # Conta sinais
    sinais = []
    if "prob_engenharia" in base.columns:
        sinais.append((base["prob_engenharia"].fillna(0) > 0.5).astype(int))
    if "score_engenharia_pdf" in base.columns:
        sinais.append((base["score_engenharia_pdf"].fillna(0) > 0).astype(int))
    if "tem_mudanca_escopo" in base.columns:
        sinais.append(base["tem_mudanca_escopo"].fillna(False).astype(int))
    if "tem_cnae_eng" in base.columns:
        sinais.append(base["tem_cnae_eng"].fillna(False).astype(int))
    base["n_sinais"] = sum(sinais) if sinais else 0

    suspeitos = (base[base["rotulo"] == "geral"]
                 .sort_values(["n_sinais", "prob_engenharia"],
                              ascending=False, na_position="last"))

    saida = config.caminho(config.SUB_P9, "suspeitos_consolidados.parquet")
    salvar_parquet(suspeitos.head(2000), saida)
    print(f"[relatorio] consolidado: {len(suspeitos)} 'geral' avaliados")
    liberar(base)
    return saida


# ── Estatísticas finais ──────────────────────────────────────────────────────
def estatisticas():
    """Resumo numérico do estudo."""
    base = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"))
    susp = _ler_se_existe(config.caminho(config.SUB_P9, "suspeitos_consolidados.parquet"))

    out = {
        "n_total": int(len(base)),
        "distribuicao": base["rotulo"].value_counts().to_dict(),
    }
    if susp is not None:
        out["n_suspeitos_total"] = int(len(susp))
        if "n_sinais" in susp.columns:
            out["por_n_sinais"] = susp["n_sinais"].value_counts().sort_index() \
                                  .to_dict()
        if "valor" in susp.columns:
            out["valor_total_suspeito"] = float(susp["valor"].sum())
            out["valor_n2_sinais"] = float(
                susp[susp["n_sinais"] >= 2]["valor"].sum()
            )
    liberar(base)
    return out


# ── Markdown TCC ─────────────────────────────────────────────────────────────
def _formato_brl(v):
    if v is None or pd.isna(v):
        return "—"
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def gerar_markdown():
    """Gera o relatorio.md final do TCC, lendo tudo do disco."""
    stats = estatisticas()
    metricas = _ler_se_existe(config.caminho(config.SUB_P2, "metricas.json"),
                                eh_json=True) or {}
    eda = _ler_se_existe(config.caminho(config.SUB_EDA, "relatorio.json"),
                          eh_json=True) or {}
    grafos = _ler_se_existe(config.caminho(config.SUB_P7, "resumo.json"),
                              eh_json=True) or {}
    cnae = _ler_se_existe(config.caminho(config.SUB_P8, "resumo.json"),
                            eh_json=True) or {}
    pdfs = _ler_se_existe(config.caminho(config.SUB_C2, "resumo.json"),
                            eh_json=True) or {}

    melhor = metricas.get("melhor_modelo", "lr")
    f1_eng = (metricas.get("holdout", {}).get(melhor, {})
              .get("f1_engenharia", 0))

    linhas = [
        "# Relatório TCC — Subenquadramento de Engenharia no PNCP",
        "",
        f"**Total de contratos analisados:** {stats['n_total']:,}",
        "",
        "## 1. Distribuição por rótulo (Lei 14.133/2021)",
        "",
    ]
    for k, v in stats["distribuicao"].items():
        linhas.append(f"- **{k}**: {v:,}")

    linhas += [
        "",
        "## 2. Classificação supervisionada (Camada 1)",
        "",
        f"- Melhor modelo: **{melhor}**",
        f"- F1-engenharia (holdout): **{f1_eng:.4f}**",
    ]
    if "bootstrap" in metricas and melhor in metricas["bootstrap"]:
        ic = metricas["bootstrap"][melhor].get("f1_eng_ic95")
        if ic:
            linhas.append(f"- IC 95%: [{ic[0]:.3f}, {ic[1]:.3f}]")
    if "mcnemar_lr_vs_rf" in metricas:
        p = metricas["mcnemar_lr_vs_rf"].get("p_valor")
        if p is not None:
            linhas.append(f"- McNemar LR vs RF: p={p:.4f}")

    if pdfs:
        linhas += ["", "## 3. PDFs (Camada 2)", "",
                    f"- Contratos com PDF processado: {pdfs.get('n_contratos_processados', 0)}",
                    f"- Score médio de engenharia em PDF: "
                    f"{pdfs.get('media_score', 0):.2f}"]

    if cnae:
        linhas += ["", "## 4. Enriquecimento via CNAE",
                    f"- CNPJs consultados: {cnae.get('n_cnpjs_consultados', 0)}",
                    f"- Com CNAE de engenharia: {cnae.get('n_com_cnae_eng', 0)}",
                    f"- **Suspeitos fortes** (geral + CNAE eng): "
                    f"{cnae.get('n_suspeitos_fortes', 0)}"]

    if grafos:
        linhas += ["", "## 5. Análise de grafos",
                    f"- {grafos.get('n_orgaos', 0)} órgãos × "
                    f"{grafos.get('n_fornecedores', 0)} fornecedores",
                    f"- Red flags: {grafos.get('red_flags_count', 0)}"]

    if "n_suspeitos_total" in stats:
        linhas += ["", "## 6. Consolidação final",
                    f"- Suspeitos avaliados: {stats['n_suspeitos_total']:,}"]
        if "valor_n2_sinais" in stats:
            linhas.append(
                f"- Valor agregado com ≥2 sinais: "
                f"{_formato_brl(stats['valor_n2_sinais'])}"
            )

    if "alerta_temporal" in eda:
        linhas += ["", "## ⚠ Aviso (EDA)",
                    eda["alerta_temporal"].get("mensagem", "")]

    saida = config.caminho(config.SUB_P9, "relatorio.md")
    saida.write_text("\n".join(linhas), encoding="utf-8")
    print(f"[relatorio] markdown em {saida}")
    return saida


# ── Validação contra ground truth manual ─────────────────────────────────────
def validar_ground_truth(caminho_csv):
    """
    Compara as predições do pipeline contra revisão manual.
    O CSV deve ter coluna 'revisao_manual' com:
        'subenq' = subenquadramento confirmado
        'ok'     = rótulo correto
        'duv'    = duvidoso
    """
    df = pd.read_csv(caminho_csv)
    if "revisao_manual" not in df.columns:
        print("[gt] coluna 'revisao_manual' faltando")
        return None
    df = df[df["revisao_manual"].isin(["subenq", "ok", "duv"])]
    n_subenq = int((df["revisao_manual"] == "subenq").sum())
    n_ok = int((df["revisao_manual"] == "ok").sum())
    n_duv = int((df["revisao_manual"] == "duv").sum())

    metricas = {
        "n_revisados": int(len(df)),
        "subenquadrados": n_subenq,
        "corretos": n_ok,
        "duvidosos": n_duv,
        "precisao": n_subenq / max(1, n_subenq + n_ok),
    }
    salvar_json(metricas, config.caminho(config.SUB_P9, "ground_truth.json"))
    print(f"[gt] precisão = {metricas['precisao']:.3f} "
          f"({n_subenq}/{n_subenq + n_ok})")
    return metricas


# ── Glossário ────────────────────────────────────────────────────────────────
GLOSSARIO = {
    "F1": "Média harmônica de precisão e recall. F1=1 é perfeito; F1=0 é nada.",
    "TF-IDF": "Term Frequency × Inverse Document Frequency. Pesa palavras "
              "raras dentro de um documento mas comuns no corpus inteiro.",
    "Holdout": "Reservar uma parte dos dados (test_size=0.2) para avaliar "
                "o modelo em dados não vistos.",
    "McNemar": "Teste estatístico para comparar dois classificadores no mesmo "
                "conjunto. p<0.05 = diferença significativa.",
    "Bootstrap": "Reamostragem com reposição para estimar intervalo de "
                  "confiança de uma métrica.",
    "LDA": "Latent Dirichlet Allocation — descobre tópicos latentes em texto.",
    "BERTimbau": "BERT pré-treinado em português brasileiro (NeuralMind).",
    "CNAE": "Classificação Nacional de Atividades Econômicas — código que "
             "identifica a atividade principal de uma empresa.",
    "CONFEA/CREA": "Conselhos profissionais de engenharia. CONFEA mantém "
                     "lista oficial de 702 CNAEs de atividades de engenharia.",
    "Lei 14.133/2021": "Nova Lei de Licitações. Define categorias 7 (obras), "
                         "8 (serviços gerais) e 9 (serviços de engenharia).",
}


def glossario(termo=None):
    if termo is None:
        for k, v in GLOSSARIO.items():
            print(f"  {k}: {v}")
        return
    print(GLOSSARIO.get(termo, f"(termo '{termo}' não encontrado)"))


# ── Pipeline principal ──────────────────────────────────────────────────────
@com_gc
def gerar():
    """Roda consolidação + estatísticas + markdown final."""
    consolidar()
    salvar_json(estatisticas(),
                 config.caminho(config.SUB_P9, "estatisticas.json"))
    return gerar_markdown()
