"""
Consolidação final — junta sinais de todas as camadas e gera o relatório TCC.

Sinais (cada um vale um voto/score):
  - prob_engenharia (Camada 1, classificação TF-IDF)
  - mk_score_engenharia (Camada 2, marcadores em PDF)
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


# ── Triagem (etapa 0) ────────────────────────────────────────────────────────
def integrar_triagem(base):
    """Adiciona ao DataFrame as colunas da triagem se já existirem em disco."""
    triagem = _ler_se_existe(config.caminho("triagem", "triagem.parquet"))
    if triagem is None:
        return base
    cols = [c for c in ("numeroControlePNCP", "classificacao_triagem",
                          "eh_obvio_engenharia", "n_sinais_rito",
                          "seguiu_rito")
            if c in triagem.columns]
    if "numeroControlePNCP" not in cols:
        return base
    return base.merge(triagem[cols], on="numeroControlePNCP", how="left")


# ── Consolidação ─────────────────────────────────────────────────────────────
@com_gc
def consolidar():
    """Junta sinais de todas as etapas num único parquet de suspeitos."""
    base = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"))

    # Etapa 0 — triagem determinística (pré-filtro + verificação de rito)
    base = integrar_triagem(base)

    # Camada 1 — ranking de probabilidades
    rank = _ler_se_existe(config.caminho(config.SUB_P2, "ranking.parquet"))
    if rank is not None and "numeroControlePNCP" in rank.columns:
        base = base.merge(rank[["numeroControlePNCP", "prob_engenharia"]],
                           on="numeroControlePNCP", how="left")

    # Camada 2 — features de PDF
    pdfs = _ler_se_existe(config.caminho(config.SUB_C2, "features_pdfs.parquet"))
    if pdfs is not None:
        base = base.merge(
            pdfs[["numeroControlePNCP", "mk_score_engenharia"]],
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

    # LLM — veredito semântico (se pncp.llm.validar_suspeitos() já rodou).
    # O LLM lê os suspeitos consolidados de uma rodada anterior; por isso
    # este sinal só aparece a partir da 2ª chamada de consolidar(). Fluxo:
    #   consolidar() → llm.validar_suspeitos() → consolidar() (pega o LLM).
    llm = _ler_se_existe(config.caminho("llm", "validacao_llm.parquet"))
    if (llm is not None and "numeroControlePNCP" in llm.columns
            and "llm_classe" in llm.columns):
        cols_llm = ["numeroControlePNCP", "llm_classe"]
        if "llm_confianca" in llm.columns:
            cols_llm.append("llm_confianca")
        base = base.merge(llm[cols_llm], on="numeroControlePNCP", how="left")
        conf = (base["llm_confianca"].fillna(0)
                if "llm_confianca" in base.columns else 1.0)
        # Sinal: LLM apontou engenharia/obras com confiança >= 0.6
        base["llm_aponta_subenq"] = (
            base["llm_classe"].isin(["engenharia", "obras"]) & (conf >= 0.6)
        )

    # Conta sinais
    sinais = []
    if "prob_engenharia" in base.columns:
        sinais.append((base["prob_engenharia"].fillna(0) > 0.5).astype(int))
    if "mk_score_engenharia" in base.columns:
        sinais.append((base["mk_score_engenharia"].fillna(0) > 0).astype(int))
    if "tem_mudanca_escopo" in base.columns:
        sinais.append(base["tem_mudanca_escopo"].fillna(False).astype(int))
    if "tem_cnae_eng" in base.columns:
        sinais.append(base["tem_cnae_eng"].fillna(False).astype(int))
    if "llm_aponta_subenq" in base.columns:
        sinais.append(base["llm_aponta_subenq"].fillna(False).astype(int))
    base["n_sinais"] = sum(sinais) if sinais else 0

    suspeitos = (base[base["rotulo"] == "geral"]
                 .sort_values(["n_sinais", "prob_engenharia"],
                              ascending=False, na_position="last"))

    saida = config.caminho(config.SUB_P9, "suspeitos_consolidados.parquet")
    salvar_parquet(suspeitos.head(2000), saida)

    # Veredito final por contrato — combina triagem (determinística) + ML
    if "classificacao_triagem" in suspeitos.columns:
        # Para os 'ambiguos' da triagem, decide pelo n_sinais do ML
        veredito = suspeitos["classificacao_triagem"].copy()
        massa_ambigua = veredito == "ambiguo"
        if "n_sinais" in suspeitos.columns:
            forte = massa_ambigua & (suspeitos["n_sinais"] >= 2)
            fraco = massa_ambigua & (suspeitos["n_sinais"] < 2)
            veredito.loc[forte] = "subenquadramento_provavel_ml"
            veredito.loc[fraco] = "provavel_geral_real"
        suspeitos["veredito_final"] = veredito
        salvar_parquet(suspeitos.head(2000),
                       config.caminho(config.SUB_P9,
                                       "suspeitos_consolidados.parquet"))
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
    triagem = _ler_se_existe(config.caminho("triagem", "resumo.json"),
                              eh_json=True) or {}
    grafos = _ler_se_existe(config.caminho(config.SUB_P7, "resumo.json"),
                              eh_json=True) or {}
    cnae = _ler_se_existe(config.caminho(config.SUB_P8, "resumo.json"),
                            eh_json=True) or {}
    pdfs = _ler_se_existe(config.caminho(config.SUB_C2, "resumo.json"),
                            eh_json=True) or {}
    llm = _ler_se_existe(config.caminho("llm", "resumo.json"),
                          eh_json=True) or {}

    melhor = metricas.get("melhor_modelo", "lr")
    f1_eng = (metricas.get("holdout", {}).get(melhor, {})
              .get("f1_engenharia", 0))

    # Contador autoincremental para evitar numeração duplicada/saltada
    n_secao = [0]
    def _secao(titulo):
        n_secao[0] += 1
        return f"## {n_secao[0]}. {titulo}"

    linhas = [
        "# Relatório TCC — Subenquadramento de Engenharia no PNCP",
        "",
        f"**Total de contratos analisados:** {stats['n_total']:,}",
        "",
        _secao("Distribuição por rótulo (Lei 14.133/2021)"),
        "",
    ]
    for k, v in stats["distribuicao"].items():
        linhas.append(f"- **{k}**: {v:,}")

    if triagem:
        linhas += [
            "",
            _secao("Triagem determinística (etapa 0)"),
            "",
            f"- Contratos 'geral' óbvios de engenharia: "
            f"**{triagem.get('n_geral_obvio', 0)}**",
            "",
            "Distribuição final da triagem:",
        ]
        for k, v in (triagem.get("distribuicao_triagem") or {}).items():
            linhas.append(f"- `{k}`: {v:,}")
        linhas += [
            "",
            "Interpretação:",
            "- `subenquadramento_real`: óbvio engenharia + rito não seguido → "
            "**provável violação da Lei 14.133/2021**",
            "- `rotulacao_incorreta_processo_ok`: óbvio engenharia + rito "
            "seguido → erro de cadastro, mas processo correto",
            "- `ambiguo`: precisa de classificador ML (ver próxima seção)",
        ]

    linhas += [
        "",
        _secao("Classificação supervisionada (ML — para os ambíguos)"),
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
        linhas += ["", _secao("PDFs (Camada 2)"), "",
                    f"- Contratos com PDF processado: {pdfs.get('n_contratos_processados', 0)}",
                    f"- Score médio de engenharia em PDF: "
                    f"{pdfs.get('media_score', 0):.2f}"]

    if cnae:
        linhas += ["", _secao("Enriquecimento via CNAE"),
                    f"- CNPJs consultados: {cnae.get('n_cnpjs_consultados', 0)}",
                    f"- Com CNAE de engenharia: {cnae.get('n_com_cnae_eng', 0)}",
                    f"- **Suspeitos fortes** (geral + CNAE eng): "
                    f"{cnae.get('n_suspeitos_fortes', 0)}"]

    if llm:
        dist = llm.get("distribuicao_llm", {})
        dist_txt = ", ".join(f"{k}={v}" for k, v in dist.items()) or "—"
        linhas += ["", _secao("Validação semântica por LLM"),
                    f"- Backend: **{llm.get('backend', '?')}** "
                    f"(modelo {llm.get('modelo', '?')})",
                    f"- Contratos avaliados: {llm.get('n_avaliados', 0)}",
                    f"- Classificação do LLM: {dist_txt}",
                    f"- **Subenquadramentos apontados** (engenharia/obras): "
                    f"{llm.get('n_subenquadramentos_apontados', 0)}",
                    f"- Exigem ART segundo o LLM: "
                    f"{llm.get('n_exigem_art', 0)}",
                    "",
                    "O veredito do LLM entra como mais um sinal na consolidação "
                    "(conta junto com prob ML, marcadores em PDF, CNAE e aditivos)."]

    if grafos:
        linhas += ["", _secao("Análise de grafos"),
                    f"- {grafos.get('n_orgaos', 0)} órgãos × "
                    f"{grafos.get('n_fornecedores', 0)} fornecedores",
                    f"- Red flags: {grafos.get('red_flags_count', 0)}"]

    if "n_suspeitos_total" in stats:
        linhas += ["", _secao("Consolidação final"),
                    f"- Suspeitos avaliados: {stats['n_suspeitos_total']:,}"]
        if "valor_n2_sinais" in stats:
            linhas.append(
                f"- Valor agregado com ≥2 sinais: "
                f"{_formato_brl(stats['valor_n2_sinais'])}"
            )

    if "alerta_temporal" in eda:
        linhas += ["", "## ⚠ Aviso (EDA)",
                    eda["alerta_temporal"].get("mensagem", "")]

    # ── Interpretação dos resultados (contextualiza os números) ──────────
    linhas += ["", _secao("Interpretação dos resultados"), ""]

    n_total = stats.get("n_total", 0)
    dist = stats.get("distribuicao", {})
    n_geral = dist.get("geral", 0)
    n_eng = dist.get("engenharia", 0)
    n_obras = dist.get("obras", 0)
    pct_eng = (n_eng / n_total * 100) if n_total else 0
    pct_obras_eng = ((n_eng + n_obras) / n_total * 100) if n_total else 0

    linhas += [
        f"**Composição da base:** {n_geral:,} 'geral' ({n_geral/max(n_total,1):.1%}), "
        f"{n_obras:,} 'obras' ({n_obras/max(n_total,1):.1%}), "
        f"{n_eng:,} 'engenharia' ({n_eng/max(n_total,1):.1%}).",
        "",
        f"**Baseline a bater (F1-engenharia):** se {pct_eng:.1f}% das amostras "
        f"são engenharia, um modelo aleatório atinge F1 ≈ {pct_eng/100:.2f}. "
        f"Já um modelo trivial (sempre prevê 'geral') tem F1-engenharia = **0** — "
        f"é o piso real.",
        f"- Nosso F1-engenharia: **{f1_eng:.3f}** — "
        f"{'acima do baseline aleatório' if f1_eng > pct_eng/100 else 'precisa melhorar'}",
    ]

    # Matriz de erro detalhada (cruzamento pipeline × revisão manual)
    matriz_path = config.caminho(config.SUB_P9, "matriz_erro.json")
    if matriz_path.exists():
        me = ler_json(matriz_path)
        linhas += [
            "",
            "**Matriz de erro (pipeline × revisão manual):**",
            "",
            "| Categoria | Contagem | Significado |",
            "|---|---|---|",
            f"| VP (verdadeiro positivo) | {me.get('VP_verdadeiro_positivo', 0)} | "
            f"Pipeline marcou e é mesmo subenquadramento — ✅ acertou |",
            f"| FP (falso positivo)     | {me.get('FP_falso_positivo', 0)} | "
            f"Pipeline marcou mas é ok — ⚠ falso alarme |",
            f"| FN (falso negativo)     | {me.get('FN_falso_negativo', 0)} | "
            f"Pipeline NÃO marcou mas é subenq — ❌ vacilo grave |",
            f"| VN (verdadeiro negativo)| {me.get('VN_verdadeiro_negativo', 0)} | "
            f"Pipeline não marcou e é ok — ✅ correto |",
            "",
            f"- **Precisão**: {me.get('precisao', 0):.1%} "
            f"(dos suspeitos do pipeline, quantos são reais)",
            f"- **Recall**: {me.get('recall', 0):.1%} "
            f"(dos subenq reais, quantos o pipeline pegou)",
            f"- **F1**: {me.get('f1', 0):.3f}",
            f"- **Acurácia**: {me.get('acuracia', 0):.1%}",
        ]
    else:
        # Ground truth simples (se só houver precisão básica)
        gt_path = config.caminho(config.SUB_P9, "ground_truth.json")
        if gt_path.exists():
            gt = ler_json(gt_path)
            prec = gt.get("precisao", 0)
            n_rev = gt.get("n_revisados", 0)
            linhas += [
                "",
                f"**Validação manual ({n_rev} revisados):**",
                f"- Subenquadramentos confirmados: {gt.get('subenquadrados', 0)}",
                f"- Rotulação correta: {gt.get('corretos', 0)}",
                f"- Duvidosos: {gt.get('duvidosos', 0)}",
                f"- **Precisão da pipeline: {prec:.1%}**",
            ]

    # Achados-chave
    linhas += ["", "### Achados-chave", ""]
    achados = []
    if cnae and cnae.get("n_suspeitos_fortes"):
        achados.append(
            f"- **{cnae['n_suspeitos_fortes']:,} contratos 'geral' com "
            f"fornecedor de CNAE engenharia** (CONFEA) — alta chance de "
            f"subenquadramento."
        )
    if triagem:
        d = triagem.get("distribuicao_triagem", {})
        n_real = d.get("subenquadramento_real", 0)
        if n_real:
            achados.append(
                f"- **{n_real:,} contratos com sinal de subenquadramento real** "
                f"(óbvio engenharia + sem rito formal nos PDFs)."
            )
    if pdfs and pdfs.get("media_score"):
        achados.append(
            f"- Score médio de engenharia em PDFs analisados: "
            f"{pdfs['media_score']:.2f}/9.0"
        )
    if grafos and grafos.get("red_flags_count"):
        achados.append(
            f"- {grafos['red_flags_count']} red flags no grafo "
            f"órgão↔fornecedor (concentração suspeita)."
        )
    if not achados:
        achados = ["(rode mais etapas do pipeline para gerar achados)"]
    linhas += achados

    # Nota sobre desbalanceamento (se aplicável)
    if pct_eng < 10:
        linhas += [
            "",
            "### ⚠ Nota sobre desbalanceamento",
            "",
            f"Apenas {pct_eng:.1f}% da base é 'engenharia'. Métricas de classes "
            f"raras (F1, recall) tendem a ser baixas mesmo com bom modelo. "
            f"Estratégias adotadas: `class_weight='balanced'`, oversampling "
            f"(SMOTE opcional), bootstrap para IC.",
        ]

    # ── Glossário (sempre no final do relatório) ────────────────────────
    linhas += ["", _secao("Glossário"), ""]
    # Remove duplicatas mantendo a primeira definição de cada termo
    vistos = set()
    for termo, definicao in GLOSSARIO.items():
        if termo in vistos:
            continue
        vistos.add(termo)
        linhas.append(f"- **{termo}**: {definicao}")

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
    "Rito de engenharia": "Conjunto de exigências formais para um contrato de "
                            "engenharia: ART, projeto básico/executivo, memorial "
                            "descritivo, engenheiro responsável, normas ABNT NBR.",
    "ART": "Anotação de Responsabilidade Técnica — emitida pelo CREA. "
            "Obrigatória para qualquer obra/serviço de engenharia. "
            "Escopo do estudo: apenas CREA/ART (engenharia); CAU/RRT "
            "(arquitetura) ficam fora.",
    "Subenquadramento real": "Contrato de engenharia rotulado como 'serviços "
                                "gerais' E que não seguiu o rito formal. Viola "
                                "a Lei 14.133/2021.",
    "Outlier (anomalia)": "Contrato com características muito distintas dos seus "
                            "pares no cluster 'geral'. Candidato a subenquadramento.",
    "Precisão": "P = VP / (VP + FP). Dos contratos que o pipeline marcou como "
                "suspeitos, quantos são realmente subenquadramento? Mais alta = "
                "menos falso alarme.",
    "Recall (revocação)": "R = VP / (VP + FN). Dos contratos que SÃO subenquadramento "
                            "(verdade), quantos o pipeline detectou? Mais alto = pega "
                            "mais casos reais (a custo de mais falso alarme).",
    "VP/FP/VN/FN": "Verdadeiro/Falso Positivo, Verdadeiro/Falso Negativo. "
                     "Combinados, definem precisão, recall, F1 e acurácia.",
    "Matriz de confusão": "Tabela 4×N classes mostrando previsões corretas (diagonal) "
                            "e erradas (fora da diagonal). Visualiza onde o modelo "
                            "se confunde.",
    "Categoria PNCP": "Lei 14.133/2021 art. 6º. Categorias relevantes: 7=Obras, "
                        "8=Serviços (Gerais), 9=Serviços de Engenharia.",
    "Stratified sampling": "Amostragem que preserva proporções (por classe, órgão, "
                            "ano…). Evita que a amostra seja enviesada para uma "
                            "subpopulação.",
    "Uncertainty sampling": "Active learning: para revisão humana, escolher os casos "
                              "em que o modelo está MAIS INCERTO (prob ≈ 0.5), pois "
                              "rotular esses é mais informativo que rotular óbvios.",
    "Skip-if-exists": "Pular um passo do pipeline se o output dele já existe em "
                        "disco. Use forcar=True para refazer mesmo se já rodou.",
    "Snapshot": "Cópia congelada do estado de dados/ em dado momento. Use "
                  "pncp.snapshot_auto() antes de Run all repetido para preservar.",
    "Distant supervision": "Em vez de rotular manualmente, usar uma heurística "
                              "(ex: regex de termos óbvios) para gerar rótulos fracos. "
                              "Usado na triagem etapa 0.",
    "Imbalanced classes": "Quando uma classe domina (ex: 80% 'geral' vs 6% "
                            "'engenharia'). F1 baixo é matematicamente esperado. "
                            "Comparar com baseline de classe majoritária.",
    "Baseline majoritário": "Modelo trivial que sempre prevê a classe mais comum. "
                              "Em 80% geral, ele acerta 80% das vezes (acurácia) mas "
                              "F1 da classe minoritária = 0. Mede o piso real.",
    "Mudança de escopo (Camada 3)": "Contrato original 'geral' que recebeu aditivo "
                                       "com marcadores de engenharia. A licitação não "
                                       "seguiu rito formal mas a execução incluiu "
                                       "trabalho que exigiria.",
    "ROC-AUC": "Área sob a curva ROC. 0.5 = aleatório, 1.0 = perfeito. Mede "
                  "capacidade de ranking (independente de threshold).",
}


def glossario(termo=None):
    if termo is None:
        for k, v in GLOSSARIO.items():
            print(f"  {k}: {v}")
        return
    print(GLOSSARIO.get(termo, f"(termo '{termo}' não encontrado)"))


# ── Pipeline principal ──────────────────────────────────────────────────────
def exportar_suspeitos_completo():
    """
    Gera a LISTA COMPLETA de suspeitos consolidados em CSV + XLSX,
    estratificada por nível de confiança. Inclui todas as evidências
    (ML prob, marcadores, CNAE, aditivos) para uso jurídico/auditorial.

    Saídas em dados/relatorio/:
      - suspeitos_completos.csv      (todos os contratos 'geral' avaliados)
      - suspeitos_alta_confianca.csv (subset: >= 3 sinais positivos)
      - suspeitos_completos.xlsx     (mesmo conteúdo, formato planilha)
    """
    p = config.caminho(config.SUB_P9, "suspeitos_consolidados.parquet")
    if not Path(p).exists():
        print("[exportar] rode pncp.relatorio.gerar() primeiro")
        return None
    df = ler_parquet(p)
    if df.empty:
        print("[exportar] nenhum suspeito consolidado")
        return None

    # Categoriza por nível de confiança
    n_sinais = df.get("n_sinais", 0)
    df["nivel_confianca"] = pd.cut(
        n_sinais,
        bins=[-0.1, 0, 1, 2, 99],
        labels=["sem_sinal", "fraco_1", "moderado_2", "forte_3+"],
    )

    # CSV completo
    saida_csv = config.caminho(config.SUB_P9, "suspeitos_completos.csv")
    df.to_csv(saida_csv, index=False, encoding="utf-8-sig")
    print(f"[exportar] CSV completo: {saida_csv} ({len(df):,} linhas)")

    # Subset alta confiança
    alta = df[df["nivel_confianca"] == "forte_3+"]
    if not alta.empty:
        saida_alta = config.caminho(config.SUB_P9,
                                      "suspeitos_alta_confianca.csv")
        alta.to_csv(saida_alta, index=False, encoding="utf-8-sig")
        print(f"[exportar] alta confiança (≥3 sinais): {saida_alta} "
              f"({len(alta):,} linhas)")

    # XLSX para revisão fácil (se openpyxl estiver disponível)
    try:
        saida_xlsx = config.caminho(config.SUB_P9, "suspeitos_completos.xlsx")
        with pd.ExcelWriter(saida_xlsx, engine="openpyxl") as wr:
            df.to_excel(wr, sheet_name="todos", index=False)
            if not alta.empty:
                alta.to_excel(wr, sheet_name="alta_confianca", index=False)
        print(f"[exportar] XLSX: {saida_xlsx}")
    except Exception as e:
        print(f"[exportar] XLSX falhou (openpyxl ausente?): {e}")

    return saida_csv


def gerar_matriz_erro():
    """
    Cruza predições do pipeline com revisão manual (ground truth) e
    calcula matriz de confusão jurídica: VP, FP, VN, FN.

    Salva tabela com nivel_de_erro por contrato revisado.
    """
    p_gt = config.caminho(config.SUB_P8, "amostra_revisao_manual.csv")
    p_susp = config.caminho(config.SUB_P9, "suspeitos_consolidados.parquet")
    if not Path(p_gt).exists() or not Path(p_susp).exists():
        return None
    gt = pd.read_csv(p_gt)
    if "revisao_manual" not in gt.columns:
        return None
    gt = gt[gt["revisao_manual"].fillna("").astype(str).ne("")]
    if gt.empty:
        return None

    susp = ler_parquet(p_susp)
    # Suspeito = contrato com ≥1 sinal positivo no pipeline
    susp["pipeline_suspeito"] = susp.get("n_sinais", 0) >= 1
    gt_m = gt.merge(
        susp[["numeroControlePNCP", "pipeline_suspeito", "n_sinais"]],
        on="numeroControlePNCP", how="left",
    )
    gt_m["pipeline_suspeito"] = gt_m["pipeline_suspeito"].fillna(False)
    gt_m["realmente_subenq"] = gt_m["revisao_manual"] == "subenq"

    def _tipo(r):
        if r["pipeline_suspeito"] and r["realmente_subenq"]:
            return "VP (acerto: subenq detectado)"
        if r["pipeline_suspeito"] and not r["realmente_subenq"]:
            return "FP (falso alarme: ok mas pipeline marcou)"
        if not r["pipeline_suspeito"] and r["realmente_subenq"]:
            return "FN (vacilo: subenq mas pipeline NÃO detectou)"
        return "VN (correto: ok e pipeline também)"

    gt_m["tipo_erro"] = gt_m.apply(_tipo, axis=1)
    cont = gt_m["tipo_erro"].value_counts()
    vp = int((gt_m["tipo_erro"].str.startswith("VP")).sum())
    fp = int((gt_m["tipo_erro"].str.startswith("FP")).sum())
    fn = int((gt_m["tipo_erro"].str.startswith("FN")).sum())
    vn = int((gt_m["tipo_erro"].str.startswith("VN")).sum())

    metricas = {
        "VP_verdadeiro_positivo": vp,
        "FP_falso_positivo": fp,
        "FN_falso_negativo": fn,
        "VN_verdadeiro_negativo": vn,
        "precisao": vp / max(vp + fp, 1),
        "recall": vp / max(vp + fn, 1),
        "f1": 2 * vp / max(2 * vp + fp + fn, 1),
        "acuracia": (vp + vn) / max(vp + vn + fp + fn, 1),
        "contagem_por_tipo": cont.to_dict(),
    }
    saida = config.caminho(config.SUB_P9, "matriz_erro.json")
    salvar_json(metricas, saida)
    # CSV com classificação por contrato (para inspeção)
    gt_m[["numeroControlePNCP", "objeto", "revisao_manual",
            "pipeline_suspeito", "n_sinais", "tipo_erro"]].to_csv(
        config.caminho(config.SUB_P9, "matriz_erro_por_contrato.csv"),
        index=False, encoding="utf-8-sig",
    )
    print(f"[matriz_erro] VP={vp}, FP={fp}, FN={fn}, VN={vn}")
    print(f"   precisão={metricas['precisao']:.3f}, "
          f"recall={metricas['recall']:.3f}, "
          f"F1={metricas['f1']:.3f}")
    return metricas


@com_gc
def gerar():
    """Roda consolidação + estatísticas + markdown final + matriz de erro."""
    from pncp.ram import precisa_de
    if not precisa_de(config.caminho(config.SUB_COLETA, "contratos.parquet"),
                       "relatorio",
                       "rode pncp.coleta.coletar(...) primeiro"):
        return None
    consolidar()
    try:
        exportar_suspeitos_completo()
    except Exception as e:
        print(f"[relatorio] export completo falhou: {e}")
    try:
        gerar_matriz_erro()
    except Exception as e:
        print(f"[relatorio] matriz_erro falhou: {e}")
    salvar_json(estatisticas(),
                 config.caminho(config.SUB_P9, "estatisticas.json"))
    md = gerar_markdown()
    # Snapshot automático dos resultados (NÃO inclui coleta nem cache PDF)
    try:
        from pncp import snapshot_resultados
        snapshot_resultados()
    except Exception as e:
        print(f"[relatorio] snapshot automático falhou: {e}")
    return md
