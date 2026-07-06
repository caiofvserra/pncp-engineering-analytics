# Briefing para gerar o slide de Metodologia (TCC)

> Autossuficiente: entregue-o a uma IA geradora de imagens/slides para produzir
> **um único slide** de metodologia. Contém contexto, descrição detalhada de cada
> etapa (o que faz / por quê / entra / sai / **técnica**), o resumo técnico do
> stack e as instruções de design.

---

## 1. Contexto da pesquisa

**Problema.** No **PNCP** (Portal Nacional de Contratações Públicas), cada
contrato recebe uma **categoria** informada pelo órgão: *obras*, *serviços de
engenharia* ou *serviços gerais*, entre outras. Muitos contratos de
**engenharia/obras** são cadastrados como **"serviços gerais"** —
**subenquadramento**. É irregular pela **Lei 14.133/2021**, pois obras e serviços
de engenharia exigem **rito próprio** (ART/CREA, projeto básico, responsável
técnico, ABNT, planilha orçamentária). Rotular como "serviços gerais" burla esse
rito.

**Objetivo.** Pipeline de dados + IA que **detecte, na massa de "serviços
gerais", os contratos que são de fato engenharia/obras** e, para os mais
suspeitos, **verifique documentalmente** o cumprimento do rito — separando
*subenquadramento real* de *mero erro de rótulo*.

**Escopo.** Somente **engenharia** (CREA/ART). Arquitetura (CAU/RRT) fora.

**Insight metodológico — Positive-Unlabeled (PU) Learning.** *engenharia* e
*obras* são **positivos confiáveis**; *serviços gerais* é a classe **não-rotulada
ruidosa**. Aprendizado feito só com positivos confiáveis + não-rotulados.

**Dados.** ~30 mil+ contratos da API pública do PNCP; unidade de texto = **objeto
da contratação**.

**Base legal.** Lei 14.133/2021 (licitações) · Lei 5.194/1966 (exercício da
engenharia) · resoluções do **CONFEA**.

---

## 2. Fases e etapas (o que / por quê / entra / sai / **técnica**)

### FASE A — Coleta e preparação
**1. Coleta de dados**
- *O que / por quê:* baixa contratos do PNCP e filtra as categorias de interesse.
- *Entra:* API REST do PNCP. *Sai:* base rotulada (`categoriaProcesso`).
- *Técnica:* ingestão via API `/v1/contratos`, paginação, persistência em
  **Parquet**; rótulo derivado de `categoriaProcessoId`.

**2. Pré-processamento**
- *O que / por quê:* padroniza o objeto e remove o **boilerplate** burocrático
  ("contratação de empresa especializada para prestação de serviços de…") que não
  discrimina e infla a similaridade.
- *Entra:* base bruta. *Sai:* objetos limpos.
- *Técnica:* normalização Unicode + *lowercasing*; **regex** de prefixos;
  tokenização + **stemmer RSLP** (português) e **stopwords** de domínio para as
  análises lexicais.

### FASE B — Modelagem (PU Learning)
**3. Representação semântica + filtro PU**
- *O que / por quê:* vetoriza cada objeto e separa, entre os "gerais", os
  **candidatos** (próximos da engenharia) dos **negativos confiáveis** (distantes).
- *Entra:* objetos limpos. *Sai:* embeddings + máscara candidato/negativo.
- *Técnica:* **Sentence-BERT** (modelo `distiluse-base-multilingual-cased-v1`),
  embeddings **normalizados em L2**; **centróide** dos positivos (eng+obras);
  **similaridade do cosseno** objeto↔centróide; corte por **quantil** (top 30%
  dos "gerais" mais próximos = candidatos).

**4. Agrupamento (clusterização)**
- *O que / por quê:* agrupa os candidatos e mede a **pureza** (densidade de
  eng/obras) de cada grupo, usada depois como *prior*.
- *Entra:* embeddings dos candidatos. *Sai:* clusters + `pct_certeiros`.
- *Técnica:* **K-Means** com **k automático (6–12)** escolhido pelo maior
  **silhouette score** (métrica do cosseno); descritores por cluster via
  **TF-IDF** (uni+bigramas, stemmer RSLP).

**5. Vocabulário de domínio (apoio de LLM)**
- *O que / por quê:* extrai da própria base o léxico que **discrimina** eng ×
  não-eng; vira **contexto** para as etapas de LLM (10 e 11).
- *Entra:* casos confiáveis. *Sai:* vocabulário/perfis de domínio.
- *Técnica:* **log-odds ratio** (com suavização) sobre `CountVectorizer` binário
  entre positivos e negativos; perfis sintetizados por **LLM**; curadoria manual
  para remover ruído (topônimos, numerais).

**6. Treino + calibração do classificador**
- *O que / por quê:* treina só com **casos confiáveis** (positivos = eng+obras;
  negativos = "gerais" distantes), seleciona o melhor e calibra a probabilidade.
- *Entra:* embeddings rotulados. *Sai:* modelo calibrado.
- *Técnica:* comparação de **8 classificadores** — **Regressão Logística,
  Random Forest, Extra-Trees, Gradient Boosting, SVM (kernel RBF), k-NN, MLP,
  Naive Bayes** — sobre os embeddings SBERT; *holdout* estratificado (80/20);
  seleção por **F1 macro**; **calibração isotônica** (`CalibratedClassifierCV`)
  para probabilidades confiáveis.

### FASE C — Detecção e validação
**7. Pontuação + ranqueamento**
- *O que / por quê:* aplica o modelo a **todos** os "gerais" e prioriza.
- *Entra:* embeddings dos "gerais" + modelo. *Sai:* **ranking de suspeitos**.
- *Técnica:* `predict_proba` (prob. de engenharia); **score combinado** =
  `prob × (0,4 + 0,6 × pureza_do_cluster)` (injeta o *prior* de domínio da etapa
  4); ordenação decrescente.

**8. Validação manual**
- *O que / por quê:* mede o desempenho **real** e fixa o corte.
- *Entra:* amostra rotulada à mão (n≈200). *Sai:* métricas + limiar.
- *Técnica:* amostragem **aleatória com cota mínima por faixa** de probabilidade
  (estratificação leve); *ground truth* humano; **varredura de limiar** (0,30–0,95)
  maximizando **F1**; reporta **precisão, recall, F1** e **matriz de confusão**.

**9. Visualização**
- *O que / por quê:* evidência visual de que os suspeitos ficam junto da
  engenharia confirmada.
- *Entra:* embeddings + classe. *Sai:* mapas e rede.
- *Técnica:* redução de dimensionalidade **UMAP** (2D, `n_neighbors=15`,
  `min_dist=0.1`, métrica do cosseno); **grafo de k-vizinhos** (`kneighbors_graph`)
  renderizado com **NetworkX** + detecção de **comunidades por modularidade**;
  gráficos em Matplotlib/Plotly.

**10. Revisão por IA (checagem cruzada)**
- *O que / por quê:* filtra falsos positivos antes da verificação documental.
- *Entra:* suspeitos do topo. *Sai:* suspeitos confirmados/refinados.
- *Técnica:* **LLM Llama 3.1** (via **Ollama**, local em GPU) com **saída
  estruturada JSON** e **contexto de domínio** (etapa 5); regra de decisão por
  classe + confiança.

### FASE D — Verificação e entrega
**11. Análise do rito de engenharia (evidência definitiva)**
- *O que / por quê:* confirma documentalmente se o rito foi seguido.
- *Entra:* documentos da licitação dos suspeitos. *Sai:* **veredito por contrato**.
- *Técnica:* resolução **contrato → compra** pelo `numeroControlePNCP` (API do
  PNCP `/orgaos/{cnpj}/compras/{ano}/{seq}/arquivos`); download de Edital/Termo de
  Referência/Projeto Básico; **extração de texto com PyMuPDF**; **marcadores
  legais por regex** (ART, CREA, projeto básico, responsável técnico, ABNT NBR,
  art. 6º da Lei 14.133, **planilha orçamentária, BDI, SINAPI, SICRO, cronograma
  físico-financeiro, caderno de encargos, alvará**); veredito do **LLM** sobre o
  trecho do TR; classificação em **subenquadramento_real** vs
  **rótulo_incorreto_processo_ok** vs *indeterminado*.

**12. Consolidação e reuso**
- *O que / por quê:* entrega a ferramenta e os resultados.
- *Entra:* modelo + novos contratos. *Sai:* relatório + triagem automática.
- *Técnica:* serialização do **modelo (`joblib`)**; exportação de **CSVs** e
  **relatório**; função de inferência para novos objetos (SBERT → modelo →
  probabilidade), consultando a API do PNCP para o período mais recente.

---

## 3. Resumo técnico (stack e artefatos gerados)

**Representação:** Sentence-BERT `distiluse-base-multilingual-cased-v1`
(embeddings L2) · TF-IDF (uni+bigramas) + stemmer RSLP como descritor.
**Aprendizado:** PU Learning (centróide + limiar por quantil); K-Means (silhouette);
8 classificadores supervisionados; seleção por F1 macro; calibração isotônica.
**Priorização:** probabilidade calibrada × pureza de cluster (score combinado).
**Avaliação:** validação manual estratificada; precisão/recall/F1; varredura de
limiar; matriz de confusão; silhouette.
**IA generativa (LLM):** Llama 3.1 via Ollama (GPU), saída JSON, contexto de
domínio derivado por log-odds.
**Verificação documental:** API PNCP + PyMuPDF + marcadores legais por regex.
**Visualização:** UMAP 2D, grafo k-NN (NetworkX + modularidade), Matplotlib/Plotly.
**Ferramentas:** Python, scikit-learn, sentence-transformers, umap-learn, NetworkX,
PyMuPDF, pandas; execução em **Google Colab (GPU)**.

**Artefatos gerados:** matriz de embeddings; ranking de suspeitos; métricas de
validação (precisão/recall/F1 + limiar); figuras (distribuição, filtro PU,
clusters, UMAP, grafo k-NN, curva de limiar, matriz de confusão); veredito de
rito por contrato; **modelo serializado** e relatório final.

---

## 4. Fio condutor (a "história" do slide)

Base bruta → limpeza → **separar o suspeito do comum** (PU + SBERT) →
**treinar e calibrar o modelo** → **ranquear todos os "gerais"** →
**validar com rigor** (precisão/recall) → **confirmar com LLM** →
**provar no documento** (rito) → **entregar a ferramenta**.

---

## 5. Instruções de design (para a IA que vai gerar o slide)

- **UM slide, 16:9, horizontal.**
- **4 blocos/colunas** (as 4 fases) com etapas numeradas curtas; **setas** entre
  blocos rotuladas com o que passa: *objetos limpos → modelo calibrado →
  suspeitos priorizados*.
- **Cores por fase:** A azul, B laranja, C verde, D vermelho.
- **Hierarquia:** título; subtítulo com a ideia central (PU Learning); blocos;
  rodapé com a base legal (Lei 14.133/2021 · Lei 5.194/66 · CONFEA — só CREA/ART).
- Incluir, discretamente, os **termos técnicos-chave** por fase (ex.: “SBERT +
  cosseno”, “K-Means/silhouette”, “8 classificadores + calibração”, “UMAP + grafo
  k-NN”, “LLM Llama 3.1”, “ART/CREA · PyMuPDF · regex”).
- **Tom:** acadêmico, limpo, legível de longe. Ícones simples (banco de dados,
  vetor/rede, gráfico, lupa/documento). **Sem** nomes de arquivo.
- **Enfatizar:** (a) o insight PU; (b) etapa 8 = validação/rigor; (c) etapa 11 =
  rito/prova definitiva.
