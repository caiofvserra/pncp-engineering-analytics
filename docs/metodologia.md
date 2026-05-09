# Metodologia — Identificação de Subenquadramento de Engenharia em PNCP

## Problema

A Lei 14.133/2021 distingue três categorias de contrato:

- **7 — Obras**
- **8 — Serviços gerais** (limpeza, vigilância, alimentação…)
- **9 — Serviços de engenharia**

Quando um contrato de engenharia é cadastrado como "serviços gerais",
o ente público pode estar (a) cometendo erro de cadastro inofensivo,
ou (b) **fugindo do rito formal** que a lei exige para engenharia
(ART/RRT, projeto básico/executivo, memorial descritivo, engenheiro
responsável, normas ABNT). O caso (b) é o **subenquadramento real**
e é o foco do TCC.

## Pipeline em 3 etapas

### Etapa 0 — Triagem determinística (`pncp.triagem`)

Sua reflexão metodológica chave: separar o "óbvio" antes do ML.

**0a. Pré-filtro lexical.** Lista de padrões regex de alta precisão que
indicam engenharia sem ambiguidade (`construção de ponte`, `pavimentação
asfáltica`, `reforma estrutural`, `projeto executivo`…). Um contrato
'geral' que casa com qualquer padrão é marcado `obvio_engenharia`.

**0b. Verificação de rito.** Para cada `obvio_engenharia`, conta sinais
do rito de engenharia presentes nos PDFs (Camada 2):

| Sinal | Onde aparece |
|---|---|
| ART (Anotação de Responsabilidade Técnica) | Memorial, edital |
| RRT (Registro de Responsabilidade Técnica) | Memorial, edital |
| Memorial descritivo | TR/Edital |
| Projeto executivo / básico | TR/Edital |
| Engenheiro responsável citado | TR/Edital |
| Norma ABNT NBR | Especificação técnica |
| As-built | Termo de recebimento |

Limiar: **≥2 sinais = rito seguido**.

**Veredito da etapa 0:**

| Classificação | Significado | Ação |
|---|---|---|
| `rotulacao_incorreta_processo_ok` | Óbvio eng + rito seguido | Nota técnica para o órgão (corrigir cadastro) — **sem violação legal** |
| `subenquadramento_real` | Óbvio eng + rito **não** seguido | **Violação Lei 14.133/2021** — encaminhar a controle |
| `ambiguo` | Não casa com pré-filtro | Vai para o ML (etapa 1) |
| `fora_escopo` | Já rotulado obras/eng | Não analisado |

### Etapa 1 — ML para os ambíguos (`pncp.classificacao`)

Treina TF-IDF + Regressão Logística + Random Forest + LinearSVC nos
contratos rotulados (engenharia, obras, geral-não-óbvio). Aplica nos
'geral' ambíguos. Saídas:

- Holdout estratificado (test_size=0.2) com F1 por classe
- McNemar (LR vs RF) — diferença significativa?
- Bootstrap 1000× — IC 95% do F1-engenharia
- Ranking de suspeitos por `prob_engenharia`

Opcionalmente: BERTimbau / Sentence-BERT (`pncp.embeddings`),
geralmente +5 a +10 p.p. no F1-engenharia mas custa ~10× mais tempo.

### Etapa 2 — Camadas complementares

Cada uma adiciona um **sinal binário** ao veredito final. O contrato
fica mais "forte" como suspeito conforme acumula sinais:

| Camada | Módulo | Sinal |
|---|---|---|
| Outliers contextuais | `pncp.outliers` | IsolationForest no cluster 'geral' |
| Termos aditivos | `pncp.aditivos` | Aditivo com mudança de objeto/escopo |
| Grafos | `pncp.grafos` | Concentração suspeita órgão↔fornecedor |
| CNAE / CONFEA | `pncp.cnae` | Fornecedor tem CNAE de engenharia? |

### Etapa 3 — Veredito final (`pncp.relatorio`)

O relatório consolidado combina:
1. **Veredito da triagem** (determinístico, alta confiança)
2. **Probabilidade ML** + **n_sinais** das camadas complementares para
   os ambíguos

Ordena os 'geral' por número de sinais positivos e gera:
- `relatorio.md` — texto do TCC com métricas, gráficos, citações
- `suspeitos_consolidados.parquet` — ranking final
- `amostra_revisao_manual.csv` — top-50 para validação humana

---

## Por que esta ordem importa

Antes da refatoração, o pipeline rodava ML em **todos** os 'geral',
incluindo casos onde o objeto dizia explicitamente "construção de
ponte". Isso:

1. **Polui o conjunto de treino** — o classificador "aprende" que
   "construção de ponte" pode ser 'geral' (porque foi rotulado assim).
2. **Desperdiça PDFs** — gastamos OCR em casos óbvios.
3. **Confunde o relatório jurídico** — mistura erros de cadastro
   (não-problema) com violações reais da lei.

Com a triagem antecipada:
- O ML treina apenas nos casos **realmente ambíguos**, melhorando o F1.
- Os PDFs vão direto para os contratos onde a verificação de rito
  importa.
- O relatório final separa "rotulação errada" de "violação Lei 14.133".

---

## Onde rodar?

### Colab Free
- ✅ Grátis, GPU T4 ocasional, fácil compartilhar
- ❌ 12GB RAM, 12h por sessão, kernel cai com OOM
- **Recomendado**: pipeline rápido + iteração

### Colab Pro (R$ 50/mês)
- ✅ 24-50GB RAM, GPU mais consistente, sessões mais longas
- **Recomendado**: BERTimbau + multi-UF

### Kaggle Notebooks
- ✅ Grátis, **30GB RAM**, 12h por sessão, GPU T4, mais estável que Colab Free
- ❌ Datasets devem ser públicos ou privados a até 5 colaboradores
- **Recomendado**: alternativa grátis com mais RAM

### Local (notebook do TCC)
- ✅ Sem desconexão, sem fila, sem custo extra
- ❌ Sem GPU (BERTimbau fica lento), depende da RAM da sua máquina
- **Recomendado**: 90% do trabalho. 16GB de RAM já roda 300k contratos confortavelmente com o pipeline novo.

### Cloud por hora (GCP / AWS Spot / RunPod)
- ✅ 32-64GB RAM + GPU sob demanda; ~R$ 3-8/hora
- ❌ Setup inicial mais complexo
- **Recomendado**: rodar BERTimbau na versão final do TCC (1-2h, < R$ 20).

### Recomendação prática para o TCC

| Etapa | Onde rodar |
|---|---|
| Coleta + EDA + triagem + classificação clássica | **Local** (sem GPU) |
| Iteração / experimentos | **Local** ou Colab Free |
| Geração final BERTimbau | **Colab Pro** ou **Kaggle** (GPU T4) |
| Apresentação para a banca | **Local** (sem dependência de internet) |

O pipeline novo foi desenhado para funcionar igualmente bem em qualquer
um deles — tudo é parquet em disco, sem estado vivo entre células.

---

## Comparativo prático de plataformas

Após observação real do uso (1M+ contratos SP, coleta interrompida 3×,
classificação ~1h em Colab Free), eis o veredito honesto:

### 🥇 Local com Jupyter / VSCode (RECOMENDADO se possível)

**Pré-requisitos:** 16GB+ RAM, Python 3.10+, ~5GB de disco.

**Vantagens:**
- Sem timeout de 12h, sem desconexão de Drive
- TF-IDF de 1M linhas em <5min (vs 20min no Colab Free)
- Classificação completa em <30min (vs 1-2h no Colab Free)
- Zero gambiarra de keep-alive, snapshot, retomada
- Drive nem precisa — escreve direto em `dados/` local
- Pode rodar coleta em background enquanto trabalha

**Como começar:**
```bash
git clone https://github.com/caiofvserra/pncp-engineering-analytics
cd pncp-engineering-analytics
git checkout claude/identify-engineering-underclassification-nImeQ
pip install -r requirements.txt   # ou pip install pandas pyarrow scikit-learn ...
jupyter lab
```
Abra `notebook/pipeline_pncp.ipynb`, **comente a célula 2** (clone+drive),
substitua por `import pncp` e siga normal. `pncp.config.PASTA_DADOS` aponta
para `./dados` por default.

### 🥈 Kaggle Notebooks (alternativa Colab Free, melhor)

**Vantagens vs Colab Free:**
- **30GB de RAM** (vs 12GB Colab Free) — RAM era seu maior gargalo
- 9 horas de execução contínua (vs 12 que Colab cortava)
- GPU T4 grátis, mais estável
- Não cai com idle de 30min

**Desvantagens:**
- Datasets > 20GB precisam ser uploaded como Kaggle Dataset
- Não monta Google Drive direto (precisa adaptar paths)

**Setup:** crie notebook em kaggle.com/code, suba `dados/` como dataset
privado, ajuste `PASTA_DADOS = "/kaggle/working/dados"`.

### 🥉 Colab Pro (R$ 50/mês)

Vale se você **não tem máquina local com 16GB+** mas precisa rodar
BERTimbau ou análises pesadas. 24-50GB RAM, GPU mais consistente, sem
desconexão por idle.

### Colab Free (sua situação atual)

**Limitações reais que você está sentindo:**
- 12GB RAM aperta com 1M linhas + TF-IDF + RF
- Drive desconecta sob I/O alto (Errno 107)
- Sessão cai em 12h ou se aba minimizar
- GridSearch + CV + Bootstrap em 1M = 1-2h facilmente

**Como sobreviver:**
- Use os defaults novos (subsample, CV3, RF=100) — corta tempo em ~3×
- `forcar=False` por default em cada `executar()` — não re-faz se já existe
- Cache PDF/CNAE em disco (já implementado)
- Snapshots para pontos de verificação
- Coleta robusta a desconexão (parquet por mês, retomada exata)

### Cloud por hora (GCP / AWS / RunPod)

Para o pico do TCC (BERTimbau final), uma VM com 32GB+GPU custa ~R$ 5-10/h.
Roda em 1-2h e desliga. < R$ 30 total.

---

## Tempos reais observados (SP, ~1M contratos)

| Etapa | Colab Free | Local 16GB | Kaggle 30GB |
|---|---|---|---|
| TF-IDF (max_features=30k) | ~10min | <3min | ~5min |
| Classificação completa (com defaults novos) | ~25min | ~10min | ~15min |
| LDA + KMeans + GMM | ~15min | ~5min | ~8min |
| Outliers (4 detectores + ensemble) | ~10min | ~3min | ~5min |
| **Pipeline completo (sem coleta)** | **~70min** | **~25min** | **~35min** |

A coleta é cara em qualquer lugar (limitada pela API PNCP) — ~6h para SP
2024-2026, mas é resistente a interrupção em todos os ambientes.

---

## Referências teóricas — mapeamento técnica × capítulo

Cada técnica do pipeline tem fundamentação direta nos capítulos do
livro Han, Kamber & Pei (Data Mining: Concepts and Techniques) +
Aggarwal & Zhai (Mining Text Data, Cap. 6).

### Cap. 6 (Aggarwal & Zhai) — Text Classification Survey
- **TF-IDF + 1,2-grams** (`pncp.texto.construir_tfidf`)
- **Feature selection por Chi²** (`pncp.texto.selecao_chi2`) — top-K
  termos mais discriminativos antes de descartar o vocabulário restante
- **Logística / SVM linear / Naive Bayes** sobre TF-IDF
  (`pncp.classificacao`)

### Cap. 7 — Classification: Advanced Methods
- **7.1 Feature selection (filter)** → `selecao_chi2`
- **7.3 Linear SVM** → `LinearSVC` em `classificacao`
- **7.4.4 Associative classification** → `pncp.avancado.apriori` gera
  regras que podem ser usadas como classificador (CBA-style)
- **7.5.1 Semi-supervised** → `pncp.avancado.label_propagation`
- **7.5.2 Active Learning (uncertainty sampling)** →
  `pncp.classificacao.amostra_active_learning` — escolhe os 50
  contratos mais informativos (não os mais óbvios) para revisão humana
- **7.5.4 Distant Supervision** → `pncp.triagem` formaliza isto: a
  triagem determinística produz rótulos fracos para os ambíguos
- **7.7.3 Interpretability** → top features do LR + SHAP/LIME
  (recomendação de extensão futura)

### Cap. 9 — Cluster Analysis: Advanced
- **9.1.3 EM / GMM** → `pncp.avancado.gmm` — soft clustering com
  probabilidade de pertinência por cluster (vs. KMeans hard)
- **9.4.2 NMF** → tópicos via NMF (presente no original, mantido como
  alternativa ao LDA)
- **9.5.2 SimRank / Personalized PageRank** → `pncp.grafos.pagerank`
- **9.6 Semi-supervised clustering** → motivação para LP

### Cap. 10 — Deep Learning
- **10.1–10.2 Backprop, dropout, cross-entropy** → fundamentação para
  `pncp.embeddings` (BERTimbau / SBERT)

### Cap. 11 — Outlier Detection
- **11.2.1 Univariate (Z-score, IQR, Grubb's test)** →
  `pncp.outliers.zscore_valor` no campo `valor` do contrato
- **11.3.2 Density-based (LOF)** → `pncp.outliers.lof`
- **11.5.2 Classification-based (One-Class SVM)** →
  `pncp.outliers.one_class_svm`
- **11.6 Contextual outliers** → justifica treinar IsolationForest
  apenas no cluster 'geral' (cap 11.6.1: transformar contextual
  outlier em conventional via condicionamento no contexto)
- **11.7.3 Outlier Detection Ensemble** →
  `pncp.outliers.ensemble` — min-max normalize + média/máximo entre
  IsolationForest + LOF + OCSVM

### Cap. 5 — Pattern Mining: Advanced
- **5.1.5 Negative / rare patterns** → contratos óbvios + rotulo
  errado são padrões raros (motivação da triagem)
- **5.4 Sequential patterns (PrefixSpan)** → extensão futura: padrões
  temporais por órgão (sequência de contratos do mesmo órgão ao longo
  dos anos)

### Cap. 12 — Trends and Frontiers
- **12.1.1 Mining text data** → embasamento geral
- **12.2.2 Truth discovery** → motivação para combinar sinais
  contraditórios (Camada 1 texto, Camada 2 PDF, CNAE) — extensão futura
  via modelo Bayesiano (Cap. 7.2)
- **12.3.1 Structuring unstructured data** → `pncp.pdfs` faz
  exatamente isso: extrai marcadores estruturados de PDFs livres
- **12.3.3 Correlation vs causality** → discussão de limitações do TCC

---

## Resumo executivo das técnicas adicionadas (post-leitura dos PDFs)

| Técnica nova | Módulo | Cap. fonte | Para que serve |
|---|---|---|---|
| Triagem (regex+rito) | `triagem` | 7.5.4 distant sup. | Reclassifica óbvios antes do ML |
| Chi² feature selection | `texto` | 6, 7.1.1 | Reduz vocab para os termos discriminativos |
| Active learning | `classificacao` | 7.5.2 | Escolhe 50 contratos *mais informativos* p/ revisão |
| Calibração Platt/Isotonic | `classificacao` | 6 (Aggarwal) | Probabilidades fiéis ao threshold 0.5 |
| GMM (EM) | `avancado` | 9.1.3 | Soft clustering: contratos "no meio" eng↔geral |
| One-Class SVM | `outliers` | 11.5.2 | 3º detector de anomalias no cluster geral |
| IQR + Z-score (valor) | `outliers` | 11.2.1 | Outlier univariado simples e pedagógico |
| Ensemble de outliers | `outliers` | 11.7.3 | Combina IsolationForest+LOF+OCSVM |
| PageRank no grafo | `grafos` | 9.5.2 | Fornecedores estruturalmente importantes |
