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

## Referências teóricas (capítulos de Han, Kamber & Pei)

- Cap. 6 (Aggarwal & Zhai): Survey de Classificação de Texto — embasa
  TF-IDF, Naive Bayes, SVM, KNN aplicados a texto.
- Cap. 7: Classificação avançada (SVM, Bayesian networks, regras) —
  embasa a escolha de Logística + RF + LinearSVC.
- Cap. 9: Cluster analysis avançado — embasa LDA, KMeans, hierárquico.
- Cap. 10: Deep Learning — embasa BERTimbau / Sentence-BERT.
- Cap. 11: **Outlier Detection** — embasa o módulo `pncp.outliers`
  (IsolationForest e LOF para detectar suspeitos contextuais).
- Cap. 5 + 12: Pattern Mining e Trends — embasam Apriori e a
  análise de subenquadramento como caso de "fraud detection" textual.
