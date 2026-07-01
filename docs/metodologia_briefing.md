# Briefing para gerar o slide de Metodologia (TCC)

> Este documento é autossuficiente: entregue-o a uma IA geradora de imagens/slides
> para produzir **um único slide** de metodologia. Contém o contexto da pesquisa,
> a descrição detalhada de cada etapa (o que faz, por quê, o que entra e o que
> sai) e as instruções de design ao final.

---

## 1. Contexto da pesquisa (para a IA entender o tema)

**Problema.** No Brasil, as contratações públicas são publicadas no **PNCP**
(Portal Nacional de Contratações Públicas). Cada contrato recebe uma
**categoria** informada pelo próprio órgão: *obras*, *serviços de engenharia* ou
*serviços gerais*, entre outras. Ocorre que muitos contratos de **engenharia/obras**
são cadastrados como **"serviços gerais"** — fenômeno chamado **subenquadramento**.
Isso é irregular à luz da **Lei 14.133/2021** (Nova Lei de Licitações), porque
obras e serviços de engenharia exigem um **rito próprio**: ART/CREA (Anotação de
Responsabilidade Técnica / Conselho Regional de Engenharia), projeto básico,
responsável técnico, normas ABNT, planilha orçamentária etc. Ao rotular como
"serviços gerais", o órgão **escapa desse rito**.

**Objetivo do TCC.** Construir um pipeline de dados + IA que **encontre, na
massa de "serviços gerais", os contratos que são de fato engenharia/obras** e,
para os mais suspeitos, **verifique documentalmente** se o rito de engenharia
foi cumprido — separando *subenquadramento real* de *mero erro de rótulo*.

**Escopo.** Apenas **engenharia** (CREA/ART). **Arquitetura (CAU/RRT) está fora**
do escopo.

**Insight metodológico central — PU Learning (Positive-Unlabeled).** Os rótulos
*engenharia* e *obras* são **confiáveis** (quando o órgão marca assim, é porque
é). Só o rótulo *serviços gerais* é **ruidoso**: parte é serviço comum de
verdade, parte é engenharia disfarçada. Então usamos os contratos de
engenharia/obras como **âncora de positivos confiáveis** e tratamos "serviços
gerais" como **não-rotulados** — daí o nome Positive-Unlabeled.

**Dados.** ~30 mil+ contratos coletados da API pública do PNCP; o texto usado é
o **objeto da contratação** (a descrição do que está sendo contratado).

**Fundamentação legal.** Lei 14.133/2021 (licitações); Lei 5.194/1966 (regula o
exercício da engenharia); resoluções do **CONFEA** (definem o que é atividade de
engenharia).

---

## 2. Fases e etapas (descrição detalhada — o que / por quê / entra / sai)

O pipeline tem **4 fases** e **12 etapas**. Descrição de cada uma:

### FASE A — Coleta e preparação
**1. Coleta de dados**
- *O que faz:* baixa os contratos publicados no PNCP via API oficial e filtra as
  categorias de interesse (obras, serviços de engenharia, serviços gerais).
- *Por quê:* montar a base de trabalho com o rótulo original do órgão.
- *Entra:* API pública do PNCP. *Sai:* base de contratos rotulada por categoria.

**2. Pré-processamento**
- *O que faz:* padroniza o texto do objeto e **remove os termos burocráticos
  repetidos** (ex.: "contratação de empresa especializada para prestação de
  serviços de…") que aparecem em quase todo contrato e não ajudam a distinguir.
- *Por quê:* sem isso, todos os contratos parecem parecidos e a análise semântica
  perde discriminação.
- *Entra:* base bruta. *Sai:* objetos limpos e padronizados.

### FASE B — Modelagem (PU Learning)
**3. Representação semântica + filtro PU**
- *O que faz:* transforma cada objeto em um **vetor de significado** (embedding
  SBERT, modelo de linguagem multilíngue) e mede a **proximidade** de cada
  "serviços gerais" ao **núcleo semântico** dos contratos de engenharia/obras.
  Os "gerais" próximos viram **candidatos suspeitos**; os distantes são tratados
  como **não-engenharia confiável**.
- *Por quê:* é o coração do PU — usar os positivos confiáveis para separar, no
  meio ruidoso, o que merece investigação.
- *Entra:* objetos limpos. *Sai:* vetores + separação candidatos × não-eng.

**4. Agrupamento (clusterização)**
- *O que faz:* agrupa os candidatos por semelhança (KMeans, k escolhido por
  qualidade de coesão) e mede, em cada grupo, a **densidade de engenharia
  confirmada** (quantos eng/obras caem ali).
- *Por quê:* grupos com muita engenharia confirmada indicam que os "gerais"
  vizinhos provavelmente também são engenharia. Serve de interpretação e de
  **peso** na priorização.
- *Entra:* vetores dos candidatos. *Sai:* grupos temáticos + índice de pureza.

**5. Vocabulário de domínio (apoio de IA)**
- *O que faz:* extrai da própria base o **vocabulário típico** de engenharia
  versus não-engenharia (termos que mais distinguem as duas classes) e uma IA
  descreve os perfis. Esse vocabulário vira **contexto** para as etapas com IA
  mais adiante (10 e 11).
- *Por quê:* aterrar a IA no vocabulário real do domínio, em vez de listas
  subjetivas escritas à mão.
- *Entra:* casos confiáveis. *Sai:* vocabulário/perfis de domínio.

**6. Treino + calibração do classificador**
- *O que faz:* treina classificadores (regressão logística, florestas, SVM, kNN,
  redes neurais etc.) usando **só os casos confiáveis** (positivos = eng/obras;
  negativos = "gerais" distantes), escolhe o melhor por F1 e **calibra** as
  probabilidades.
- *Por quê:* obter um modelo que estime, para qualquer contrato, a **chance de
  ser engenharia**; a calibração faz o limiar de decisão ter sentido.
- *Entra:* vetores dos casos confiáveis. *Sai:* modelo treinado e calibrado.

### FASE C — Detecção e validação
**7. Pontuação + ranqueamento**
- *O que faz:* aplica o modelo a **todos** os "serviços gerais" e combina a
  probabilidade com a densidade do grupo (etapa 4), gerando um **ranking de
  suspeitos** de subenquadramento.
- *Por quê:* é o produto central — a lista priorizada do que investigar.
- *Entra:* todos os "gerais" + modelo. *Sai:* ranking de suspeitos.

**8. Validação manual**
- *O que faz:* uma **amostra aleatória** de contratos é rotulada à mão pelo
  pesquisador (engenheiro) e comparada ao modelo, produzindo **precisão, recall
  e F1 reais**, além do **ponto de corte** ideal.
- *Por quê:* é a única medida honesta de desempenho — dá rigor acadêmico ao TCC.
- *Entra:* amostra rotulada à mão. *Sai:* métricas + limiar de decisão.

**9. Visualização**
- *O que faz:* projeta os contratos em 2D (UMAP) e como **rede de similaridade
  (grafo k-NN)**, colorindo por classe.
- *Por quê:* evidência visual de que os suspeitos ficam **colados** aos contratos
  de engenharia confirmada — reforça o argumento.
- *Entra:* vetores + classe. *Sai:* mapas e rede de similaridade.

**10. Revisão por IA**
- *O que faz:* uma IA (LLM) revisa os suspeitos do topo do ranking usando o
  **contexto de domínio** (etapa 5), confirmando ou descartando cada um.
- *Por quê:* reduz falsos positivos antes da etapa cara de verificação
  documental.
- *Entra:* suspeitos do topo. *Sai:* suspeitos filtrados/priorizados.

### FASE D — Verificação e entrega
**11. Análise do rito de engenharia (evidência definitiva)**
- *O que faz:* para os suspeitos priorizados, **baixa o edital / Termo de
  Referência / Projeto Básico** da licitação e verifica se o **rito de
  engenharia** foi seguido — procura ART/CREA, projeto básico, responsável
  técnico, normas ABNT, planilha orçamentária, BDI, cronograma físico-financeiro
  etc. Classifica cada caso como **subenquadramento real** (é engenharia e o rito
  NÃO foi seguido) ou **rótulo equivocado com processo correto** (é engenharia
  mas o rito foi seguido).
- *Por quê:* é a **prova documental** — transforma "suspeita estatística" em
  evidência jurídica.
- *Entra:* documentos da licitação dos suspeitos. *Sai:* veredito por contrato.

**12. Consolidação e reuso**
- *O que faz:* reúne modelo, ranking e evidências num relatório; permite
  **classificar automaticamente novos contratos** publicados no PNCP.
- *Por quê:* deixa uma ferramenta de triagem reaproveitável (ex.: para órgãos de
  controle).
- *Entra:* modelo + novos contratos. *Sai:* ferramenta/relatório final.

---

## 3. Fio condutor (a "história" do slide)

Base bruta → limpeza → **separar o suspeito do comum** (PU) → **treinar o modelo**
→ **ranquear todos os "gerais"** → **validar com rigor** → **confirmar com IA** →
**provar no documento** (rito) → **entregar a ferramenta**. Ou seja: da massa de
dados até a evidência de subenquadramento.

---

## 4. Instruções de design (para a IA que vai gerar o slide)

- **Formato:** UM slide único, proporção 16:9, horizontal.
- **Estrutura:** 4 blocos/colunas (as 4 fases), com as etapas numeradas dentro de
  cada bloco como itens curtos; setas horizontais entre os blocos com o rótulo do
  que passa adiante (**objetos limpos → modelo treinado → suspeitos
  priorizados**).
- **Cores por fase (sugestão):** A = azul, B = laranja, C = verde, D = vermelho.
- **Hierarquia:** título no topo; uma linha de "ideia central" (PU Learning) como
  subtítulo; blocos no meio; fundamentação legal no rodapé (Lei 14.133/2021 · Lei
  5.194/66 · CONFEA — apenas CREA/ART).
- **Tom:** acadêmico, limpo, legível a distância (é para banca). Evitar poluição;
  ícones simples ajudam (banco de dados, vetor/rede, gráfico, lupa/documento).
- **Não usar** nomes de arquivo ou jargão de código; descrever conceitos.
- **Destaques a enfatizar:** (a) o insight PU (positivos confiáveis × ruidoso);
  (b) a etapa 8 (validação manual = rigor); (c) a etapa 11 (rito = prova
  definitiva).
