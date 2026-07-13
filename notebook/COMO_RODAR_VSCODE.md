# Rodar o notebook no VSCode (local)

O mesmo `pesquisa_subenquadramento.ipynb` roda no Colab e localmente — ele
detecta o ambiente sozinho (`EM_COLAB`). No local, siga os passos abaixo.

## 1. Pré-requisitos
- **Python 3.10+** e **VSCode** com as extensões **Python** e **Jupyter**.
- **GPU NVIDIA + CUDA** é fortemente recomendada (SBERT em 126 mil contratos e a
  LLM local). Sem GPU, o SBERT roda em CPU (lento, porém funciona) e a LLM deve
  ser um modelo pequeno (ver passo 5).
- **Ollama** instalado: https://ollama.com/download (app para Windows/Mac/Linux).

## 2. Ambiente Python
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -U pip
pip install sentence-transformers scikit-learn umap-learn nltk networkx \
            pymupdf pyarrow pandas matplotlib seaborn plotly tabulate pyvis \
            ipykernel requests joblib
```
> A célula de dependências do notebook (`!pip install`) também funciona no
> VSCode, mas instalar antes no venv evita surpresas.

## 3. Estrutura de pastas (equivalente ao Drive)
Crie uma pasta do projeto e aponte o notebook para ela com a variável de
ambiente **`PNCP_TCC_DIR`** (se não definir, ele usa `./PNCP_TCC`):
```
<PNCP_TCC_DIR>/
  dados/coleta/contratos.parquet      <- baixe do seu Google Drive e coloque aqui
  rodadas/rodada_NNN/                 <- criadas automaticamente (uma por rodada)
  cache_compartilhado/                <- embeddings + PDFs (valem entre rodadas)
```
Defina a variável (exemplos):
```bash
# Mac/Linux (no terminal antes de abrir o VSCode):
export PNCP_TCC_DIR="/caminho/para/PNCP_TCC"
# Windows PowerShell:
setx PNCP_TCC_DIR "C:\caminho\para\PNCP_TCC"
```
Alternativa sem variável: deixe a pasta `PNCP_TCC` ao lado do notebook.

## 4. Selecionar o kernel no VSCode
Abra o `.ipynb` → canto superior direito **"Select Kernel"** → escolha o
Python do `.venv` que você criou.

## 4b. Modelo de embeddings
- Padrão: **`intfloat/multilingual-e5-large`** (~2,2 GB no 1º download; melhor
  qualidade para similaridade em português). O prefixo `query:` exigido pelo
  e5 é aplicado automaticamente pelo helper `embutir()`.
- GPU fraca ou CPU: use um modelo menor antes de abrir o notebook:
  ```bash
  export PNCP_EMB_MODELO="paraphrase-multilingual-mpnet-base-v2"
  ```
- Na 1ª rodada após a troca, embeddings e treino são recomputados (caches
  `_v4`; o cache de embeddings é separado por modelo). Os rótulos humanos
  (`08_validacao.csv`), os vereditos da LLM e a análise de rito são
  preservados e realimentam o novo treino automaticamente.

## 5. LLM (Ollama) local
- Deixe o **app do Ollama aberto** (ou rode `ollama serve` num terminal). A
  célula da Etapa 0 detecta o Ollama já no ar e não tenta instalar nada.
- **Modelo**: o padrão é `qwen2.5:32b` (~20 GB, precisa de GPU grande). Sem GPU
  potente, use um menor definindo a variável **`PNCP_LLM_MODELO`** antes de
  abrir o VSCode:
  ```bash
  export PNCP_LLM_MODELO="qwen2.5:7b"     # ou llama3.1  (~5 GB)
  ollama pull qwen2.5:7b
  ```
- **Sinal de alerta**: se o veredito (Etapa 10) rodar a dezenas de segundos por
  contrato, o modelo não coube na GPU e está na RAM — a própria célula avisa
  (checagem `ollama /api/ps`); troque para um modelo menor.
- **Orçamento do veredito**: a Etapa 10 julga a fila em sessões de até
  `PNCP_VEREDITO_HORAS` horas (padrão 8; `0` = sem teto), sempre dos suspeitos
  mais críticos para os menos; re-executar continua de onde parou.
- **Memória da LLM**: `PNCP_LLM_NUM_CTX` (padrão 4096) limita o contexto e o
  consumo de memória por chamada.
- **OCR (Etapa 11)**: PDFs escaneados são lidos por OCR. No Colab a instalação
  é automática; localmente instale o Tesseract com o idioma português
  (`apt install tesseract-ocr tesseract-ocr-por` no Linux; no macOS
  `brew install tesseract tesseract-lang`) e `pip install pytesseract pillow`.
- **Classes do rito**: `subenquadramento_real` (rito ausente, confirmado por
  dupla checagem da LLM), `rito_parcial` (evidência incompleta — fila de
  revisão), `rotulacao_incorreta_processo_ok` e `indeterminado_*`.

## 4c. Rodadas e reaproveitamento (automático)
- **Nada de renomear pasta**: cada rodada ganha a própria pasta
  (`rodadas/rodada_001`, `002`, …) criada sozinha. Se a última rodada ainda
  não terminou (queda de sessão, parada de rotulação), o "Executar tudo"
  **retoma ela**; rodada concluída (Etapa 12 grava `_rodada_concluida.json`)
  → a próxima execução abre uma rodada nova.
- Controle manual: `PNCP_RODADA=nova` força rodada nova; `PNCP_RODADA=7`
  reabre a `rodada_007`. Pastas do esquema antigo (`resultados_pesquisaN`)
  são reconhecidas como fonte de herança.
- **Herança automática** da rodada mais recente: rótulos humanos (validação
  e teste-ouro), vereditos da LLM, análise de rito, contexto aprendido e
  bancada — nada disso é refeito. Modelo/treino NUNCA atravessam rodadas.
- **Caches caros compartilhados** (`cache_compartilhado/`): embeddings por
  modelo e PDFs baixados valem entre rodadas — renomear a pasta não re-gasta
  GPU nem re-baixa documentos (caches em pastas antigas são lidos onde estão).
- O que recomputa por rodada (por desenho): filtro PU, clusters, treino,
  re-treino, pontuação e UMAP (~30–60 min de GPU no total).
- **Critério do limiar**: `PNCP_LIMIAR_METRICA=f2` troca a escolha do limiar
  de F1 para F2 (recall pesa dobrado — coerente com a triagem em cascata);
  a tabela sempre mostra F1 e F2 e os dois ótimos.

## 5a. Bancada de experimentos (Etapa 8b)
- **Ligada por padrão** (todos os braços: congelados, TF-IDF, tabular-DL,
  fine-tuning e LLM zero-shot). Enquanto a validação/teste-ouro não estiverem
  rotulados, a bancada é **adiada com aviso** — o "Executar tudo" não para.
- É retomável em três níveis: cada execução vira uma linha em
  `08b_bancada.csv`; os embeddings de cada encoder ficam em cache próprio; e
  o braço LLM zero-shot salva checkpoint parcial a cada 25 respostas.
- Para uma rodada rápida só do pipeline, desligue em `BENCH` (Etapa 0).
- TabNet, TabPFN e FT-Transformer instalam sob demanda; se a biblioteca não
  estiver disponível, o braço é pulado com aviso (nada quebra).
- Resultados em `08b_bancada.csv` (retomável linha a linha) + bootstrap
  pareado na célula 8b.3.

## 5a-bis. Saídas visuais e relatório
- **Nuvens de palavras**: `01c_nuvens.png` (eng/obras × gerais) e
  `12_nuvem_subenq.png` (vocabulário dos subenquadramentos).
- **Mapa geográfico de SP** (11.7): choropleth por município com contornos
  e números das GREs do CREA-SP (coloque `GRE.xlsx` na pasta base); botões
  alternam as camadas (subenquadramentos × suspeitos × engenharia/obras) —
  `12_mapa_sp.html` (interativo, hover) + 3 PNGs estáticos (um por camada).
- **Sem valor de contrato nas saídas**: a fiscalização verifica a presença
  de profissional habilitado (rito), não o montante — rankings, painéis e
  mapas ordenam por probabilidade/contagem.
- **Relatório Word** (12.2): `relatorio_final.docx` com todas as figuras em
  ordem, explicações e os números do relatório vivo.
- **Anexos não-PDF**: o rito agora extrai DOCX (zip/XML) e imagens
  (JPG/PNG via OCR) além de PDF — o despacho é pelo conteúdo do arquivo.

## 5b. Ferramenta operacional (Etapa 13)
- A Etapa 13 é **autossuficiente**: copie as células 13.1–13.5 para um
  notebook vazio e distribua com o `pacote_reuso.joblib` (gerado na Etapa 12).
- O operador roda célula a célula: **seleciona o pacote no computador**
  (janela de upload no Colab; janela do sistema no local), digita as datas
  (`dd/mm/aaaa`) e a UF, opcionalmente liga a LLM (padrão leve `qwen2.5:7b`)
  e **escolhe a pasta de destino numa janela** (no Colab, o navegador baixa
  um `.zip`). A ferramenta não usa o Drive.
- Sem LLM, o rito é decidido por marcadores: ≥ 2 = processo ok; 0–1 = fila
  de revisão humana (nunca acusa subenquadramento sem juiz).
- **Funciona SEM GPU**: em CPU a classificação usa lotes menores (só mais
  lenta) e o veredito LLM é pulado automaticamente (force com
  `PNCP_LLM_FORCAR=1`); o rito e o mapa não precisam de GPU.
- A 13.5 também gera `mapa_subenquadramentos_<período>.html` (bolhas por
  município, interativo) — útil para o CREA visualizar as concentrações.

## 6. Rodar
"Run All". As etapas caras gravam cache em `resultados_pesquisa/_ckpt_*`; a
Etapa 8 **para** para você rotular `08_validacao.csv` (abra no Excel/planilha,
preencha `rotulo_verdade`, salve, e rode tudo de novo — ele retoma).

## Diferenças Colab × local (automáticas)
| Item | Colab | Local (VSCode) |
|---|---|---|
| Pasta base | Google Drive montado | `PNCP_TCC_DIR` (ou `./PNCP_TCC`) |
| Ollama | instala via script | usa o app já instalado |
| GPU | A100 (recomendada) | sua GPU/CPU |
| Modelo LLM | qwen2.5:32b | `PNCP_LLM_MODELO` (menor se sem GPU) |
