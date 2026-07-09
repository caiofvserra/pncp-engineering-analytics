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
  resultados_pesquisa/                <- criada automaticamente (saídas + caches)
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
