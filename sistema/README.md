# Monitor de Subenquadramento de Engenharia — PNCP

Sistema **autossuficiente** (não depende do notebook) para uso institucional:
importa uma base de contratos já baixada, treina o próprio classificador,
identifica “serviços gerais” que podem ser engenharia/obras, apoia a decisão
com uma LLM (opcional), coleta a revisão humana, verifica o **rito documental**
e **monitora o PNCP continuamente** (por padrão, 1× por mês).

> O notebook (`notebook/`) é a **pesquisa inicial** do TCC. Este sistema é o
> **produto para operação** — faz tudo internamente.

## Etapas (todas dentro do sistema)
1. **Importação** da base já baixada (parquet/csv com `objeto` e `categoria`).
2. **Classificação** — o sistema treina um classificador próprio (TF-IDF +
   Regressão Logística calibrada): positivos = contratos que o órgão já rotula
   como engenharia/obras; negativos = amostra de “serviços gerais”. Sem GPU.
3. **Veredito da LLM** (opcional) — dá uma segunda opinião no objeto.
4. **Ranking** dos “serviços gerais” por probabilidade.
5. **Triagem humana** — o revisor confirma/descarta; cada decisão **re-treina**
   o classificador (aprendizado com humano no loop, frequência configurável).
6. **Análise de rito** (posterior, só para os confirmados) — baixa o edital/TR
   da licitação no PNCP, extrai o texto (PyMuPDF), detecta os marcadores
   (ART/CREA, projeto básico, ABNT, planilha, BDI…) e, se a LLM estiver ativa,
   lê o documento. O revisor dá o veredito: **rito seguido** (rótulo incorreto)
   × **subenquadramento real** (rito ausente).
7. **Monitoramento contínuo** — a cada N dias (30 = mensal) busca novos
   contratos no PNCP, classifica e adiciona os suspeitos à triagem.

## Rodar
```bash
cd sistema
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload                      # http://127.0.0.1:8000
```
Sem arquivo de importação, sobe em **demonstração** (dados de exemplo).

### Conectar aos dados já baixados
```bash
export PNCP_IMPORT_FILE="/caminho/contratos.parquet"   # colunas objeto, categoria[, orgao, valor, uf]
```

### LLM (opcional)
Em *Configurações → Apoio por LLM*: ative e informe o servidor Ollama
(`http://127.0.0.1:11434`) e o modelo (ex.: `qwen2.5:7b`). Desligada, o sistema
funciona só com o classificador + revisor.

## Arquitetura
```
backend/
  main.py        API FastAPI + serve o frontend
  db.py          SQLite (contratos, triagem, rito, config, eventos)
  classifier.py  classificador próprio (TF-IDF + LogReg calibrada; treino atômico)
  llm.py         apoio por LLM (Ollama), opcional e tolerante a falhas
  pipeline.py    importação, classificação e ingestão contínua do PNCP
  rito.py        resolve compra, baixa PDFs, marcadores + leitura por LLM
  learning.py    re-treino a partir da triagem humana
  scheduler.py   APScheduler: ingestão mensal + re-treino por tempo
  config.py      caminhos e padrões
frontend/        Painel · Ranking · Triagem · Rito · Modelo & IA · Config · Histórico
```

## Desenho (barato, sem quebras)
- **Sem GPU**: classificador TF-IDF/LogReg (CPU, segundos). O SBERT/notebook não
  é necessário em produção.
- **Sem servidor de banco**: SQLite (arquivo).
- **Robusto**: treino com troca atômica (falhou → mantém o anterior); rito e LLM
  degradam com elegância; a UI funciona mesmo sem modelo/LLM/rede.
