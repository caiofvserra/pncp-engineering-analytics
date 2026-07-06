# Monitor de Subenquadramento de Engenharia — PNCP

Sistema web que **espelha o pipeline do notebook** em abas operacionais e faz o
monitoramento contínuo de contratos "serviços gerais" suspeitos de serem
engenharia/obras (subenquadramento).

**Fluxo (mesma ordem do notebook):**
1. **Ranking** — saída do modelo treinado no notebook (probabilidade por objeto).
2. **Triagem do objeto** — o revisor decide, pelo objeto, se é engenharia. A
   decisão retroalimenta o modelo (aprendizado com humano no loop).
3. **Análise de rito** (posterior, só para os confirmados) — o sistema **baixa o
   edital/Termo de Referência** da licitação no PNCP, extrai o texto (PyMuPDF) e
   detecta os marcadores do rito (ART/CREA, projeto básico, ABNT, planilha
   orçamentária, BDI…). O revisor dá o veredito final: rito seguido (rótulo
   incorreto) × **subenquadramento real** (rito não seguido).

A triagem e o rito são etapas **separadas** justamente porque o rito é caro
(baixar e ler PDFs) e só faz sentido para os casos já confirmados.

## Por que este desenho (barato, eficiente, sem quebras)

- **Aprendizado**, não "RL clássico": cada feedback é um rótulo. O modelo online
  é um `SGDClassifier(log_loss)` sobre `HashingVectorizer` (sem vocabulário para
  manter), atualizado por **`partial_fit`** — aprendizado *incremental* de custo
  O(lote), **sem GPU e sem SBERT** no servidor. O SBERT pesado fica no notebook
  (offline); aqui o modelo apenas **corrige** a probabilidade base com o feedback.
- **Barato**: um único processo FastAPI + **SQLite** (arquivo, sem servidor de
  banco) + frontend estático (sem etapa de build). Roda num laptop ou numa VM
  mínima / free-tier.
- **Sem quebras**: o feedback é **gravado antes** de qualquer coisa; o re-treino
  faz **troca atômica** do arquivo do modelo (se falhar, mantém o anterior); a UI
  funciona mesmo sem modelo (usa o score salvo). Sem o ranking do notebook, sobe
  em **modo demonstração** com dados de exemplo.

## Como o modelo é atualizado

1. O funcionário responde na **Fila de revisão** (Concordo = é subenquadramento;
   Discordo = serviço comum), opcionalmente marcando se o rito foi seguido e
   escrevendo a justificativa.
2. Ao atingir a política de frequência (**a cada N feedbacks** ou **por
   intervalo de tempo**, definido em *Configurações*), o modelo é re-treinado
   incrementalmente com os feedbacks novos (peso configurável).
3. A fila é **repontuada** e reordenada com o modelo atualizado.

## Rodar

```bash
cd sistema
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
# abra http://127.0.0.1:8000
```

### Conectar aos resultados do notebook (opcional)
Aponte para o ranking gerado pela pesquisa para semear a fila com dados reais:
```bash
export PNCP_RANKING_CSV="/caminho/resultados_pesquisa/07_ranking_suspeitos.csv"
```
Sem essa variável, o sistema usa dados de demonstração (a UI funciona igual).
Dados do sistema (banco, modelo) ficam em `PNCP_SISTEMA_DIR` (default
`./dados_sistema`).

## Arquitetura

```
backend/
  main.py        API FastAPI + serve o frontend
  db.py          SQLite (contratos, feedback, config, retrain_log)
  model.py       modelo online incremental (Hashing + SGD, troca atômica)
  learning.py    aplica feedbacks ao modelo (política de frequência)
  ingest.py      semeadura (ranking/demo) + ingestão do PNCP
  scheduler.py   APScheduler: re-treino por tempo + ingestão automática
  config.py      caminhos e padrões
frontend/        index.html + styles.css + app.js  (HTML/JS puro, sem build)
```

## Endpoints principais
`GET /api/fila` · `POST /api/feedback` · `GET /api/stats` · `GET|POST /api/config`
· `POST /api/retrain` · `POST /api/ingest` · `GET /api/historico`

## Identidade visual
Azul 800 `#0e1732` (institucional), Amarelo `#ffb001` (destaque/CTA), Azul 700
`#15265c`, Azul 500 `#3662e2` (interativo), Azul claro `#6077b6`, Preto.

> **Nota**: monitoramento é apoio à decisão. Um caso "confirmado" pela IA + revisor
> ainda exige a verificação documental do rito (etapa do notebook) antes de
> qualquer encaminhamento formal a órgão de controle.
