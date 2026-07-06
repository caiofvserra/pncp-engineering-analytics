# Monitor de Subenquadramento de Engenharia — PNCP

Sistema web de **monitoramento contínuo** de contratos "serviços gerais" que
podem ser engenharia/obras (subenquadramento). O funcionário revisa cada caso
suspeito, diz se **concorda** ou **discorda**, e essa decisão **retroalimenta o
modelo** (aprendizado com humano no loop), com frequência de atualização
configurável na própria interface.

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
