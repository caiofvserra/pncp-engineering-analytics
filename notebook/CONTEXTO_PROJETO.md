# Contexto do Projeto — Identificação de Subenquadramento de Engenharia no PNCP

> **Documento de handoff.** Reúne tudo que é preciso para retomar o projeto em
> outra sessão/chat sem perder contexto: objetivo, decisões, arquitetura do
> notebook, parâmetros exatos, estado atual, arquivos e pendências. Escrito em
> 2026-07-14.

---

## 1. Objetivo e enquadramento

- **TCC** do MBA em IA & Big Data (ICMC/USP). **Autor: Caio Francisco Valente
  Serra** (caio.serra@fmrp.usp.br).
- **Problema**: identificar contratos do PNCP (Portal Nacional de Contratações
  Públicas) rotulados pelo próprio órgão como **"serviços gerais"** que são, na
  verdade, **serviços de engenharia/obras disfarçados** (subenquadramento sob a
  **Lei 14.133/2021**), recorte do **estado de São Paulo**.
- O que define a irregularidade **não é o valor do contrato**, e sim a
  **presença ou ausência de profissional habilitado** — o "rito documental"
  (ART/CREA, projeto/laudo, responsável técnico). É o CREA-SP quem fiscaliza.
- **Saída-alvo**: uma lista de contratos suspeitos com o rito ausente, pronta
  para o CREA verificar um a um no PNCP.
- **Restrições editoriais**: o código será lido pelo **professor** — NÃO deve
  conter comentários sobre "TCC", "banca" ou "relatório". Comentários devem
  marcar o que é `[ESSENCIAL]` vs `[APOIO]`/`[ROBUSTEZ]`/`[DESEMPENHO]`.
- O modelo de IA que assiste este projeto **não deve** colocar seu identificador
  de modelo em commits, PRs, comentários de código ou qualquer artefato do repo.

## 2. Repositório, branch e convenções de commit

- **Repo**: `caiofvserra/pncp-engineering-analytics` (acesso via GitHub MCP,
  restrito a esse repo).
- **Branch de trabalho**: `claude/identify-engineering-underclassification-nImeQ`
  (SEMPRE commitar e pushar aqui; `git push -u origin <branch>` com retry
  exponencial em falha de rede). **Não** abrir PR sem pedido explícito.
- **Artefato central**: `notebook/pesquisa_subenquadramento.ipynb` (71 células,
  50 de código). Guia de execução: `notebook/COMO_RODAR_VSCODE.md`.
- Rodadas de dados vivem no **Google Drive** do usuário (pasta `PNCP_TCC/`),
  acessível por MCP do Drive (ferramentas `search_files`,
  `download_file_content` etc. — **sem** ferramenta de delete).
- **Ambiente**: sessão remota efêmera (Claude Code on the web). Só o que é
  commitado/pushado sobrevive. Chromium headless disponível, mas **sem WebGL**
  (não renderiza mapas mapbox) e o sandbox **não tem internet** (proxy só para
  HTTPS de ferramentas). Scratchpad de trabalho:
  `/tmp/claude-0/.../scratchpad`.

## 3. Pipeline — visão geral (5 fases, mapeadas às células)

O notecook detecta Colab vs local (`EM_COLAB`). Roda com "Executar tudo"; para
na Etapa 8 para rotulação humana e retoma sozinho.

### Fase 1 — Coleta e preparação
- **Base**: contratos do PNCP (SP) em Parquet, com rótulo do órgão. Na última
  rodada: **126.465 contratos** (117.882 gerais, 4.667 engenharia, 3.916 obras).
- **Célula 9**: pré-processamento — `chave_texto` (minúsculas/sem acento/sem
  pontuação), `chave_grupo` (texto **sem dígitos** = chave de quase-duplicata,
  usada para blindar vazamento), `limpar_boilerplate`, `_STOPS_NUVEM` +
  `tokens_nuvem` (nuvem de palavras sem jargão administrativo: compara forma
  **sem acento** contra STOP_TUDO + ~50 termos como contratação, município,
  prestação, preço, prefeitura, empresa, edital...).
- **Célula 11 (EDA)**: distribuição, top termos por rótulo (TF-IDF), nuvens.
  **Sem valor** (removido de todas as saídas — ver §6).
- **Célula 13 (embeddings)**: `intfloat/multilingual-e5-large` (1024d, L2,
  prefixo `query:` via helper `embutir()`, `max_seq_length=192`). Cache por
  modelo em `cache_compartilhado/`. Trocável por
  `PNCP_EMB_MODELO=paraphrase-multilingual-mpnet-base-v2` (GPU fraca).

### Fase 2 — Aprendizado fraco (PU Learning)
- **Célula 14 (filtro PU multi-protótipo)**: `KMeans(n_clusters=8,
  random_state=42, n_init=10)` sobre os **8.583 positivos** (eng+obras) → 8
  protótipos. Candidato = similaridade de cosseno máxima aos protótipos no **top
  30%** (35.494). Negativos fracos = resto dos gerais (82.388).
- **Célula 28 (treino)**: compara **8 classificadores** (logreg, SVM-RBF, kNN,
  MLP, HistGradientBoosting, RandomForest, ExtraTrees, NaiveBayes), todos com
  balanceamento de classe. **Ensemble = média das probabilidades dos 3
  melhores** (F1). Calibração: `isotonic` se ≥800 amostras, senão `sigmoid`
  (Platt) — via `calibrar_prefit()` (compat FrozenEstimator / cv='prefit').
  Pesos amostrais por tipo de engenharia.
- **Célula 31 (pontuação)**: `score_suspeita = prob × (0,5 + 0,5 × percentil da
  similaridade aos protótipos)`. Pontua os 117.882 gerais.

### Fase 3 — Validação científica
- **Célula 34 (validação + teste-ouro)**: amostra de **~200** (fronteira +
  aleatórios) rotulada por engenheiro em `08_validacao.csv`; **teste-ouro de
  100** intocável (`08_teste_ouro.csv`), fora de toda decisão.
- **Célula 37 (8.2)**: re-treino human-in-the-loop (peso **humano 3×**, veredito
  **LLM 2×**, fraco 1×); varredura de limiar **0,10–0,95** passo 0,05 (piso era
  0,30 — estendido porque o ótimo travava na borda); critério **F1** (mudar para
  **F2** com `PNCP_LIMIAR_METRICA=f2` — recall dobrado); `StratifiedGroupKFold(
  n_splits=5, shuffle=True, random_state=42)` agrupado por `chave_grupo`. As
  curvas "antes" são rotuladas **"(otimista*)"** — o modelo pré-re-treino é
  otimista por vazamento; a curva de decisão é a "depois (CV)".
- **Teste-ouro**: precisão/recall/F1/F2 com **IC de Wilson 95%**.
- **Células 40–42 (bancada RQ1–RQ8, Etapa 8b)** — comparação científica:
  - **E0** TF-IDF(1-2g)+logreg (RQ1 piso lexical)
  - **E1** 4 encoders congelados×logreg: distiluse, e5-large, BGE-M3, BERTimbau
  - **E2** e5-large × {kNN, HistGB, MLP} (RQ3 encoder vs RQ4 classificador)
  - **E3** fine-tuning: **BERTimbau-FT** (3 seeds) e **e5+LoRA** (`r=16,
    lora_alpha=32, lora_dropout=0.05`, 2 seeds), `max_length=192`
  - **E4** ensemble média-4; **E5** ablação do filtro PU (negativos aleatórios,
    RQ6); **E6** LLM zero-shot (qwen2.5:32b, RQ7, com custo ms/contrato)
  - **Anexo A** RF/ET/SVM/NB/CatBoost (1 seed); **Anexo B** TabNet/TabPFN/
    FT-Transformer sobre **PCA-256**
  - **Métrica de conclusão**: PR-AUC no **teste-ouro**; **decisões de adoção**
    só na fatia de **validação** (`pr_auc_val`); bootstrap **pareado IC 99%**;
    sensibilidade no estrato aleatório; **quadro-resumo 8d** (Grupo|RQ|Melhor|
    PR-AUC|F1|F2|Tempo|Memória|Conclusão); Pareto numerado.
  - **Adoção automática** do vencedor na rodada seguinte, restrita a E1/E2, com
    **trava de estabilidade** (só troca se ganho > 0,01).
  - **Saneamento** (célula 40): ao herdar `08b_bancada.csv` de rodada anterior
    ao ajuste de balanceamento, descarta os braços contaminados (histgb,
    catboost, ensemble, anexos) para re-medição; preserva logreg, fine-tuning e
    LLM (marca `balanceado=1`, idempotente).

### Fase 4 — Julgamento e rito
- **Célula 6 (contexto da juíza)**: `contexto_llm()` concatena partes
  persistidas em `_ckpt_contexto_llm.json`, na ordem `_ORDEM_CTX`
  (exemplos_engenheiro, falsos_positivos, **licoes_rodada**, politica_eventos,
  armadilhas, vocabulario, tipos_engenharia, perfis, ...). **`CTX_LICOES`**
  (parte `licoes_rodada`, versão **`CTX_VER='ctx-licoes-v1'`**) traz os padrões
  de **falso positivo** aprendidos na auditoria (contas de utilidade
  água/esgoto/energia/telefonia, seguros, gráficos, limpeza predial, banheiros
  químicos, transporte, cursos, shows, SaaS, consultorias...) e os **positivos**
  claros (obra civil, sondagem SPT/topografia, projeto/laudo por CREA/CAU,
  instalações elétricas/hidráulicas com execução).
- **Célula 50 (veredito)**: LLM local **qwen2.5:32b** (Ollama, `format json`,
  `temperature 0.2`, `num_ctx 4096`). Confirmação = classe `eng_obra` e
  `llm_conf ≥ 0.6`. Dedup por texto normalizado com propagação às cópias.
  Orçamento **8h** retomável (`PNCP_VEREDITO_HORAS`). **Re-julgamento
  versionado**: se `CTX_VER` mudou, os vereditos `eng_obra` herdados são
  re-enfileirados (`PNCP_REJULGAR=confirmados|tudo|nada`, default `confirmados`);
  os `nao` antigos permanecem; cada linha grava `ctx_ver`.
- **Célula 54 (rito)**: resolve a **compra** de origem (fallback API de detalhe
  do contrato). Baixa até **`MAX_DOCS_POR_CONTRATO=5`** documentos (PDF, DOCX,
  imagem) com validação de assinatura (`_MAGICS_OK`: %PDF, PK, JPEG, PNG),
  extração multi-formato (`_texto_docx` via zip/XML, `_texto_imagem` via
  pytesseract). **OCR** Tesseract-por, 8 páginas, 200 dpi. **12 marcadores**
  determinísticos de rito (`MARCADORES`, categorias: ART, CREA, ENGENHEIRO_RESP,
  ATESTADO_CAP_TEC, PROJETO_BASICO, OBRA_SERV_ENG, ABNT_NORMA, LEI_14133_ENG,
  PLANILHA_ORCAMENTARIA, CRONOGRAMA_FIS_FIN, CADERNO_ENCARGOS, EXECUCAO_OBRA):
  **≥2 marcadores = rito ok, decidido ANTES da LLM**. Fallback por marcadores
  quando a LLM falha; **dupla checagem cética** antes de `subenquadramento_real`.
  Classes: `subenquadramento_real`, `rito_parcial` (revisão humana),
  `rotulacao_incorreta_processo_ok`, `indeterminado_*`. **Coerência**: registros
  de rito herdados de contratos que deixaram de ser confirmados são descartados.

### Fase 5 — Produtos e operação
- **Célula 57 (consolidação, PRODUTO FINAL)**: `12_subenquadramento_consolidado
  .csv` (ordenado identificação→local→diagnóstico, sem valor) + **`12_LISTA_CREA_
  subenquadramentos.csv/.xlsx`** — 1 linha por subenquadramento real, ordenada
  por município→órgão (GRE), com **links diretos do PNCP** (contrato
  `pncp.gov.br/app/contratos/<cnpj>/<ano>/<seq>`; processo `/app/editais/...`),
  objeto completo, tipo de engenharia, probabilidade, justificativa da juíza,
  marcadores e documentos analisados.
- **Célula 59 (mapa geográfico SP, 11.7)**: **choropleth municipal** (malha
  IBGE) com **contornos + números das GREs do CREA-SP**; botões alternam camadas
  (subenquadramentos × suspeitos × engenharia/obras); escala **log(1+n)** que
  escurece rápido nos primeiros valores; zeros em cinza. Interativo
  `12_mapa_sp.html` + 3 PNGs (`12_mapa_sp.png`, `_suspeitos`, `_eng_obras`).
  **Usa shapely** (não geopandas — que falhava no Colab). Ver §7 sobre GREs.
- **Células 45–48, 55–58**: UMAP 2D/interativo, grafo kNN/PyVis, gráfico de
  rito, funil Sankey (`11b_funil_sankey.html`), mapa do desfecho.
- **Célula 61 (12.1)**: exporta `pacote_reuso.joblib` (modelo, limiar, nome_emb,
  prefixo, contexto_llm, rubrica SYS_VER) + `modelo_final.joblib` +
  `12_df_classificado.csv`; grava `_rodada_concluida.json`.
- **Célula 62 (12.2)**: relatório **Word** (`relatorio_final.docx`, python-docx)
  com todas as figuras e números do relatório vivo.
- **Células 64–68 (Etapa 13 — ferramenta operacional CREA)**: autossuficiente
  (copiar as células para notebook novo + `pacote_reuso.joblib`). Seleção de
  pacote e pasta de destino por **janelas do sistema** (tkinter/Colab upload).
  Período por datas `dd/mm/aaaa`. **Sem GPU** (LLM auto-pulada, força com
  `PNCP_LLM_FORCAR=1`) e **sem Drive**. Gera ranking, rito e
  `mapa_subenquadramentos_<período>.html`.

## 4. Rodadas automáticas e herança (mecânica)

- Cada rodada ganha pasta própria: `rodadas/rodada_NNN` (criada sozinha).
  `_rodada_concluida.json` marca conclusão; a próxima execução abre rodada nova
  (ou retoma a inacabada). Override: `PNCP_RODADA=nova|7`.
- **Herança automática** (`_HERDAVEIS`, copiada com `copy2` para a pasta nova —
  então **tudo herdado também aparece na pasta da rodada**): `08_validacao.csv`,
  `08_teste_ouro.csv`, `10_veredito_llm.csv`, `11_analise_rito.csv`,
  `enriquecimento_compra.csv`, `_ckpt_contexto_llm.json`, `08b_bancada.csv`,
  `08b_bancada_probs.json`, `08b_zero_shot.json`. Pastas do esquema antigo
  (`resultados_pesquisaN`) são reconhecidas como fonte.
- **Caches caros compartilhados** em `cache_compartilhado/`: embeddings por
  modelo e PDFs baixados (renomear pasta não re-gasta GPU nem re-baixa).
- **Modelo/treino NUNCA** atravessam rodadas (recomputam por desenho: filtro PU,
  clusters, treino, re-treino, pontuação, UMAP).

## 5. Variáveis de ambiente

`PNCP_EMB_MODELO`, `PNCP_LLM_MODELO` (default qwen2.5:32b; operacional
qwen2.5:7b), `PNCP_VEREDITO_HORAS` (8; 0=sem teto), `PNCP_LLM_NUM_CTX` (4096),
`PNCP_LIMIAR_METRICA` (f1|f2), `PNCP_RODADA`, `PNCP_MAPA_FUNDO` (satelite|ruas —
legado do mapa antigo), `PNCP_LLM_FORCAR`, `PNCP_PACOTE`, `PNCP_REJULGAR`
(confirmados|tudo|nada), `PNCP_TCC_DIR` (base local).

## 6. Decisão: VALOR do contrato fora das saídas

A pedido do usuário (commit `8cad2e8`): a fiscalização verifica **profissional
habilitado**, não montante. Removido de EDA, ranking (órgãos por Nº de
suspeitos), consolidação (ordena por probabilidade; painel com municípios e
distribuição de probabilidade), mapas (hover sem R$), Word e Etapa 13. A lista
do CREA **não** tem coluna de valor.

## 7. GREs do CREA-SP (mapa geográfico) — pontos críticos

- Arquivo **`GRE.xlsx`** (colunas `GERÊNCIA REGIONAL`, `Município`; 662 linhas,
  16 GREs). Deve ficar na pasta base (`PNCP_TCC/GRE.xlsx`); a célula 59 procura
  em PASTA_BASE, pasta da rodada, `dados/` e cache.
- **Correções obrigatórias** (confirmadas contra o IBGE):
  - **São Paulo → GRE05**: no Excel a capital vem **fatiada** ("São Paulo -
    Leste/Centro/Norte/Oeste/Sul" sob "GRE05 - Capital"), que não casam com o
    município único "São Paulo" (IBGE 3550308). O código une qualquer linha que
    comece com "são paulo" na capital.
  - **Ouro Verde → GRE15** (IBGE 3534807). **NÃO existe "Cabo Verde" em SP** (é
    de Minas Gerais) — o usuário disse "cabo verde" mas quis dizer Ouro Verde.
  - **Florínia (grafia IBGE, 3516101) / Florínea (grafia do Excel) → GRE16**.
  - Casamento **por espaço/hífen** (fallback sem espaço) para pegar
    "Biritiba-Mirim" (IBGE) × "Biritiba Mirim" (Excel). Cobertura 645/645.
- **Falha resolvida**: as GREs não saíam no HTML porque o código usava
  **geopandas**, que falhava ao instalar/importar no Colab (exceção engolida).
  Trocado por **shapely** (`unary_union` para dissolver por GRE) no interativo e
  **matplotlib puro** (PatchCollection) nos PNGs. Commit `f3d988d`.
- **Correção de HTML já gerado** (sem re-rodar): o geojson dos 645 municípios
  fica **embutido** no HTML do plotly; dá para reconstruir as GREs a partir dele
  e **injetar na figura inicial** (splice do `Plotly.newPlot`, mesmo plotly.js).
  Script de referência no scratchpad: `build_splice.py`.

## 8. Resultados da última rodada real (interpretação)

- Precisão teste-ouro **100%** (IC95 87–100%), recall **75,8%**, F1 0,862, F2
  0,796 no LIMIAR 0,30 (que **travava no piso** da varredura — daí a extensão
  para 0,10). Validação CV: precisão 73,1%, recall 66,7%, F1 0,697.
- **6.927 suspeitos** (prob≥limiar) → **4.605 confirmados** pela LLM → **550
  subenquadramentos reais** + 523 rito parcial + 3.277 processo ok + 255
  indeterminados.
- **Probabilidades polarizadas** (prob≥0,5: 4.577 mas prob≥0,9: 4.097) — modelo
  superconfiante; recall ~⅔–¾ é o ponto fraco (casos difíceis escapam).
- Bancada: **BERTimbau-FT venceu** (PR-AUC 0,810, +0,059 vs congelado); LLM
  zero-shot 0,809 mas 803 ms/contrato; BGE-M3 0,797 (adotável); filtro PU
  agregou só +0,002 (RQ6 honesta). **Atenção**: `08b_bancada.csv` da rodada
  entregue é herdado de 12/07, **antes** do ajuste de balanceamento — as
  conclusões RQ3/RQ4 (histgb 0,667) estão contaminadas; a próxima rodada
  re-mede os braços saneados automaticamente.

## 8b. Notebook FINAL enxuto (`subenquadramento_pncp.ipynb`)

Derivado do notebook completo para a **rodada única e decisiva** (2026-07):
- **Removido**: bancada científica inteira (Etapa 8b/RQ1–RQ8, células 39–42),
  adoção automática do vencedor, config `BENCH`, sistema de **rodadas
  automáticas + herança** (pasta única `PNCP_TCC/resultados`), re-julgamento
  por versão de contexto herdado, flag `_rodada_concluida.json`, caches
  legados de rodadas.
- **Adicionado**: célula **8.4 baseline lexical** (TF-IDF 1–2g + logística
  balanceada, mesmos rótulos de treino, limiar próprio na validação, métricas
  no teste-ouro, gráfico `08c_baseline.png`); seção correspondente no Word.
- **Corrigido**: `_ctxv` agora definido fora do `if` (no completo, primeira
  execução sem CSV de veredito daria NameError).
- **Mantido**: TODAS as visualizações, EDA, checkpoints/retomada intra-rodada,
  caches de embeddings/PDFs, veredito/rito/consolidação/lista CREA/mapas/Word
  e **Etapa 13 completa**. Sem QUALQUER menção a TCC/banca/autores no código.
- Ordem das etapas revisada e mantida (dados → representação → rótulos fracos
  → modelo → validação+baseline → julgamento → rito → produto).
- O `pesquisa_subenquadramento.ipynb` (completo) permanece intacto no repo.

## 9. Estado atual e pendências

- **Notebook pronto para a rodada final do TCC.** Checklist antes de rodar:
  1. `GRE.xlsx` na pasta base (o mapa depende dele).
  2. Decidir F1 (default) vs `PNCP_LIMIAR_METRICA=f2` **antes** de rodar.
  3. A rodada herda rótulos/teste-ouro/vereditos/rito/bancada; re-julga só os
     `eng_obra` confirmados (contexto novo `ctx-licoes-v1`, ~1–2h de LLM); a
     bancada re-mede só os braços contaminados.
  4. Esperar limiar possivelmente <0,30, RQ3/RQ4 honestas, saídas sem valor,
     lista do CREA no final.
- **Limpeza opcional do Drive**: apagar mapas de opções descartadas
  (`mapa_municipios_*.html`, `mapa_gre*.{html,png}`, `mapa_bolhas_limites.html`,
  `relatorio_vivo_backup.md`) — sem ferramenta de delete no MCP; usar `os.remove`
  no notebook.
- **Entregáveis de apresentação** já gerados no scratchpad (não versionados):
  `metodologia_v4.pptx` (slide de método, 5 fases/17 blocos), 4 figuras de apoio
  (`fig_05_filtro_pu.png`, `fig_08_protocolo_validacao.png`,
  `fig_11_matriz_bancada.png`, `fig_12_cascata_julgamento.png`).

## 10. Histórico de correções relevantes (por que as coisas são como são)

- **Rito refém da LLM**: regra determinística de marcadores aplicada ANTES da
  LLM + retry + fallback; normalização de resposta (`str().strip().lower()`,
  "Não"→"nao").
- **Cache de PDFs sujo** (HTML de erro gravado): valida `%PDF`/assinatura antes
  de gravar; re-baixa se inválido; ilegíveis re-enfileiram quando há OCR.
- **kaleido exige Chrome** → PNGs estáticos por matplotlib.
- **Eleição usava o ouro (leakage sutil)** → passou a `pr_auc_val` (só
  validação); ouro virgem; trava de estabilidade.
- **Nuvem com stopwords** ("contratação"≠"contratacao") → `tokens_nuvem` compara
  sem acento.
- **Etapa 13 100% sem_compra** (dados reais) → variantes do nome do campo da API
  + `_resolver_compra_op()` via API de detalhe.
- **Avisos**: `cv='prefit'` deprecado → `calibrar_prefit()`; glyph 🏆 fora de
  título de figura; `logging.set_verbosity_error()` para tabela MISSING/
  UNEXPECTED do transformers (é esperada).

## 11. Padrão de trabalho no notebook (para edições futuras)

- Editar via script Python: `json.load` do .ipynb → `sub(célula, old, new)` com
  `assert` de contagem → `json.dump` (falha = nada salvo). **Relocalizar sempre
  por conteúdo** (edits anteriores deslocam linhas); rodar do repo root.
- **Verificar SEMPRE** com AST em todas as células (tolerância à célula 3, que
  tem `!pip`) e, quando possível, **executar a célula real** com dados
  sintéticos no scratchpad antes de commitar.
- Toda célula deve produzir **alguma saída** (o usuário exige feedback visível).
- Cache de arte/simulações: usar `PIP_NO_INDEX=1 PIP_RETRIES=0` em simulações
  para evitar timeout; `HOME` gravável para LibreOffice; matplotlib `Agg`.

---

_Arquivos-chave do repo_: `notebook/pesquisa_subenquadramento.ipynb` (pipeline),
`notebook/COMO_RODAR_VSCODE.md` (execução), este `CONTEXTO_PROJETO.md` (handoff).
_Branch_: `claude/identify-engineering-underclassification-nImeQ`.
