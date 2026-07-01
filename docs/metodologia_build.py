phases = [
 ("A · COLETA E PREPARAÇÃO", "#2c5f8a", "#eef4fa", "#dbe7f3", [
    ("1","Coleta de dados","Baixa contratos do PNCP e filtra as categorias de interesse.",["contratos.parquet"]),
    ("2","Pré-processamento + EDA","Limpa boilerplate, tokeniza (RSLP) e resume a base.",["df limpo","01_distribuicao.png"]),
 ], "objetos limpos"),
 ("B · MODELAGEM PU","#c8702a","#fdf3ec","#fde3cc",[
    ("3","Embeddings SBERT + filtro PU","Vetoriza os objetos e mede a similaridade dos “geral” ao centróide de eng+obras.",["X_emb","rotulo_treino","02_filtro_pu.png"]),
    ("4","Clusterização auto-k","Agrupa os candidatos (KMeans k=6–12 por silhouette) e mede a pureza de cada cluster.",["clusters + pureza","03_clusters.png"]),
    ("5","Perfis (LLM)","LLM descreve o padrão léxico de engenharia vs não-engenharia.",["05_perfis.json"]),
    ("6","Treino + calibração","Treina 8 classificadores, escolhe o melhor (F1 macro) e calibra as probabilidades.",["modelo_final.joblib","06_confusao.png"]),
 ], "modelo + X_emb"),
 ("C · DETECÇÃO E VALIDAÇÃO","#5a8a2c","#f0f5e7","#e7f0d9",[
    ("7","Pontuação + score de suspeita","Pontua todos os “geral” e combina a probabilidade com a pureza do cluster.",["07_ranking_suspeitos.csv"]),
    ("8","Validação manual","Amostra rotulada à mão define precisão/recall e o limiar ótimo.",["métricas","08_curva_limiar.png"]),
    ("9","Visualização","Projeção UMAP 2D, rede de similaridade k-NN e curva de limiar.",["09_umap","09_grafo_knn"]),
    ("10","Veredito LLM","LLM revisa os suspeitos do topo (reforma e instalação predial contam como engenharia).",["10_veredito_llm.csv"]),
 ], "suspeitos"),
 ("D · RITO E ENTREGA","#b03030","#f9ecec","#f6dcdc",[
    ("11a","Enriquecimento da compra","Resolve o vínculo contrato→compra dos suspeitos, com cache.",["enriquecimento_compra.csv"]),
    ("11b","Análise de rito","Baixa TR / Projeto Básico / Edital e verifica CREA/ART, projeto básico e ABNT.",["11_analise_rito.csv","11_rito.png"]),
    ("12","Exportação + reuso","Salva modelo, CSVs e relatório; classifica novos contratos do último mês.",["modelo_final.joblib","relatorio_vivo.md"]),
 ], None),
]

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

cards_cols=[]
for title,c_strong,c_bg,c_soft,cards,handoff in phases:
    cblocks=[]
    for j,(num,t,desc,outs) in enumerate(cards):
        pills="".join(f'<span class="pill">{esc(o)}</span>' for o in outs)
        card=f'''<div class="card" style="border-left:6px solid {c_strong}">
          <div class="chead"><span class="badge" style="background:{c_strong}">{num}</span>
            <span class="ctitle">{esc(t)}</span></div>
          <div class="cdesc">{esc(desc)}</div>
          <div class="pills">{pills}</div>
        </div>'''
        cblocks.append(card)
        if j < len(cards)-1:
            cblocks.append('<div class="vconn">&#8595;</div>')
    col=f'''<div class="phase">
      <div class="phead" style="background:{c_strong}">FASE {esc(title)}</div>
      <div class="pbody" style="background:{c_bg};border:1px solid {c_soft}">{"".join(cblocks)}</div>
    </div>'''
    cards_cols.append(col)
    if handoff is not None:
        cards_cols.append(f'''<div class="hconn"><div class="arrowline"></div>
          <div class="hlabel">{esc(handoff)}</div><div class="chev">&#8250;</div></div>''')

html=f'''<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",-apple-system,"Helvetica Neue",Arial,sans-serif;
  background:#ffffff;color:#1f2933;padding:44px 40px 48px;width:2320px}}
h1{{font-size:30px;font-weight:800;color:#12303f;letter-spacing:.2px}}
.sub{{font-size:15.5px;color:#5b6b78;margin:8px 0 30px}}
.sub b{{color:#12303f}}
.board{{display:flex;align-items:stretch;gap:0}}
.phase{{display:flex;flex-direction:column;width:520px}}
.phead{{color:#fff;font-weight:800;font-size:15px;letter-spacing:.6px;
  padding:11px 16px;border-radius:12px 12px 0 0;text-align:center}}
.pbody{{flex:1;border-radius:0 0 14px 14px;padding:18px 16px 20px;
  display:flex;flex-direction:column}}
.card{{background:#fff;border-radius:12px;padding:14px 16px 13px;
  box-shadow:0 3px 10px rgba(20,40,60,.09);border:1px solid #eef1f4}}
.chead{{display:flex;align-items:center;gap:10px;margin-bottom:7px}}
.badge{{color:#fff;font-weight:800;font-size:14px;min-width:30px;height:30px;
  border-radius:50%;display:flex;align-items:center;justify-content:center;
  padding:0 6px}}
.ctitle{{font-size:16.5px;font-weight:700;color:#16232e;line-height:1.15}}
.cdesc{{font-size:13.8px;color:#46545f;line-height:1.4;margin:2px 2px 11px}}
.pills{{display:flex;flex-wrap:wrap;gap:6px}}
.pill{{font-family:"Consolas","SF Mono",monospace;font-size:11.7px;
  background:#f1f4f7;color:#2b3a45;border:1px solid #e2e8ee;
  padding:3px 9px;border-radius:999px;white-space:nowrap}}
.vconn{{text-align:center;color:#9aa7b1;font-size:22px;line-height:1;
  margin:7px 0}}
.hconn{{display:flex;flex-direction:column;align-items:center;justify-content:center;
  width:120px;position:relative}}
.arrowline{{position:absolute;top:50%;left:6px;right:6px;height:3px;
  background:#c3ccd4;border-radius:2px}}
.hlabel{{position:relative;background:#fff;border:1px solid #d5dde3;
  color:#4a5a66;font-size:12.5px;font-style:italic;padding:3px 10px;
  border-radius:999px;margin-bottom:6px;z-index:2}}
.chev{{position:relative;color:#8a97a1;font-size:40px;line-height:.6;z-index:2;
  background:#fff;padding:0 2px}}
.foot{{margin-top:26px;font-size:13px;color:#7a8893}}
.foot b{{color:#3a4a55}}
</style></head><body>
<h1>Metodologia — Identificação de subenquadramento de engenharia no PNCP</h1>
<div class="sub">Insight central: <b>PU Learning</b> — os rótulos <b>engenharia</b> e <b>obras</b> são positivos confiáveis; só o rótulo <b>serviços gerais</b> é ruidoso. As etiquetas em cada cartão são os <b>artefatos gerados</b>; os rótulos nas setas indicam o dado que passa adiante.</div>
<div class="board">{"".join(cards_cols)}</div>
<div class="foot"><b>Legenda:</b> azul = coleta &nbsp;·&nbsp; laranja = modelagem PU &nbsp;·&nbsp; verde = detecção/validação &nbsp;·&nbsp; vermelho = rito e entrega &nbsp;|&nbsp; conformidade: Lei 14.133/2021 · Lei 5.194/66 · CONFEA (apenas CREA/ART).</div>
</body></html>'''
open('/tmp/metodologia.html','w',encoding='utf-8').write(html)
print("html bytes:", len(html))
