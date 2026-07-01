# stage: (num, title, desc, ins[dashed=entradas externas/dados consumidos], outs[artefatos], conn_to_next)
phases = [
 ("A · COLETA E PREPARAÇÃO","#2c5f8a","#eef4fa","#dbe7f3",[
    ("1","Coleta de dados","Baixa contratos do PNCP e filtra as categorias de interesse.",
        ["API PNCP /v1/contratos"],["contratos.parquet"],"contratos.parquet"),
    ("2","Pré-processamento + EDA","Limpa boilerplate, tokeniza (RSLP) e resume a base.",
        [],["df limpo (objeto)","01_distribuicao.png"],None),
 ], "objetos limpos"),
 ("B · MODELAGEM PU","#c8702a","#fdf3ec","#fde3cc",[
    ("3","Embeddings SBERT + filtro PU","Vetoriza os objetos e mede a similaridade dos “geral” ao centróide de eng+obras.",
        [],["X_emb","rotulo_treino","02_filtro_pu.png"],"X_emb dos candidatos"),
    ("4","Clusterização auto-k","Agrupa os candidatos (KMeans k=6–12) e mede a pureza de cada cluster.",
        [],["clusters + pureza","03_clusters.png"],"amostras por grupo"),
    ("5","Perfis (LLM)","LLM descreve o padrão léxico de engenharia vs não-engenharia.",
        [],["05_perfis.json"],"X_emb + rótulos certeiros"),
    ("6","Treino + calibração","Treina 8 classificadores, escolhe o melhor (F1 macro) e calibra as probabilidades.",
        [],["modelo_final.joblib","06_confusao.png"],None),
 ], "modelo calibrado"),
 ("C · DETECÇÃO E VALIDAÇÃO","#5a8a2c","#f0f5e7","#e7f0d9",[
    ("7","Pontuação + score de suspeita","Pontua todos os “geral” e combina a probabilidade com a pureza do cluster.",
        ["pureza dos clusters (etapa 4)"],["07_ranking_suspeitos.csv"],"ranking"),
    ("8","Validação manual","Amostra rotulada à mão define precisão/recall e o limiar ótimo.",
        ["08_validacao.csv (rótulos humanos)"],["métricas","08_curva_limiar.png"],"limiar + classe_final"),
    ("9","Visualização","Projeção UMAP 2D, rede k-NN e curva de limiar.",
        [],["09_umap","09_grafo_knn"],"top suspeitos"),
    ("10","Veredito LLM","LLM revisa os suspeitos do topo (reforma e instalação predial contam como engenharia).",
        [],["10_veredito_llm.csv"],None),
 ], "suspeitos"),
 ("D · RITO E ENTREGA","#b03030","#f9ecec","#f6dcdc",[
    ("11a","Enriquecimento da compra","Resolve o vínculo contrato→compra dos suspeitos, com cache.",
        ["API PNCP (detalhe)"],["enriquecimento_compra.csv"],"vínculo da compra"),
    ("11b","Análise de rito","Baixa TR / Projeto Básico / Edital e verifica CREA/ART, projeto básico e ABNT.",
        ["PDFs da licitação"],["11_analise_rito.csv","11_rito.png"],"vereditos"),
    ("12","Exportação + reuso","Salva modelo, CSVs e relatório; classifica novos contratos do último mês.",
        [],["modelo_final.joblib","relatorio_vivo.md"],None),
 ], None),
]
def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

cols=[]
for title,cs,cbg,csoft,cards,handoff in phases:
    blocks=[]
    for j,(num,t,desc,ins,outs,conn) in enumerate(cards):
        inrow=""
        if ins:
            inch="".join(f'<span class="cin">{esc(x)}</span>' for x in ins)
            inrow=f'<div class="inrow">{inch}</div>'
        outch="".join(f'<span class="pill" style="border-color:{cs};background:{cbg}">{esc(o)}</span>' for o in outs)
        blocks.append(f'''<div class="card" style="border-left:6px solid {cs}">
          {inrow}
          <div class="chead"><span class="badge" style="background:{cs}">{num}</span>
            <span class="ctitle">{esc(t)}</span></div>
          <div class="cdesc">{esc(desc)}</div>
          <div class="outrow">{outch}</div>
        </div>''')
        if j < len(cards)-1:
            lbl=f'<span class="vlabel">{esc(conn)}</span>' if conn else ''
            blocks.append(f'<div class="vconn">{lbl}<span class="varr">&#8595;</span></div>')
    cols.append(f'''<div class="phase"><div class="phead" style="background:{cs}">FASE {esc(title)}</div>
      <div class="pbody" style="background:{cbg};border:1px solid {csoft}">{"".join(blocks)}</div></div>''')
    if handoff is not None:
        cols.append(f'''<div class="hconn"><div class="arrowline"></div>
          <div class="hlabel">{esc(handoff)}</div><div class="chev">&#8250;</div></div>''')

html=f'''<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",-apple-system,"Helvetica Neue",Arial,sans-serif;background:#fff;color:#1f2933;padding:44px 40px 46px;width:2380px}}
h1{{font-size:30px;font-weight:800;color:#12303f}}
.sub{{font-size:15px;color:#5b6b78;margin:8px 0 14px}} .sub b{{color:#12303f}}
.leg{{display:flex;gap:26px;align-items:center;font-size:13.5px;color:#54636e;margin-bottom:26px;flex-wrap:wrap}}
.leg .k{{display:inline-flex;align-items:center;gap:8px}}
.sw{{width:34px;height:20px;border-radius:6px;display:inline-block}}
.sw.in{{background:#f7f9fb;border:1.6px dashed #9aa7b1}}
.sw.out{{background:#fdf3ec;border:1.6px solid #c8702a}}
.sw.ar{{width:26px;height:0;border-top:3px solid #c3ccd4;position:relative}}
.board{{display:flex;align-items:stretch}}
.phase{{display:flex;flex-direction:column;width:520px}}
.phead{{color:#fff;font-weight:800;font-size:15px;letter-spacing:.5px;padding:11px 16px;border-radius:12px 12px 0 0;text-align:center}}
.pbody{{flex:1;border-radius:0 0 14px 14px;padding:18px 16px 20px;display:flex;flex-direction:column}}
.card{{background:#fff;border-radius:12px;padding:13px 16px;box-shadow:0 3px 10px rgba(20,40,60,.09);border:1px solid #eef1f4}}
.inrow{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:9px}}
.cin{{font-size:12px;color:#5b6b78;background:#f7f9fb;border:1.4px dashed #a8b4bd;padding:3px 9px;border-radius:999px}}
.chead{{display:flex;align-items:center;gap:10px;margin-bottom:6px}}
.badge{{color:#fff;font-weight:800;font-size:14px;min-width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;padding:0 6px}}
.ctitle{{font-size:16.5px;font-weight:700;color:#16232e;line-height:1.15}}
.cdesc{{font-size:13.6px;color:#46545f;line-height:1.4;margin:2px 2px 11px}}
.outrow{{display:flex;flex-wrap:wrap;gap:6px}}
.pill{{font-family:"Consolas","SF Mono",monospace;font-size:11.6px;color:#2b3a45;border:1.5px solid #ccc;padding:3px 9px;border-radius:8px;white-space:nowrap}}
.vconn{{display:flex;flex-direction:column;align-items:center;margin:6px 0}}
.vlabel{{background:#fff;border:1px solid #d5dde3;color:#4a5a66;font-size:12px;font-style:italic;padding:2px 10px;border-radius:999px}}
.varr{{color:#9aa7b1;font-size:20px;line-height:1}}
.hconn{{display:flex;flex-direction:column;align-items:center;justify-content:center;width:120px;position:relative}}
.arrowline{{position:absolute;top:50%;left:6px;right:6px;height:3px;background:#c3ccd4;border-radius:2px}}
.hlabel{{position:relative;background:#fff;border:1px solid #d5dde3;color:#3f4f5b;font-size:12.5px;font-weight:600;padding:3px 10px;border-radius:999px;margin-bottom:6px;z-index:2}}
.chev{{position:relative;color:#8a97a1;font-size:40px;line-height:.6;z-index:2;background:#fff;padding:0 2px}}
.foot{{margin-top:24px;font-size:13px;color:#7a8893}} .foot b{{color:#3a4a55}}
</style></head><body>
<h1>Metodologia — Identificação de subenquadramento de engenharia no PNCP</h1>
<div class="sub"><b>PU Learning:</b> os rótulos <b>engenharia</b> e <b>obras</b> são positivos confiáveis; só <b>serviços gerais</b> é ruidoso.</div>
<div class="leg">
  <span class="k"><span class="sw in"></span> entrada (dado/arquivo consumido pela etapa)</span>
  <span class="k"><span class="sw out"></span> saída (artefato gerado)</span>
  <span class="k"><span class="sw ar"></span> rótulo na seta = dado que segue adiante</span>
</div>
<div class="board">{"".join(cols)}</div>
<div class="foot"><b>Cores por fase:</b> azul = coleta · laranja = modelagem PU · verde = detecção/validação · vermelho = rito e entrega &nbsp;|&nbsp; conformidade: Lei 14.133/2021 · Lei 5.194/66 · CONFEA (apenas CREA/ART).</div>
</body></html>'''
open('/tmp/metodologia.html','w',encoding='utf-8').write(html)
print("ok", len(html))
