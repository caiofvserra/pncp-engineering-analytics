# (num, título, frase conceitual) ; conn = rótulo do dado que passa à próxima
phases = [
 ("A · COLETA E PREPARAÇÃO","#2c5f8a","#eef4fa","#dbe7f3",[
   ("1","Coleta de dados","Baixa contratos do PNCP e separa obras, engenharia e serviços gerais pela categoria informada pelo órgão."),
   ("2","Pré-processamento","Padroniza os objetos e remove os termos burocráticos repetidos que não distinguem o serviço."),
 ],"objetos padronizados"),
 ("B · MODELAGEM (PU Learning)","#c8702a","#fdf3ec","#fde3cc",[
   ("3","Representação semântica + filtro PU","Transforma cada objeto em vetor de significado e aproxima os “serviços gerais” do núcleo de engenharia, isolando os candidatos suspeitos."),
   ("4","Agrupamento","Agrupa os candidatos por semelhança e mede a densidade de engenharia confirmada em cada grupo."),
   ("5","Perfis de domínio (IA)","A IA descreve o vocabulário típico de engenharia e de não-engenharia, formando o contexto do domínio."),
   ("6","Treino + calibração","Treina classificadores com os casos confiáveis e calibra a chance de um contrato ser engenharia."),
 ],"modelo treinado"),
 ("C · DETECÇÃO E VALIDAÇÃO","#5a8a2c","#f0f5e7","#e7f0d9",[
   ("7","Pontuação + ranqueamento","Pontua todos os “serviços gerais” e combina a probabilidade com a densidade do grupo, gerando o ranking de suspeitos."),
   ("8","Validação manual","Uma amostra aleatória rotulada à mão mede a precisão real do modelo e define o ponto de corte."),
   ("9","Visualização","Projeta os contratos em 2D e em rede de similaridade: os suspeitos aparecem colados à engenharia confirmada."),
   ("10","Revisão por IA","A IA revisa os suspeitos do topo usando o contexto do domínio, descartando falsos positivos."),
 ],"suspeitos priorizados"),
 ("D · VERIFICAÇÃO E ENTREGA","#b03030","#f9ecec","#f6dcdc",[
   ("11","Análise do rito de engenharia","Abre o edital / Termo de Referência da licitação e verifica se o rito foi seguido (ART/CREA, projeto básico, normas ABNT): distingue subenquadramento real de rótulo apenas equivocado."),
   ("12","Consolidação e reuso","Reúne modelo, ranking e evidências num relatório e permite triar automaticamente novos contratos."),
 ],None),
]
def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
cols=[]
for title,cs,cbg,csoft,cards,handoff in phases:
    blocks=[]
    for j,(num,t,desc) in enumerate(cards):
        blocks.append(f'''<div class="card" style="border-left:6px solid {cs}">
          <div class="chead"><span class="badge" style="background:{cs}">{num}</span>
            <span class="ctitle">{esc(t)}</span></div>
          <div class="cdesc">{esc(desc)}</div></div>''')
        if j<len(cards)-1:
            blocks.append('<div class="vconn"><span class="varr">&#8595;</span></div>')
    cols.append(f'''<div class="phase"><div class="phead" style="background:{cs}">FASE {esc(title)}</div>
      <div class="pbody" style="background:{cbg};border:1px solid {csoft}">{"".join(blocks)}</div></div>''')
    if handoff is not None:
        cols.append(f'''<div class="hconn"><div class="arrowline"></div>
          <div class="hlabel">{esc(handoff)}</div><div class="chev">&#8250;</div></div>''')
html=f'''<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",-apple-system,"Helvetica Neue",Arial,sans-serif;background:#fff;color:#1f2933;padding:46px 42px 40px;width:2360px}}
h1{{font-size:32px;font-weight:800;color:#12303f}}
.sub{{font-size:16px;color:#5b6b78;margin:10px 0 30px}} .sub b{{color:#12303f}}
.board{{display:flex;align-items:stretch}}
.phase{{display:flex;flex-direction:column;width:524px}}
.phead{{color:#fff;font-weight:800;font-size:16px;letter-spacing:.4px;padding:12px 16px;border-radius:12px 12px 0 0;text-align:center}}
.pbody{{flex:1;border-radius:0 0 14px 14px;padding:20px 18px 22px;display:flex;flex-direction:column;justify-content:flex-start}}
.card{{background:#fff;border-radius:12px;padding:15px 17px;box-shadow:0 3px 10px rgba(20,40,60,.10);border:1px solid #eef1f4;margin:0}}
.chead{{display:flex;align-items:center;gap:11px;margin-bottom:8px}}
.badge{{color:#fff;font-weight:800;font-size:15px;min-width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center}}
.ctitle{{font-size:17.5px;font-weight:700;color:#16232e;line-height:1.15}}
.cdesc{{font-size:14.5px;color:#40505b;line-height:1.46}}
.vconn{{text-align:center;margin:8px 0}} .varr{{color:#9aa7b1;font-size:22px}}
.hconn{{display:flex;flex-direction:column;align-items:center;justify-content:center;width:118px;position:relative}}
.arrowline{{position:absolute;top:50%;left:6px;right:6px;height:3px;background:#c3ccd4;border-radius:2px}}
.hlabel{{position:relative;background:#fff;border:1px solid #d5dde3;color:#3f4f5b;font-size:12.5px;font-weight:600;font-style:italic;padding:3px 11px;border-radius:999px;margin-bottom:6px;z-index:2;text-align:center}}
.chev{{position:relative;color:#8a97a1;font-size:42px;line-height:.6;z-index:2;background:#fff;padding:0 2px}}
.foot{{margin-top:28px;font-size:13.5px;color:#7a8893}} .foot b{{color:#3a4a55}}
</style></head><body>
<h1>Metodologia — Identificação de subenquadramento de engenharia no PNCP</h1>
<div class="sub"><b>Ideia central (PU Learning):</b> os contratos rotulados como <b>engenharia</b> e <b>obras</b> são exemplos confiáveis; só o rótulo <b>serviços gerais</b> é ruidoso — e é nele que se escondem obras/engenharia mal classificadas. As setas indicam o que passa de uma fase à seguinte.</div>
<div class="board">{"".join(cols)}</div>
<div class="foot"><b>Fundamentação:</b> Lei 14.133/2021 · Lei 5.194/66 · resoluções do CONFEA (apenas engenharia — CREA/ART; arquitetura fora do escopo).</div>
</body></html>'''
open('/tmp/metodologia.html','w',encoding='utf-8').write(html)
print('ok',len(html))
