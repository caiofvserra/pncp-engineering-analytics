"use strict";
const api = (p, o) => fetch(p, o).then(r => r.json());
const $ = s => document.querySelector(s), $$ = s => document.querySelectorAll(s);
const brl = v => v == null ? "—" : "R$ " + Number(v).toLocaleString("pt-BR", { maximumFractionDigits: 0 });
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const ROTULO_STATUS = {
  novo: ["Aguardando triagem", "st-novo"],
  aguarda_rito: ["Confirmado → rito", "st-rito"],
  triagem_nao: ["Serviço comum", "st-nao"],
  subenq_real: ["Subenquadramento real", "st-real"],
  rito_seguido: ["Rótulo incorreto, processo ok", "st-ok"],
  rito_indeterminado: ["Indeterminado", "st-ind"],
};

let tT;
function toast(m) { const t = $("#toast"); t.textContent = m; t.classList.add("on"); clearTimeout(tT); tT = setTimeout(() => t.classList.remove("on"), 3200); }

function ir(v) {
  $$("#nav button").forEach(b => b.setAttribute("aria-current", b.dataset.v === v));
  $$(".view").forEach(x => x.classList.toggle("on", x.id === v));
  location.hash = v;
  ({ painel: painel, ranking: ranking, triagem: triagem, rito: ritoView, config: config, historico: historico }[v] || (() => {}))();
}
$$("#nav button").forEach(b => b.onclick = () => ir(b.dataset.v));
window.addEventListener("hashchange", () => { const v = location.hash.slice(1); if (v) ir(v); });

// ── Painel ───────────────────────────────────────────────────────────────
async function painel() {
  const s = await api("/api/stats");
  $("#kpis").innerHTML = [
    ["Aguardando triagem", s.novos, true],
    ["Na fila de rito", s.aguarda_rito],
    ["Subenquadramento real", s.subenq_real],
    ["Valor (subenq. real)", brl(s.valor_subenq)],
    ["Triagens p/ próximo aprendizado", s.triagens_novas],
  ].map(([k, v, d]) => `<div class="kpi ${d ? "dark" : ""}"><div class="k">${k}</div><div class="v tnum">${v}</div></div>`).join("");

  const mx = Math.max(1, ...s.por_tipo.map(t => t.n));
  $("#por-tipo").innerHTML = s.por_tipo.length ? s.por_tipo.map(t =>
    `<div class="barra"><span class="lab" title="${esc(t.tipo)}">${esc(t.tipo)}</span>
      <span class="track"><span class="fill" style="width:${100 * t.n / mx}%"></span></span>
      <span class="n tnum">${t.n}</span></div>`).join("") : `<p class="muted">Fila de triagem vazia.</p>`;

  const fun = [["Triados como serviço comum", s.triagem_nao, "var(--verde)"],
  ["Confirmados como engenharia (→ rito)", s.aguarda_rito, "var(--azul500)"],
  ["Rito seguido (rótulo incorreto)", s.rito_seguido, "var(--amarelo)"],
  ["Subenquadramento real", s.subenq_real, "var(--vermelho)"]];
  const mf = Math.max(1, ...fun.map(x => x[1]));
  $("#funil").innerHTML = fun.map(([l, n, c]) =>
    `<div class="barra"><span class="lab">${l}</span>
      <span class="track"><span class="fill" style="width:${100 * n / mf}%;background:${c}"></span></span>
      <span class="n tnum">${n}</span></div>`).join("");
}

// ── Ranking ────────────────────────────────────────────────────────────
async function ranking() {
  const { itens } = await api("/api/ranking?limit=100");
  $("#tb-ranking").innerHTML = itens.map((c, i) => {
    const [rot, cls] = ROTULO_STATUS[c.status] || [c.status, ""];
    return `<tr><td class="tnum">${i + 1}</td><td>${esc(c.objeto)}</td>
      <td>${esc(c.orgao || "—")}</td><td>${esc(c.tipo_eng || "—")}</td>
      <td class="tnum">${brl(c.valor)}</td>
      <td class="tnum"><b>${Math.round(c.score * 100)}%</b></td>
      <td><span class="pill ${cls}">${rot}</span></td></tr>`;
  }).join("") || `<tr><td colspan="7" class="muted" style="padding:24px">Sem contratos.</td></tr>`;
}

// ── Triagem ────────────────────────────────────────────────────────────
async function triagem() {
  const { itens } = await api("/api/triagem/fila?limit=40");
  const el = $("#lista-triagem");
  if (!itens.length) { el.innerHTML = `<div class="vazio">Nenhum contrato aguardando triagem.</div>`; return; }
  el.innerHTML = itens.map(c => `
    <article class="item" id="tr-${esc(c.id)}">
      <div class="topo"><div>
        <div class="obj">${esc(c.objeto)}</div>
        <div class="metas">
          <span class="tag">🏛 ${esc(c.orgao || "—")}</span>
          <span class="tag tnum">💰 ${brl(c.valor)}</span>
          ${c.tipo_eng ? `<span class="tag tp">🔧 ${esc(c.tipo_eng)}</span>` : ""}
          <span class="tag">${esc(c.id)}</span></div>
      </div><div class="medidor"><div class="pct tnum">${Math.round(c.score * 100)}%</div>
        <div class="cap">suspeita</div></div></div>
      <div class="acoes">
        <div class="pergunta">Pelo objeto, este contrato é <b>engenharia/obras</b> registrado como serviço comum?</div>
        <textarea id="j-${esc(c.id)}" placeholder="Justificativa (opcional)"></textarea>
        <div class="linha">
          <button class="btn sim" onclick="decidir('${esc(c.id)}',true)">Sim, é engenharia → enviar ao rito</button>
          <button class="btn nao" onclick="decidir('${esc(c.id)}',false)">Não, é serviço comum</button>
        </div></div>
    </article>`).join("");
}
async function decidir(id, eh) {
  const r = await api("/api/triagem", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contrato_id: id, eh_eng: eh, justificativa: $("#j-" + CSS.escape(id))?.value || "" }),
  });
  const card = $("#tr-" + CSS.escape(id));
  if (card) { card.style.opacity = "0"; card.style.transform = "translateX(28px)"; setTimeout(triagem, 280); }
  let msg = eh ? "Confirmado como engenharia — enviado à análise de rito." : "Registrado como serviço comum.";
  if (r.retreino?.ok && r.retreino.aplicados) msg += " " + r.retreino.msg;
  toast(msg);
}
window.decidir = decidir;

// ── Análise de rito ─────────────────────────────────────────────────────
async function ritoView() {
  const { itens } = await api("/api/rito/fila?limit=40");
  const el = $("#lista-rito");
  if (!itens.length) { el.innerHTML = `<div class="vazio">Nenhum contrato na fila de rito. Confirme suspeitos na aba <b>Triagem</b>.</div>`; return; }
  el.innerHTML = itens.map(c => cardRito(c)).join("");
}
function cardRito(c) {
  const r = c.rito;
  let evid = `<button class="btn amar" onclick="analisarRito('${esc(c.id)}')">📥 Baixar documento e analisar o rito</button>
    <p class="muted" style="margin-top:8px">Busca o edital/TR da licitação no PNCP e detecta os marcadores.</p>`;
  if (r) {
    const marc = JSON.parse(r.marcadores || "[]");
    const forte = r.mk_score >= 2;
    evid = `
      <div class="evid">
        <div class="evid-head">
          <span class="pill ${forte ? "st-ok" : "st-real"}">${r.mk_score} marcador(es) de rito encontrados</span>
          <span class="muted">compra ${esc(r.ncp_compra || "—")} · ${r.n_docs} doc(s) · ${r.chars} caracteres</span>
        </div>
        <div class="marcadores">${marc.length ? marc.map(m => `<span class="mk on">✓ ${esc(m)}</span>`).join("")
        : `<span class="mk off">nenhum marcador de engenharia encontrado</span>`}</div>
        <details><summary>Trecho do documento</summary><pre class="trecho">${esc(r.trecho || "")}</pre></details>
        <div class="pergunta" style="margin-top:14px">O <b>rito de engenharia</b> foi seguido neste processo?</div>
        <div class="linha">
          <button class="btn nao" onclick="vereditoRito('${esc(c.id)}',true)">Sim — rito seguido (rótulo incorreto)</button>
          <button class="btn sim" onclick="vereditoRito('${esc(c.id)}',false)">Não — SUBENQUADRAMENTO REAL</button>
          <button class="btn ghost" onclick="analisarRito('${esc(c.id)}')">Reanalisar</button>
        </div></div>`;
  }
  return `<article class="item" id="ri-${esc(c.id)}" style="border-left-color:var(--amarelo)">
      <div class="topo"><div>
        <div class="obj">${esc(c.objeto)}</div>
        <div class="metas"><span class="tag">🏛 ${esc(c.orgao || "—")}</span>
          <span class="tag tnum">💰 ${brl(c.valor)}</span>
          ${c.tipo_eng ? `<span class="tag tp">🔧 ${esc(c.tipo_eng)}</span>` : ""}
          <span class="tag">${esc(c.id)}</span></div></div>
        <div class="medidor"><div class="pct tnum">${Math.round(c.score * 100)}%</div><div class="cap">suspeita</div></div></div>
      <div class="acoes">${evid}</div></article>`;
}
async function analisarRito(id) {
  toast("Baixando e analisando o documento…");
  const r = await api("/api/rito/analisar/" + encodeURIComponent(id), { method: "POST" });
  if (!r.rito?.obtido) toast("Documento não obtido — veja a mensagem no card.");
  ritoView();
}
async function vereditoRito(id, seguido) {
  await api("/api/rito/veredito", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contrato_id: id, rito_seguido: seguido }),
  });
  const card = $("#ri-" + CSS.escape(id));
  if (card) { card.style.opacity = "0"; setTimeout(ritoView, 280); }
  toast(seguido ? "Registrado: rótulo incorreto, mas processo correto."
    : "Registrado: SUBENQUADRAMENTO REAL (rito não seguido).");
}
window.analisarRito = analisarRito; window.vereditoRito = vereditoRito;

// ── Config ──────────────────────────────────────────────────────────────
async function config() {
  const c = await api("/api/config");
  $("#form").innerHTML = `
    <div class="campo"><label>Modo de atualização do modelo</label>
      <select id="cf_retrain_modo">
        <option value="por_feedbacks" ${c.retrain_modo === "por_feedbacks" ? "selected" : ""}>A cada N triagens</option>
        <option value="por_tempo" ${c.retrain_modo === "por_tempo" ? "selected" : ""}>Por intervalo de tempo</option>
      </select></div>
    <div class="campo"><label>Atualizar a cada N triagens</label>
      <input type="number" id="cf_retrain_n_feedbacks" min="1" value="${esc(c.retrain_n_feedbacks)}"></div>
    <div class="campo"><label>Intervalo de re-treino (minutos, modo por tempo)</label>
      <input type="number" id="cf_retrain_intervalo_min" min="5" value="${esc(c.retrain_intervalo_min)}"></div>
    <div class="campo"><label>Peso da triagem humana no aprendizado</label>
      <input type="number" id="cf_peso_feedback" min="1" step="0.5" value="${esc(c.peso_feedback)}"></div>
    <div class="campo"><label>Limiar de suspeita (0–1)</label>
      <input type="number" id="cf_limiar" min="0" max="1" step="0.05" value="${esc(c.limiar)}"></div>
    <div class="campo"><label>PDFs analisados por contrato no rito</label>
      <input type="number" id="cf_rito_max_docs" min="1" max="10" value="${esc(c.rito_max_docs)}"></div>
    <hr class="divisor">
    <div class="campo"><label>Monitoramento contínuo (buscar novos contratos no PNCP)</label>
      <select id="cf_ingest_ativo">
        <option value="0" ${c.ingest_ativo === "0" ? "selected" : ""}>Desativado</option>
        <option value="1" ${c.ingest_ativo === "1" ? "selected" : ""}>Ativado</option></select></div>
    <div class="campo"><label>Frequência de ingestão (minutos)</label>
      <input type="number" id="cf_ingest_intervalo_min" min="5" value="${esc(c.ingest_intervalo_min)}"></div>
    <div class="campo"><label>UF monitorada</label>
      <input type="text" id="cf_ingest_uf" value="${esc(c.ingest_uf)}"></div>
    <div class="linha" style="margin-top:6px">
      <button class="btn amar" onclick="salvarConfig()">Salvar</button>
      <button class="btn ghost" onclick="acao('/api/retrain','Modelo')">Atualizar modelo agora</button>
      <button class="btn ghost" onclick="acao('/api/ingest','Ingestão')">Buscar contratos agora</button>
    </div>`;
}
async function salvarConfig() {
  const ks = ["retrain_modo", "retrain_n_feedbacks", "retrain_intervalo_min", "peso_feedback",
    "limiar", "rito_max_docs", "ingest_ativo", "ingest_intervalo_min", "ingest_uf"];
  const dados = {}; ks.forEach(k => dados[k] = $("#cf_" + k).value);
  await api("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dados }) });
  toast("Configurações salvas.");
}
async function acao(url, nome) { const r = await api(url, { method: "POST" }); toast(r.msg || `${nome}: ${r.novos ?? "ok"}.`); }
window.salvarConfig = salvarConfig; window.acao = acao;

// ── Histórico ────────────────────────────────────────────────────────────
async function historico() {
  const { itens } = await api("/api/historico");
  const cls = { aprendizado: "st-ok", rito: "st-rito", veredito: "st-real", triagem: "st-novo", config: "st-ind", sistema: "st-ind" };
  $("#tb-hist").innerHTML = itens.length ? itens.map(h =>
    `<tr><td class="tnum">${esc(h.quando)}</td>
      <td><span class="pill ${cls[h.tipo] || ""}">${esc(h.tipo)}</span></td>
      <td>${esc(h.detalhe)}</td></tr>`).join("")
    : `<tr><td colspan="3" class="muted" style="padding:24px">Nenhum evento ainda.</td></tr>`;
}

ir(location.hash.slice(1) || "painel");
