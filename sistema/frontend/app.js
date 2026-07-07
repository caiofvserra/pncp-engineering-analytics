"use strict";
const api = (p, o) => fetch(p, o).then(r => r.json());
const $ = s => document.querySelector(s), $$ = s => document.querySelectorAll(s);
const brl = v => v == null ? "—" : "R$ " + Number(v).toLocaleString("pt-BR", { maximumFractionDigits: 0 });
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const STATUS = {
  novo: ["Aguardando triagem", "st-novo"], aguarda_rito: ["Confirmado → rito", "st-rito"],
  triagem_nao: ["Serviço comum", "st-nao"], subenq_real: ["Subenquadramento real", "st-real"],
  rito_seguido: ["Rótulo incorreto, processo ok", "st-ok"],
  rito_indeterminado: ["Indeterminado", "st-ind"], baixa: ["Baixa suspeita", "st-ind"],
};
const iaHint = c => c.llm_classe
  ? `<span class="tag ${c.llm_classe === "engenharia" ? "tp" : ""}" title="${esc(c.llm_motivo || "")}">🤖 IA: ${esc(c.llm_classe)}</span>` : "";

let tT;
function toast(m) { const t = $("#toast"); t.textContent = m; t.classList.add("on"); clearTimeout(tT); tT = setTimeout(() => t.classList.remove("on"), 3200); }

function ir(v) {
  $$("#nav button").forEach(b => b.setAttribute("aria-current", b.dataset.v === v));
  $$(".view").forEach(x => x.classList.toggle("on", x.id === v));
  location.hash = v;
  ({ painel, ranking, triagem, rito: ritoView, modelo, config, historico }[v] || (() => {}))();
}
$$("#nav button").forEach(b => b.onclick = () => ir(b.dataset.v));
window.addEventListener("hashchange", () => { const v = location.hash.slice(1); if (v) ir(v); });

// ── Painel ─────────────────────────────────────────────────────────────
async function painel() {
  const s = await api("/api/stats");
  $("#kpis").innerHTML = [
    ["Aguardando triagem", s.novos, true], ["Na fila de rito", s.aguarda_rito],
    ["Subenquadramento real", s.subenq_real], ["Valor (subenq. real)", brl(s.valor_subenq)],
    ["Contratos monitorados", s.gerais],
  ].map(([k, v, d]) => `<div class="kpi ${d ? "dark" : ""}"><div class="k">${k}</div><div class="v tnum">${v}</div></div>`).join("");
  const mx = Math.max(1, ...s.por_tipo.map(t => t.n));
  $("#por-tipo").innerHTML = s.por_tipo.length ? s.por_tipo.map(t =>
    `<div class="barra"><span class="lab" title="${esc(t.tipo)}">${esc(t.tipo)}</span>
      <span class="track"><span class="fill" style="width:${100 * t.n / mx}%"></span></span>
      <span class="n tnum">${t.n}</span></div>`).join("") : `<p class="muted">Fila vazia.</p>`;
  const f = [["Serviço comum", s.triagem_nao, "var(--verde)"], ["Confirmado → rito", s.aguarda_rito, "var(--azul500)"],
  ["Rito seguido", s.rito_seguido, "var(--amarelo)"], ["Subenquadramento real", s.subenq_real, "var(--vermelho)"]];
  const mf = Math.max(1, ...f.map(x => x[1]));
  $("#funil").innerHTML = f.map(([l, n, c]) =>
    `<div class="barra"><span class="lab">${l}</span><span class="track">
      <span class="fill" style="width:${100 * n / mf}%;background:${c}"></span></span>
      <span class="n tnum">${n}</span></div>`).join("");
}

// ── Ranking ────────────────────────────────────────────────────────────
async function ranking() {
  const { itens } = await api("/api/ranking?limit=100");
  $("#tb-ranking").innerHTML = itens.map((c, i) => {
    const [rot, cls] = STATUS[c.status] || [c.status, ""];
    return `<tr><td class="tnum">${i + 1}</td><td>${esc(c.objeto)}</td><td>${esc(c.orgao || "—")}</td>
      <td class="tnum">${brl(c.valor)}</td><td class="tnum"><b>${Math.round((c.score || 0) * 100)}%</b></td>
      <td>${c.llm_classe ? `<span class="pill ${c.llm_classe === "engenharia" ? "st-rito" : "st-nao"}">${esc(c.llm_classe)}</span>` : "—"}</td>
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
        <div class="metas"><span class="tag">🏛 ${esc(c.orgao || "—")}</span>
          <span class="tag tnum">💰 ${brl(c.valor)}</span>${iaHint(c)}
          <span class="tag">${esc(c.id)}</span></div>
      </div><div class="medidor"><div class="pct tnum">${Math.round((c.score || 0) * 100)}%</div>
        <div class="cap">suspeita</div></div></div>
      <div class="acoes">
        <div class="pergunta">Pelo objeto, é <b>engenharia/obras</b> registrado como serviço comum?</div>
        <textarea id="j-${esc(c.id)}" placeholder="Justificativa (opcional)"></textarea>
        <div class="linha">
          <button class="btn sim" onclick="decidir('${esc(c.id)}',true)">Sim, é engenharia → rito</button>
          <button class="btn nao" onclick="decidir('${esc(c.id)}',false)">Não, é serviço comum</button>
        </div></div>
    </article>`).join("");
}
async function decidir(id, eh) {
  const r = await api("/api/triagem", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contrato_id: id, eh_eng: eh, justificativa: $("#j-" + CSS.escape(id))?.value || "" }) });
  const card = $("#tr-" + CSS.escape(id));
  if (card) { card.style.opacity = "0"; card.style.transform = "translateX(28px)"; setTimeout(triagem, 280); }
  let msg = eh ? "Confirmado como engenharia — enviado ao rito." : "Registrado como serviço comum.";
  if (r.retreino?.ok) msg += " " + r.retreino.msg;
  toast(msg);
}
window.decidir = decidir;

// ── Rito ────────────────────────────────────────────────────────────────
async function ritoView() {
  const { itens } = await api("/api/rito/fila?limit=40");
  const el = $("#lista-rito");
  if (!itens.length) { el.innerHTML = `<div class="vazio">Nenhum contrato na fila de rito. Confirme suspeitos na aba <b>Triagem</b>.</div>`; return; }
  el.innerHTML = itens.map(cardRito).join("");
}
function cardRito(c) {
  const r = c.rito;
  let corpo = `<button class="btn amar" onclick="analisarRito('${esc(c.id)}')">📥 Baixar documento e analisar o rito</button>
    <p class="muted" style="margin-top:8px">Busca o edital/TR da licitação no PNCP e detecta os marcadores.</p>`;
  if (r) {
    const marc = JSON.parse(r.marcadores || "[]"), forte = r.mk_score >= 2;
    const llmTag = r.llm_rito ? `<span class="tag">🤖 IA: rito ${esc(r.llm_rito)}</span>` : "";
    corpo = `<div class="evid">
      <div class="evid-head">
        <span class="pill ${forte ? "st-ok" : "st-real"}">${r.mk_score} marcador(es) de rito</span>
        <span class="muted">compra ${esc(r.ncp_compra || "—")} · ${r.n_docs} doc(s) · ${r.chars} caracteres ${llmTag}</span>
      </div>
      <div class="marcadores">${marc.length ? marc.map(m => `<span class="mk on">✓ ${esc(m)}</span>`).join("")
        : `<span class="mk off">nenhum marcador de engenharia encontrado</span>`}</div>
      <details><summary>Trecho do documento</summary><pre class="trecho">${esc(r.trecho || "")}</pre></details>
      <div class="pergunta" style="margin-top:14px">O <b>rito de engenharia</b> foi seguido neste processo?</div>
      <div class="linha">
        <button class="btn nao" onclick="vereditoRito('${esc(c.id)}',true)">Sim — rito seguido</button>
        <button class="btn sim" onclick="vereditoRito('${esc(c.id)}',false)">Não — SUBENQUADRAMENTO REAL</button>
        <button class="btn ghost" onclick="analisarRito('${esc(c.id)}')">Reanalisar</button>
      </div></div>`;
  }
  return `<article class="item" id="ri-${esc(c.id)}" style="border-left-color:var(--amarelo)">
    <div class="topo"><div><div class="obj">${esc(c.objeto)}</div>
      <div class="metas"><span class="tag">🏛 ${esc(c.orgao || "—")}</span>
        <span class="tag tnum">💰 ${brl(c.valor)}</span><span class="tag">${esc(c.id)}</span></div></div>
      <div class="medidor"><div class="pct tnum">${Math.round((c.score || 0) * 100)}%</div><div class="cap">suspeita</div></div></div>
    <div class="acoes">${corpo}</div></article>`;
}
async function analisarRito(id) {
  toast("Baixando e analisando o documento…");
  const r = await api("/api/rito/analisar/" + encodeURIComponent(id), { method: "POST" });
  if (!r.rito?.obtido) toast("Documento não obtido — veja a mensagem no card.");
  ritoView();
}
async function vereditoRito(id, seguido) {
  await api("/api/rito/veredito", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contrato_id: id, rito_seguido: seguido }) });
  const card = $("#ri-" + CSS.escape(id));
  if (card) { card.style.opacity = "0"; setTimeout(ritoView, 280); }
  toast(seguido ? "Registrado: rótulo incorreto, processo correto." : "Registrado: SUBENQUADRAMENTO REAL.");
}
window.analisarRito = analisarRito; window.vereditoRito = vereditoRito;

// ── Modelo & IA ──────────────────────────────────────────────────────────
async function modelo() {
  const s = await api("/api/stats");
  $("#modelo-kpis").innerHTML = [
    ["Classificador", s.modelo_treinado ? "Treinado" : "Não treinado", s.modelo_treinado],
    ["Contratos de referência", s.referencia],
    ["Triagens acumuladas", s.triagens],
    ["Apoio por LLM", s.llm_disponivel ? "Disponível" : "Desativado"],
  ].map(([k, v, d]) => `<div class="kpi ${d ? "dark" : ""}"><div class="k">${k}</div><div class="v" style="font-size:20px">${v}</div></div>`).join("");
}

// ── Config ──────────────────────────────────────────────────────────────
async function config() {
  const c = await api("/api/config");
  const campo = (id, lab, tipo = "number", extra = "", dica = "") =>
    `<div class="campo"><label>${lab}</label><input type="${tipo}" id="cf_${id}" value="${esc(c[id])}" ${extra}>${dica ? `<div class="dica">${dica}</div>` : ""}</div>`;
  $("#form").innerHTML = `
    <h3 style="color:var(--azul700);margin-bottom:14px">Aprendizado</h3>
    <div class="campo"><label>Modo de atualização do modelo</label>
      <select id="cf_retrain_modo">
        <option value="por_feedbacks" ${c.retrain_modo === "por_feedbacks" ? "selected" : ""}>A cada N triagens</option>
        <option value="por_tempo" ${c.retrain_modo === "por_tempo" ? "selected" : ""}>Por intervalo de tempo</option></select></div>
    ${campo("retrain_n_feedbacks", "Re-treinar a cada N triagens", "number", "min=1")}
    ${campo("peso_feedback", "Peso da triagem humana", "number", "min=1 step=0.5")}
    ${campo("limiar", "Limiar de suspeita (0–1)", "number", "min=0 max=1 step=0.05")}
    ${campo("rito_max_docs", "PDFs por contrato no rito", "number", "min=1 max=10")}
    <hr class="divisor">
    <h3 style="color:var(--azul700);margin-bottom:14px">Monitoramento contínuo (PNCP)</h3>
    <div class="campo"><label>Ingestão automática de novos contratos</label>
      <select id="cf_ingest_ativo"><option value="0" ${c.ingest_ativo === "0" ? "selected" : ""}>Desativada</option>
        <option value="1" ${c.ingest_ativo === "1" ? "selected" : ""}>Ativada</option></select></div>
    ${campo("ingest_intervalo_dias", "Frequência (dias)", "number", "min=1", "30 = uma vez por mês")}
    ${campo("ingest_uf", "UF monitorada", "text")}
    <hr class="divisor">
    <h3 style="color:var(--azul700);margin-bottom:14px">Apoio por LLM (opcional)</h3>
    <div class="campo"><label>Usar LLM (veredito + leitura do rito)</label>
      <select id="cf_llm_ativo"><option value="0" ${c.llm_ativo === "0" ? "selected" : ""}>Desativado</option>
        <option value="1" ${c.llm_ativo === "1" ? "selected" : ""}>Ativado</option></select></div>
    ${campo("llm_base_url", "Endereço do servidor (Ollama)", "text", "", "ex.: http://127.0.0.1:11434")}
    ${campo("llm_modelo", "Modelo", "text")}
    <div class="linha" style="margin-top:8px">
      <button class="btn amar" onclick="salvarConfig()">Salvar</button>
      <button class="btn ghost" onclick="acao('/api/modelo/treinar','Modelo')">Re-treinar agora</button>
    </div>`;
}
async function salvarConfig() {
  const ks = ["retrain_modo", "retrain_n_feedbacks", "peso_feedback", "limiar", "rito_max_docs",
    "ingest_ativo", "ingest_intervalo_dias", "ingest_uf", "llm_ativo", "llm_base_url", "llm_modelo"];
  const dados = {}; ks.forEach(k => { const e = $("#cf_" + k); if (e) dados[k] = e.value; });
  await api("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dados }) });
  toast("Configurações salvas.");
}
async function acao(url, nome) { const r = await api(url, { method: "POST" }); toast(r.msg || `${nome}: ${r.novos ?? "ok"}.`); }
window.salvarConfig = salvarConfig; window.acao = acao;

// ── Histórico ────────────────────────────────────────────────────────────
async function historico() {
  const { itens } = await api("/api/historico");
  const cls = { aprendizado: "st-ok", modelo: "st-ok", rito: "st-rito", veredito: "st-real",
    classificacao: "st-novo", ingestao: "st-rito", import: "st-ind", config: "st-ind", sistema: "st-ind" };
  $("#tb-hist").innerHTML = itens.length ? itens.map(h =>
    `<tr><td class="tnum">${esc(h.quando)}</td><td><span class="pill ${cls[h.tipo] || ""}">${esc(h.tipo)}</span></td>
      <td>${esc(h.detalhe)}</td></tr>`).join("")
    : `<tr><td colspan="3" class="muted" style="padding:24px">Nenhum evento ainda.</td></tr>`;
}

ir(location.hash.slice(1) || "painel");
