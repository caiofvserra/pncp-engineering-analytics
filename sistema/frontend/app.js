"use strict";
const api = (p, o) => fetch(p, o).then(r => r.json());
const $ = s => document.querySelector(s);
const brl = v => (v == null ? "—" : "R$ " + Number(v).toLocaleString("pt-BR",
                  { maximumFractionDigits: 0 }));
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2600);
}

// ── navegação (abas linkáveis por hash: /#fila, /#config, …) ────────────
function irPara(view) {
  const b = document.querySelector(`nav button[data-view="${view}"]`);
  if (!b) return;
  document.querySelectorAll("nav button").forEach(x => x.classList.remove("ativo"));
  document.querySelectorAll(".view").forEach(x => x.classList.remove("ativo"));
  b.classList.add("ativo"); $("#" + view).classList.add("ativo");
  if (location.hash.slice(1) !== view) location.hash = view;
  ({ painel: carregaPainel, fila: carregaFila, config: carregaConfig,
     historico: carregaHist }[view] || (() => {}))();
}
document.querySelectorAll("nav button").forEach(b => b.onclick = () => irPara(b.dataset.view));
window.addEventListener("hashchange", () => irPara(location.hash.slice(1) || "painel"));

// ── painel ─────────────────────────────────────────────────────────────
async function carregaPainel() {
  const s = await api("/api/stats");
  $("#cards").innerHTML = [
    ["Pendentes de revisão", s.pendentes, true],
    ["Já revisados", s.revisados],
    ["Confirmados (subenq.)", s.concordancias],
    ["Valor confirmado", brl(s.valor_confirmado)],
    ["Feedbacks p/ próximo re-treino", s.feedbacks_novos],
  ].map(([k, v, d]) => `<div class="card ${d ? "destaque" : ""}">
      <div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  const max = Math.max(1, ...s.por_tipo.map(t => t.n));
  $("#por-tipo").innerHTML = s.por_tipo.length ? s.por_tipo.map(t =>
    `<div class="barra"><span class="lab" title="${esc(t.tipo)}">${esc(t.tipo)}</span>
      <span class="track"><span class="fill" style="width:${100 * t.n / max}%"></span></span>
      <span class="n">${t.n}</span></div>`).join("") : `<p class="vazio">Sem dados.</p>`;

  const tot = s.concordancias + s.discordancias;
  $("#feedback-resumo").innerHTML = `
    <div class="barra"><span class="lab">Concordou (é subenq.)</span>
      <span class="track"><span class="fill" style="width:${tot ? 100 * s.concordancias / tot : 0}%;background:var(--vermelho)"></span></span>
      <span class="n">${s.concordancias}</span></div>
    <div class="barra"><span class="lab">Discordou (serviço comum)</span>
      <span class="track"><span class="fill" style="width:${tot ? 100 * s.discordancias / tot : 0}%;background:var(--verde)"></span></span>
      <span class="n">${s.discordancias}</span></div>
    <p class="legenda" style="margin-top:14px">O modelo aprende com cada decisão: concordâncias
      reforçam o padrão de engenharia, discordâncias corrigem os falsos positivos.</p>`;
}

// ── fila ───────────────────────────────────────────────────────────────
async function carregaFila() {
  const { itens } = await api("/api/fila?limit=40");
  const el = $("#lista-fila");
  if (!itens.length) { el.innerHTML = `<p class="vazio">🎉 Fila vazia — todos os suspeitos foram revisados.</p>`; return; }
  el.innerHTML = itens.map(it => `
    <div class="item" id="it-${esc(it.id)}">
      <div class="top">
        <div>
          <div class="obj">${esc(it.objeto)}</div>
          <div class="meta">
            ${it.orgao ? `<span class="tag">🏛 ${esc(it.orgao)}</span>` : ""}
            ${it.valor != null ? `<span class="tag">💰 ${brl(it.valor)}</span>` : ""}
            ${it.tipo_eng ? `<span class="tag tipo">🔧 ${esc(it.tipo_eng)}</span>` : ""}
            <span class="tag">${esc(it.id)}</span>
          </div>
        </div>
        <div class="score"><div class="pct">${Math.round(it.score * 100)}%</div>
          <div class="cap">suspeita</div></div>
      </div>
      <div class="acoes">
        <div class="pergunta">Este contrato é <b>subenquadramento</b> (engenharia registrada como serviço comum)?</div>
        <textarea id="just-${esc(it.id)}" placeholder="Justificativa / evidência do rito (opcional)"></textarea>
        <div class="linha">
          <label class="chk"><input type="checkbox" id="rito-${esc(it.id)}"> O rito de engenharia (ART/CREA, projeto básico) foi seguido</label>
        </div>
        <div class="linha" style="margin-top:12px">
          <button class="btn sim" onclick="responder('${esc(it.id)}',true)">Concordo — é subenquadramento</button>
          <button class="btn nao" onclick="responder('${esc(it.id)}',false)">Discordo — é serviço comum</button>
        </div>
      </div>
    </div>`).join("");
}

async function responder(id, concorda) {
  const r = await api("/api/feedback", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contrato_id: id, concorda,
      rito_ok: $("#rito-" + CSS.escape(id))?.checked ?? null,
      justificativa: $("#just-" + CSS.escape(id))?.value || "",
    }),
  });
  const card = document.getElementById("it-" + id);
  if (card) { card.style.transition = "opacity .3s"; card.style.opacity = "0";
              setTimeout(() => card.remove(), 300); }
  let msg = concorda ? "Registrado como subenquadramento." : "Registrado como serviço comum.";
  if (r.retreino?.ok) msg += " " + r.retreino.msg;
  toast(msg);
}
window.responder = responder;

// ── config ─────────────────────────────────────────────────────────────
async function carregaConfig() {
  const c = await api("/api/config");
  $("#form-config").innerHTML = `
    <label>Modo de atualização do modelo</label>
    <select id="c_retrain_modo">
      <option value="por_feedbacks" ${c.retrain_modo === "por_feedbacks" ? "selected" : ""}>A cada N feedbacks</option>
      <option value="por_tempo" ${c.retrain_modo === "por_tempo" ? "selected" : ""}>Por intervalo de tempo</option>
    </select>
    <label>Re-treinar a cada N feedbacks novos</label>
    <input id="c_retrain_n_feedbacks" type="number" min="1" value="${c.retrain_n_feedbacks}">
    <label>Intervalo de re-treino (minutos, modo por tempo)</label>
    <input id="c_retrain_intervalo_min" type="number" min="5" value="${c.retrain_intervalo_min}">
    <div class="hint">min. 5 minutos.</div>
    <label>Peso do rótulo humano no aprendizado</label>
    <input id="c_peso_feedback" type="number" min="1" step="0.5" value="${c.peso_feedback}">
    <label>Limiar de suspeita (0–1)</label>
    <input id="c_limiar" type="number" min="0" max="1" step="0.05" value="${c.limiar}">
    <hr style="margin:22px 0;border:0;border-top:1px solid var(--borda)">
    <label>Ingestão automática de novos contratos do PNCP</label>
    <select id="c_ingest_ativo">
      <option value="0" ${c.ingest_ativo === "0" ? "selected" : ""}>Desativada</option>
      <option value="1" ${c.ingest_ativo === "1" ? "selected" : ""}>Ativada</option>
    </select>
    <label>Frequência de ingestão (minutos)</label>
    <input id="c_ingest_intervalo_min" type="number" min="5" value="${c.ingest_intervalo_min}">
    <label>UF monitorada</label>
    <input id="c_ingest_uf" value="${c.ingest_uf}">
    <div class="linha">
      <button class="btn salvar" onclick="salvarConfig()">Salvar configurações</button>
      <button class="btn ghost" onclick="acaoRetrain()">Re-treinar agora</button>
      <button class="btn ghost" onclick="acaoIngest()">Buscar contratos agora</button>
    </div>`;
}
async function salvarConfig() {
  const ids = ["retrain_modo", "retrain_n_feedbacks", "retrain_intervalo_min",
    "peso_feedback", "limiar", "ingest_ativo", "ingest_intervalo_min", "ingest_uf"];
  const dados = {}; ids.forEach(k => dados[k] = $("#c_" + k).value);
  await api("/api/config", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dados }) });
  toast("Configurações salvas.");
}
async function acaoRetrain() { const r = await api("/api/retrain", { method: "POST" }); toast(r.msg || "Re-treino disparado."); }
async function acaoIngest() { const r = await api("/api/ingest", { method: "POST" }); toast(`Ingestão: ${r.novos} novos contratos suspeitos.`); carregaPainel(); }
window.salvarConfig = salvarConfig; window.acaoRetrain = acaoRetrain; window.acaoIngest = acaoIngest;

// ── histórico ──────────────────────────────────────────────────────────
async function carregaHist() {
  const { itens } = await api("/api/historico");
  const cls = s => s === "ok" ? "ok" : s === "ingest" ? "ing" : s === "erro" ? "erro" : "ing";
  $("#tbody-hist").innerHTML = itens.length ? itens.map(h =>
    `<tr><td>${esc(h.quando)}</td><td>${h.n_feedbacks}</td>
      <td><span class="badge ${cls(h.status)}">${esc(h.status)}</span></td>
      <td>${esc(h.detalhe)}</td></tr>`).join("")
    : `<tr><td colspan="4" class="vazio">Nenhuma atualização ainda.</td></tr>`;
}

irPara(location.hash.slice(1) || "painel");
