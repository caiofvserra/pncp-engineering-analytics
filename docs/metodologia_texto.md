# Metodologia — texto para o slide do TCC

**Título:** Identificação de subenquadramento de engenharia no PNCP

**Ideia central (PU Learning):** os contratos rotulados como *engenharia* e *obras*
são exemplos confiáveis; só o rótulo *serviços gerais* é ruidoso — e é nele que se
escondem obras/engenharia mal classificadas.

---

## FASE A — Coleta e preparação
1. **Coleta de dados** — Baixa contratos do PNCP e separa obras, engenharia e
   serviços gerais pela categoria informada pelo órgão.
2. **Pré-processamento** — Padroniza os objetos e remove os termos burocráticos
   repetidos que não distinguem o serviço.

→ *objetos padronizados*

## FASE B — Modelagem (PU Learning)
3. **Representação semântica + filtro PU** — Transforma cada objeto em vetor de
   significado e aproxima os "serviços gerais" do núcleo de engenharia, isolando
   os candidatos suspeitos.
4. **Agrupamento** — Agrupa os candidatos por semelhança e mede a densidade de
   engenharia confirmada em cada grupo.
5. **Perfis de domínio (IA)** — A IA descreve o vocabulário típico de engenharia
   e de não-engenharia, formando o contexto do domínio.
6. **Treino + calibração** — Treina classificadores com os casos confiáveis e
   calibra a chance de um contrato ser engenharia.

→ *modelo treinado*

## FASE C — Detecção e validação
7. **Pontuação + ranqueamento** — Pontua todos os "serviços gerais" e combina a
   probabilidade com a densidade do grupo, gerando o ranking de suspeitos.
8. **Validação manual** — Uma amostra aleatória rotulada à mão mede a precisão
   real do modelo e define o ponto de corte.
9. **Visualização** — Projeta os contratos em 2D e em rede de similaridade: os
   suspeitos aparecem colados à engenharia confirmada.
10. **Revisão por IA** — A IA revisa os suspeitos do topo usando o contexto do
    domínio, descartando falsos positivos.

→ *suspeitos priorizados*

## FASE D — Verificação e entrega
11. **Análise do rito de engenharia** — Abre o edital / Termo de Referência da
    licitação e verifica se o rito foi seguido (ART/CREA, projeto básico, normas
    ABNT): distingue subenquadramento real de rótulo apenas equivocado.
12. **Consolidação e reuso** — Reúne modelo, ranking e evidências num relatório e
    permite triar automaticamente novos contratos.

---

**Fundamentação:** Lei 14.133/2021 · Lei 5.194/66 · resoluções do CONFEA
(apenas engenharia — CREA/ART; arquitetura fora do escopo).

---

## Como gerar/editar a imagem você mesmo

A figura é gerada a partir de `docs/metodologia.html` (HTML/CSS).

**Opção 1 — editar o HTML e re-renderizar**
1. Abra `docs/metodologia.html` no navegador para ver.
2. Edite os textos/cores no arquivo (as frases estão no gerador
   `docs/metodologia_build.py`, ou direto no HTML).
3. Para virar imagem: abra no Chrome → botão direito → "Salvar como imagem" não
   existe; use uma extensão de screenshot de página inteira, **ou** imprima em
   PDF (Ctrl+P → Salvar como PDF) e exporte a página.

**Opção 2 — refazer no PowerPoint / Google Slides (recomendado p/ editar no TCC)**
- Use esta imagem só como **layout de referência**.
- Crie 4 colunas (fases) com as cores: azul `#2c5f8a`, laranja `#c8702a`,
  verde `#5a8a2c`, vermelho `#b03030`.
- Em cada coluna, caixas arredondadas com o número + título em negrito e a frase
  abaixo. Setas entre as fases com o rótulo em itálico (objetos padronizados →
  modelo treinado → suspeitos priorizados).
- Assim você edita texto e tamanho direto no slide, sem depender do código.

**Opção 3 — draw.io**
- Importe a imagem como plano de fundo e redesenhe por cima, ou monte os blocos
  do zero com o mesmo esquema de cores/estrutura.
