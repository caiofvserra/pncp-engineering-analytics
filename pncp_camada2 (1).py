"""
pncp_camada2.py — Camada 2 do TCC: Termos de Referência e Editais (PDFs)
Projeto TCC: Identificação de Serviços de Engenharia em Contratações Públicas
Autor: Caio Serra

Este módulo ESTENDE pncp_analise.py com a coleta e análise dos PDFs anexados
às contratações no PNCP (Termos de Referência, Projetos Básicos, Editais,
Estudos Técnicos, etc.). É a Camada 2 do TCC, enquanto pncp_analise.py
implementa a Camada 1 (texto curto do objetoCompra).

Estrutura
─────────
SEÇÃO C2.1 — Endpoints de arquivos do PNCP e descoberta inicial
SEÇÃO C2.2 — Coleta robusta com cache em disco
SEÇÃO C2.3 — Extração de texto (PyMuPDF + pdfplumber + OCR)
SEÇÃO C2.4 — Features adicionais (ART/RRT/CREA/engenheiro responsável)
SEÇÃO C2.5 — Pipeline da Camada 2 (objeto + TR enriquecido)
SEÇÃO C2.6 — Comparação Camada 1 vs Camada 1+2

Pré-requisitos:
    pip install pymupdf pdfplumber pytesseract pillow
    apt-get install tesseract-ocr tesseract-ocr-por  (Linux/Colab)

Como usar no Colab:
    1. Faça upload de pncp_analise.py e pncp_camada2.py
    2. Execute pncp_analise.py primeiro (download Camada 1 + EDA)
    3. Execute:
        from pncp_camada2 import executar_camada2
        c2 = executar_camada2(df, p2, p3)   # df = limpo da Parte 1, p2/p3 do código existente
"""

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 0 — Instalação e imports
# ════════════════════════════════════════════════════════════════════════════
# No Colab descomente UMA VEZ:
# !apt-get install -y -qq tesseract-ocr tesseract-ocr-por
# !pip install -q pymupdf pdfplumber pytesseract pillow

import subprocess, sys, os, re, time, hashlib, json
from pathlib import Path
from typing import Optional


def _instalar(pacote: str, import_nome: str = None) -> bool:
    """Instala pacote com fallback para --break-system-packages."""
    import_nome = import_nome or pacote
    try:
        __import__(import_nome); return True
    except ImportError:
        pass
    print(f"   Instalando {pacote}...")
    for flags in [[], ["--break-system-packages"]]:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pacote] + flags,
                stderr=subprocess.DEVNULL)
            __import__(import_nome); return True
        except Exception:
            continue
    print(f"   [aviso] Instale manualmente: pip install {pacote}")
    return False


_instalar("pymupdf", "fitz")
_instalar("pdfplumber")
_instalar("pillow", "PIL")
_instalar("pytesseract")

# ── Imports do projeto Camada 1 ─────────────────────────────────────────────
# Reaproveita TODA a infraestrutura do pncp_analise.py: API client, tokenização,
# stopwords, EDA, classificação. A Camada 2 só adiciona PDF + comparação.
try:
    from pncp_analise import (
        # API client base
        BASE_URL, _get_com_retry, MAPA_ROTULO,
        # Tokenização e NLP
        tokenizar, bigramas, KEYWORDS_ENG, _normalizar,
        # Pipeline da Parte 2/3
        preprocessar_texto, construir_features, definir_modelos,
        treinar_com_cv, g_tabela_resultados, g_matrizes_confusao,
        g_curvas_roc_pr, gerar_ranking_subenquadramentos,
        # Helpers de gráfico
        _salvar, PALETA, EM_COLAB,
    )
    if EM_COLAB:
        from IPython.display import Image, display
except ImportError as e:
    raise ImportError(
        f"pncp_camada2.py exige pncp_analise.py no mesmo diretório.\n"
        f"Erro original: {e}"
    )

import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Bibliotecas de PDF (com fallback gracioso) ──────────────────────────────
try:
    import fitz  # PyMuPDF — extrator primário
    TEM_PYMUPDF = True
except ImportError:
    TEM_PYMUPDF = False

try:
    import pdfplumber   # fallback para tabelas e PDFs com layout complexo
    TEM_PDFPLUMBER = True
except ImportError:
    TEM_PDFPLUMBER = False

try:
    import pytesseract
    from PIL import Image as PILImage
    TEM_OCR = True
except ImportError:
    TEM_OCR = False

# Docling — fallback PESADO opcional para PDFs onde os 3 outros falharam.
# IBM Research, layout-aware, mas baixa ~2GB de modelos na primeira execução.
# Use APENAS para PDFs com texto < 200 chars após PyMuPDF + pdfplumber + OCR.
# Custo: 30-60s por PDF em CPU. Para acelerar, use GPU.
try:
    from docling.document_converter import DocumentConverter
    TEM_DOCLING = True
except ImportError:
    TEM_DOCLING = False

# Tenta usar tqdm do pncp_analise; senão fallback
try:
    from pncp_analise import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable

warnings.filterwarnings("ignore")
print(f"✅ Camada 2 — bibliotecas: PyMuPDF={TEM_PYMUPDF} | "
      f"pdfplumber={TEM_PDFPLUMBER} | OCR={TEM_OCR} | Docling={TEM_DOCLING}")
if not TEM_DOCLING:
    print("   (Docling não instalado — fallback opcional para PDFs difíceis. "
          "pip install docling p/ ativar)")


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C2.1 — Endpoints e descoberta de documentos
# ════════════════════════════════════════════════════════════════════════════
#
# Endpoints da API de integração do PNCP (https://pncp.gov.br/api/pncp):
#
#   /v1/orgaos/{cnpj}/compras/{ano}/{seq}/arquivos                  → lista
#   /v1/orgaos/{cnpj}/compras/{ano}/{seq}/arquivos/{seqDoc}         → baixa
#
#   /v1/orgaos/{cnpj}/contratos/{ano}/{seq}/arquivos                → lista
#   /v1/orgaos/{cnpj}/contratos/{ano}/{seq}/arquivos/{seqDoc}       → baixa
#
# A base de integração é diferente da base de consulta usada no Camada 1.
# GETs são públicos (sem auth), apenas POST/PUT/DELETE exigem token.
# ────────────────────────────────────────────────────────────────────────────

API_INTEGRACAO = "https://pncp.gov.br/api/pncp"

# Códigos de tipo de documento (Manual de Integração v2.2.1, §5.12)
MAPA_TIPO_DOCUMENTO = {
    1:  "Aviso de Contratação Direta",
    2:  "Edital",
    3:  "Minuta do Contrato",
    4:  "Termo de Referência",
    5:  "Anteprojeto",
    6:  "Projeto Básico",
    7:  "Estudo Técnico Preliminar",
    8:  "Projeto Executivo",
    9:  "Mapa de Riscos",
    10: "DOD/DFD",
    11: "Ata de Registro de Preço",
    12: "Contrato",
    13: "Termo de Rescisão",
    14: "Termo Aditivo",
    15: "Termo de Apostilamento",
    17: "Nota de Empenho",
}

# Tipos mais relevantes para detectar engenharia (em ordem de prioridade)
TIPOS_RELEVANTES_ENGENHARIA = {
    4:  "Termo de Referência",      # 🟢 detalhamento técnico
    6:  "Projeto Básico",           # 🟢 obrigatório para obras/serv. eng.
    7:  "Estudo Técnico Preliminar",# 🟡 contém justificativa técnica
    8:  "Projeto Executivo",        # 🟢 só existe em obras
    5:  "Anteprojeto",              # 🟡 fase preliminar de eng.
    2:  "Edital",                   # 🟡 contém todos os anexos resumidos
    3:  "Minuta do Contrato",       # 🟢 cláusulas técnicas
    12: "Contrato",                 # 🟢 versão final
    14: "Termo Aditivo",            # 🟢 alterações de escopo (Camada 3)
}


def _decompor_numero_controle_pncp(num_controle: str) -> Optional[dict]:
    """
    Decompõe `numeroControlePNCP` em {cnpj, tipo, sequencial, ano}.

    Formato (Manual PNCP §4.1):
        {cnpj14}-{tipo1}-{seq6}/{ano4}
        tipo: 1=Compra/Contratação, 2=Contrato, 3=Ata, 4=PCA

    Exemplo: "12345678000199-1-000123/2024"
        → cnpj="12345678000199", tipo=1, seq=123, ano=2024
    """
    if not num_controle or not isinstance(num_controle, str):
        return None
    m = re.match(r"^(\d{14})-(\d+)-(\d+)/(\d{4})$", num_controle.strip())
    if not m:
        return None
    return {
        "cnpj":       m.group(1),
        "tipo":       int(m.group(2)),     # 1=compra, 2=contrato, ...
        "sequencial": int(m.group(3)),
        "ano":        int(m.group(4)),
    }


def listar_documentos_compra(cnpj: str, ano: int, sequencial: int,
                              timeout: int = 60) -> list:
    """
    Lista metadados dos documentos anexados a uma contratação.

    Endpoint: GET /v1/orgaos/{cnpj}/compras/{ano}/{seq}/arquivos
    Retorna lista de dicts com:
        sequencialDocumento, titulo, tipoDocumentoId, tipoDocumentoNome,
        url, statusAtivo, dataPublicacaoPncp
    """
    url = (f"{API_INTEGRACAO}/v1/orgaos/{cnpj}/compras/"
           f"{ano}/{sequencial}/arquivos")
    resp = _get_com_retry(url, params={}, tentativas=3)
    if resp is None or resp.status_code != 200:
        return []
    try:
        payload = resp.json()
        return payload if isinstance(payload, list) else payload.get("data", [])
    except (ValueError, AttributeError):
        return []


def listar_documentos_contrato(cnpj: str, ano: int, sequencial: int) -> list:
    """Lista documentos anexados a um contrato (pós-homologação)."""
    url = (f"{API_INTEGRACAO}/v1/orgaos/{cnpj}/contratos/"
           f"{ano}/{sequencial}/arquivos")
    resp = _get_com_retry(url, params={}, tentativas=3)
    if resp is None or resp.status_code != 200:
        return []
    try:
        payload = resp.json()
        return payload if isinstance(payload, list) else payload.get("data", [])
    except (ValueError, AttributeError):
        return []


def baixar_documento(cnpj: str, ano: int, sequencial: int,
                       seq_doc: int, tipo_recurso: str = "compras") -> Optional[bytes]:
    """
    Baixa o conteúdo binário de um documento (geralmente PDF).

    tipo_recurso: "compras" ou "contratos"
    Retorna bytes do PDF ou None em caso de erro.
    """
    url = (f"{API_INTEGRACAO}/v1/orgaos/{cnpj}/{tipo_recurso}/"
           f"{ano}/{sequencial}/arquivos/{seq_doc}")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code == 200:
            # Verifica se realmente é PDF (alguns docs podem ser .docx)
            ct = resp.headers.get("Content-Type", "").lower()
            data = resp.content
            return data
        if resp.status_code == 404:
            print(f"      ⚠ Documento {seq_doc} não encontrado (404)")
        elif resp.status_code == 401 or resp.status_code == 403:
            print(f"      ⚠ Acesso negado ({resp.status_code})")
        else:
            print(f"      ⚠ HTTP {resp.status_code} ao baixar doc {seq_doc}")
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        print(f"      ⚠ {type(e).__name__} ao baixar doc {seq_doc}")
    return None


def descobrir_documentos(num_controle_pncp: str) -> dict:
    """
    Função de DESCOBERTA: dado um numeroControlePNCP, lista todos os
    documentos disponíveis e retorna estrutura com metadados.

    USE ESTA FUNÇÃO no Colab UMA VEZ com um numeroControlePNCP real
    da sua coleta SP/2025 para confirmar que os endpoints funcionam:

        descobrir_documentos("12345678000199-1-000123/2024")

    Se retornar lista de documentos com URL: ✅ está tudo OK
    Se retornar vazio: o contrato pode não ter PDF anexado. Tente outro.
    """
    info = _decompor_numero_controle_pncp(num_controle_pncp)
    if info is None:
        print(f"❌ numeroControlePNCP inválido: '{num_controle_pncp}'")
        print("   Formato esperado: '14digitos-N-NNNNNN/AAAA'")
        return {}

    print(f"\n🔍 DESCOBERTA: {num_controle_pncp}")
    print(f"   CNPJ: {info['cnpj']}  Tipo: {info['tipo']}  "
          f"Seq: {info['sequencial']}  Ano: {info['ano']}")

    # tipo=1 → contratação | tipo=2 → contrato
    if info["tipo"] == 1:
        docs = listar_documentos_compra(info["cnpj"], info["ano"], info["sequencial"])
        recurso = "compras"
    elif info["tipo"] == 2:
        docs = listar_documentos_contrato(info["cnpj"], info["ano"], info["sequencial"])
        recurso = "contratos"
    else:
        print(f"   ⚠ Tipo {info['tipo']} não suportado (apenas 1=compra, 2=contrato).")
        return info

    if not docs:
        print(f"   ⚠ Nenhum documento encontrado para este registro.")
        return {**info, "documentos": [], "recurso": recurso}

    print(f"   ✅ {len(docs)} documento(s) encontrado(s):")
    for d in docs:
        seq_doc = d.get("sequencialDocumento", "?")
        tipo_id = d.get("tipoDocumentoId", "?")
        tipo_nm = d.get("tipoDocumentoNome", MAPA_TIPO_DOCUMENTO.get(tipo_id, "Outro"))
        titulo  = d.get("titulo", "")[:60]
        ativo   = d.get("statusAtivo", True)
        print(f"      [{seq_doc}] tipo={tipo_id} ({tipo_nm}): {titulo} "
              f"{'✓' if ativo else '✗ inativo'}")

    return {**info, "documentos": docs, "recurso": recurso}


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C2.2 — Coleta de PDFs com cache em disco
# ════════════════════════════════════════════════════════════════════════════

PASTA_CACHE_PDF = "pdfs_pncp_cache"


def _path_cache(num_controle: str, seq_doc: int) -> Path:
    """Caminho local onde o PDF é salvo (cache evita re-download)."""
    safe = num_controle.replace("/", "_")
    return Path(PASTA_CACHE_PDF) / f"{safe}__doc{seq_doc}.pdf"


def coletar_pdfs_dataframe(df: pd.DataFrame,
                             tipos_aceitos: list = None,
                             max_contratos: int = None,
                             apenas_engenharia: bool = False,
                             apenas_geral: bool = False,
                             pasta_cache: str = None) -> pd.DataFrame:
    """
    Para cada contratação no df, baixa todos os PDFs anexados e salva em cache.

    Parâmetros
    ──────────
    df                : DataFrame da Camada 1 com `numeroControlePNCP` e `rotulo`
    tipos_aceitos     : lista de tipoDocumentoId (default = TIPOS_RELEVANTES_ENGENHARIA)
    max_contratos     : limita o nº de contratações processadas (None = todos)
    apenas_engenharia : se True, só baixa PDFs de contratos rotulo='engenharia'
    apenas_geral      : se True, só baixa PDFs de contratos rotulo='geral'
                        (útil para focar nos suspeitos de subenquadramento)
    pasta_cache       : pasta de cache (default = PASTA_CACHE_PDF)

    Retorna
    ───────
    DataFrame com uma linha por DOCUMENTO baixado:
        numeroControlePNCP, rotulo, sequencialDocumento, tipoDocumentoId,
        tipoDocumentoNome, titulo, caminho_local, tamanho_bytes
    """
    global PASTA_CACHE_PDF      # tem que vir ANTES de qualquer leitura/escrita
    if pasta_cache is None:
        pasta_cache = PASTA_CACHE_PDF
    Path(pasta_cache).mkdir(exist_ok=True)
    PASTA_CACHE_PDF = pasta_cache

    if tipos_aceitos is None:
        tipos_aceitos = list(TIPOS_RELEVANTES_ENGENHARIA.keys())

    # Filtra dataframe
    df_use = df.copy()
    if apenas_engenharia:
        df_use = df_use[df_use["rotulo"] == "engenharia"]
    if apenas_geral:
        df_use = df_use[df_use["rotulo"] == "geral"]
    if max_contratos:
        df_use = df_use.head(max_contratos)

    print(f"\n📥 Coleta de PDFs — {len(df_use):,} contratações alvo")
    print(f"   Tipos aceitos: {tipos_aceitos}  ({len(tipos_aceitos)} tipos)")
    print(f"   Cache em: {pasta_cache}/")

    registros = []
    n_404 = 0
    n_ok  = 0
    n_cache_hit = 0

    for _, row in tqdm(df_use.iterrows(), total=len(df_use),
                        desc="📥 Contratações"):
        num_ctrl = row["numeroControlePNCP"]
        info = _decompor_numero_controle_pncp(num_ctrl)
        if info is None:
            continue
        if info["tipo"] not in (1, 2):
            continue

        # Listar documentos
        if info["tipo"] == 1:
            docs = listar_documentos_compra(info["cnpj"], info["ano"], info["sequencial"])
            recurso = "compras"
        else:
            docs = listar_documentos_contrato(info["cnpj"], info["ano"], info["sequencial"])
            recurso = "contratos"

        if not docs:
            n_404 += 1
            continue

        for d in docs:
            # PNCP às vezes retorna apenas tipoDocumentoNome (sem ID).
            # Tentamos várias formas de identificar o tipo.
            tipo_id = d.get("tipoDocumentoId")
            if tipo_id is None:
                # Fallback: derivar ID a partir do nome
                nome_doc = (d.get("tipoDocumentoNome") or "").strip()
                # Mapa reverso nome → ID (case-insensitive, normalização)
                nome_norm = _normalizar(nome_doc) if nome_doc else ""
                for k_id, k_nome in MAPA_TIPO_DOCUMENTO.items():
                    if _normalizar(k_nome) == nome_norm:
                        tipo_id = k_id
                        break

            # Se ainda não conseguiu identificar o tipo:
            #   - se tipos_aceitos foi explicitamente fornecido como lista vazia,
            #     descarta (usuário quer estritos)
            #   - caso contrário, ACEITA com tipo_id=None (melhor pegar do que pular)
            if tipo_id is None:
                if not tipos_aceitos:
                    continue
                # Estratégia leniente: aceita documentos sem tipo identificado
                # quando o usuário não restringiu (típico em contratos —
                # muitos só vêm rotulados como "Contrato" sem ID numérico)
                tipo_aceito = True
            else:
                tipo_aceito = tipo_id in tipos_aceitos

            if not tipo_aceito:
                continue
            seq_doc = d.get("sequencialDocumento")
            if seq_doc is None:
                continue

            # Verifica cache
            caminho = _path_cache(num_ctrl, seq_doc)
            if caminho.exists() and caminho.stat().st_size > 0:
                n_cache_hit += 1
            else:
                # Baixa
                conteudo = baixar_documento(info["cnpj"], info["ano"],
                                              info["sequencial"], seq_doc, recurso)
                if conteudo is None:
                    continue
                caminho.write_bytes(conteudo)
                n_ok += 1
                time.sleep(0.4)  # respeita servidor

            registros.append({
                "numeroControlePNCP":  num_ctrl,
                "rotulo":              row.get("rotulo"),
                "sequencialDocumento": seq_doc,
                "tipoDocumentoId":     tipo_id,
                "tipoDocumentoNome":   d.get("tipoDocumentoNome",
                                              MAPA_TIPO_DOCUMENTO.get(tipo_id, "Outro")),
                "titulo":              d.get("titulo", "")[:200],
                "caminho_local":       str(caminho),
                "tamanho_bytes":       caminho.stat().st_size if caminho.exists() else 0,
            })

    print(f"\n✅ Coleta concluída:")
    print(f"   • Downloads novos:    {n_ok:,}")
    print(f"   • Hits de cache:      {n_cache_hit:,}")
    print(f"   • Sem PDF disponível: {n_404:,}")
    print(f"   • Total registrado:   {len(registros):,}")

    return pd.DataFrame(registros)


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C2.3 — Extração de texto (PyMuPDF + pdfplumber + OCR)
# ════════════════════════════════════════════════════════════════════════════

def extrair_texto_pymupdf(caminho: str) -> tuple:
    """
    Extrai texto com PyMuPDF (fitz). Rápido e preciso para PDFs nativos.
    Retorna (texto_completo, num_paginas, sucesso_bool).
    """
    if not TEM_PYMUPDF:
        return "", 0, False
    try:
        doc = fitz.open(caminho)
        n_pag = doc.page_count
        textos = []
        for pg in doc:
            t = pg.get_text("text") or ""
            textos.append(t)
        doc.close()
        texto = "\n".join(textos)
        return texto, n_pag, True
    except Exception as e:
        print(f"   [aviso] PyMuPDF falhou em {caminho}: {e}")
        return "", 0, False


def extrair_texto_pdfplumber(caminho: str) -> tuple:
    """
    Fallback com pdfplumber. Melhor para PDFs com tabelas e layout complexo.
    Mais lento que PyMuPDF.
    """
    if not TEM_PDFPLUMBER:
        return "", 0, False
    try:
        textos = []
        with pdfplumber.open(caminho) as pdf:
            n_pag = len(pdf.pages)
            for pg in pdf.pages:
                t = pg.extract_text() or ""
                textos.append(t)
        return "\n".join(textos), n_pag, True
    except Exception as e:
        print(f"   [aviso] pdfplumber falhou em {caminho}: {e}")
        return "", 0, False


def extrair_texto_ocr(caminho: str, lang: str = "por",
                        max_paginas_ocr: int = 30) -> tuple:
    """
    Último recurso: OCR via Tesseract. Para PDFs escaneados.

    Limita max_paginas_ocr porque OCR é MUITO lento (~5s/página em CPU).
    """
    if not TEM_OCR or not TEM_PYMUPDF:
        return "", 0, False
    try:
        doc = fitz.open(caminho)
        n_pag = doc.page_count
        textos = []
        for i, pg in enumerate(doc):
            if i >= max_paginas_ocr:
                textos.append(f"\n[OCR truncado em {max_paginas_ocr} páginas]")
                break
            # Renderiza página como imagem em DPI 200 (bom para OCR)
            pix = pg.get_pixmap(dpi=200)
            img_path = f"/tmp/_ocr_{os.path.basename(caminho)}_{i}.png"
            pix.save(img_path)
            try:
                t = pytesseract.image_to_string(PILImage.open(img_path), lang=lang)
                textos.append(t)
            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)
        doc.close()
        return "\n".join(textos), n_pag, True
    except Exception as e:
        print(f"   [aviso] OCR falhou em {caminho}: {e}")
        return "", 0, False


def extrair_texto_docling(caminho: str) -> tuple:
    """
    Quarta tentativa de extração — Docling (IBM Research, layout-aware).

    Use APENAS quando PyMuPDF, pdfplumber e OCR falharam (extrair_texto_robusto
    ativa Docling automaticamente nesse caso, se a biblioteca estiver instalada).

    Vantagens em relação aos outros:
      • Modelos especializados em layout (DocLayNet)
      • Reconstrução de tabelas (TableFormer) — útil para planilhas embutidas
      • Robustez para PDFs com hierarquia complexa (capas, anexos)

    Custos:
      • ~2GB de download de modelos na primeira execução
      • 30-60s por PDF em CPU; muito mais rápido em GPU
      • Não substitui PyMuPDF para PDFs nascidos digitais (mais lento sem ganho)
    """
    if not TEM_DOCLING:
        return ("", 0, False)
    try:
        converter = DocumentConverter()
        result = converter.convert(caminho)
        # Docling devolve um objeto rico; aqui pegamos o texto Markdown
        # consolidado (preserva ordem de leitura, headers, listas)
        texto = result.document.export_to_markdown()
        n_pag = len(result.document.pages) if hasattr(result.document, "pages") else 0
        return (texto, n_pag, bool(texto.strip()))
    except Exception as e:
        print(f"   [docling] erro em {os.path.basename(caminho)}: {e}")
        return ("", 0, False)


def extrair_texto_robusto(caminho: str,
                            usar_ocr_se_vazio: bool = True,
                            usar_docling_fallback: bool = True,
                            min_chars_para_ok: int = 200) -> dict:
    """
    Estratégia em até 4 camadas (cada camada só roda se a anterior falhou):
      1. PyMuPDF (rápido, ~1-3s)              → padrão p/ PDFs nascidos digitais
      2. pdfplumber (médio, ~3-10s)           → quando PyMuPDF retorna pouco
      3. OCR Tesseract (lento, ~30s)          → para PDFs scaneados
      4. Docling (muito lento, ~30-60s)       → último recurso para PDFs difíceis

    Parâmetros
    ──────────
    usar_ocr_se_vazio    : se True, ativa Tesseract quando PyMuPDF/pdfplumber
                            retornaram pouco texto (default True)
    usar_docling_fallback: se True E Docling estiver instalado, ativa Docling
                            como ÚLTIMA tentativa quando todos os outros
                            falharam (default True). Adiciona robustez para
                            PDFs com layout complexo, mas é lento (30-60s/PDF).
    min_chars_para_ok    : quantos chars uma extração precisa retornar para ser
                            considerada "boa" e parar de tentar fallbacks

    Retorna dict com:
        texto, num_paginas, metodo_usado, sucesso, tamanho_bytes
        metodo_usado ∈ {'pymupdf', 'pdfplumber', 'ocr', 'docling',
                         'pymupdf_curto', 'falha'}
    """
    resultado = {
        "caminho":       caminho,
        "texto":         "",
        "num_paginas":   0,
        "metodo_usado":  None,
        "sucesso":       False,
        "tamanho_bytes": 0,
    }
    if not os.path.exists(caminho):
        return resultado
    resultado["tamanho_bytes"] = os.path.getsize(caminho)

    # Tentativa 1: PyMuPDF
    texto, n_pag, ok = extrair_texto_pymupdf(caminho)
    if ok and len(texto.strip()) >= min_chars_para_ok:
        resultado.update({"texto": texto, "num_paginas": n_pag,
                           "metodo_usado": "pymupdf", "sucesso": True})
        return resultado

    # Tentativa 2: pdfplumber
    texto2, n_pag2, ok2 = extrair_texto_pdfplumber(caminho)
    if ok2 and len(texto2.strip()) >= min_chars_para_ok:
        resultado.update({"texto": texto2, "num_paginas": n_pag2,
                           "metodo_usado": "pdfplumber", "sucesso": True})
        return resultado

    # Tentativa 3: OCR (se há páginas mas pouco texto, provavelmente escaneado)
    if usar_ocr_se_vazio and (n_pag > 0 or n_pag2 > 0):
        print(f"   [OCR] {os.path.basename(caminho)} parece escaneado, tentando OCR...")
        texto3, n_pag3, ok3 = extrair_texto_ocr(caminho)
        if ok3 and len(texto3.strip()) >= min_chars_para_ok:
            resultado.update({"texto": texto3, "num_paginas": max(n_pag, n_pag3),
                               "metodo_usado": "ocr", "sucesso": True})
            return resultado

    # Tentativa 4: Docling (último recurso, layout-aware, lento)
    if usar_docling_fallback and TEM_DOCLING:
        print(f"   [DOCLING] {os.path.basename(caminho)} difícil, tentando Docling...")
        texto4, n_pag4, ok4 = extrair_texto_docling(caminho)
        if ok4 and len(texto4.strip()) >= min_chars_para_ok:
            resultado.update({
                "texto": texto4,
                "num_paginas": max(n_pag, n_pag2, n_pag4),
                "metodo_usado": "docling",
                "sucesso": True,
            })
            return resultado

    # Pega o melhor entre PyMuPDF e pdfplumber mesmo se curto
    melhor_texto = texto if len(texto) > len(texto2) else texto2
    melhor_npag  = n_pag if len(texto) > len(texto2) else n_pag2
    resultado.update({
        "texto": melhor_texto,
        "num_paginas": melhor_npag,
        "metodo_usado": "pymupdf_curto" if len(melhor_texto) > 0 else "falha",
        "sucesso": len(melhor_texto.strip()) > 0,
    })
    return resultado


def extrair_textos_em_lote(df_pdfs: pd.DataFrame,
                             usar_ocr_se_vazio: bool = True) -> pd.DataFrame:
    """
    Extrai texto de todos os PDFs do DataFrame de coleta.

    Retorna o df_pdfs com colunas adicionadas:
        texto_extraido, num_paginas, metodo_usado, sucesso_extracao
    """
    print(f"\n📄 Extração de texto — {len(df_pdfs):,} PDFs")
    if not TEM_PYMUPDF:
        print("   ❌ PyMuPDF não disponível. Instale: pip install pymupdf")
        return df_pdfs

    textos = []
    paginas = []
    metodos = []
    sucessos = []
    tamanhos = []

    for caminho in tqdm(df_pdfs["caminho_local"].tolist(),
                          desc="📄 Extração"):
        r = extrair_texto_robusto(caminho, usar_ocr_se_vazio=usar_ocr_se_vazio)
        textos.append(r["texto"])
        paginas.append(r["num_paginas"])
        metodos.append(r["metodo_usado"])
        sucessos.append(r["sucesso"])
        tamanhos.append(r["tamanho_bytes"])

    df_pdfs = df_pdfs.copy()
    df_pdfs["texto_extraido"]   = textos
    df_pdfs["num_paginas"]      = paginas
    df_pdfs["metodo_usado"]     = metodos
    df_pdfs["sucesso_extracao"] = sucessos
    df_pdfs["tamanho_bytes"]    = tamanhos

    # Estatísticas
    print(f"\n── Resumo da extração ──")
    print(f"   PDFs com sucesso:     {sum(sucessos):,} de {len(sucessos):,}")
    print(f"   Páginas processadas:  {sum(paginas):,}")
    print(f"   Caracteres totais:    {sum(len(t) for t in textos):,}")
    print(f"\n   Métodos utilizados:")
    print(pd.Series(metodos).value_counts().to_string())

    return df_pdfs


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C2.4 — Features adicionais específicas de engenharia
# ════════════════════════════════════════════════════════════════════════════

# Marcadores legais e técnicos que SÓ aparecem em contratos de engenharia
# (Lei 14.133/2021, Lei 5.194/1966, Resolução CONFEA 218/1973, Lei 6.496/1977)
MARCADORES_ENGENHARIA = {
    "ART": [
        # Anotação de Responsabilidade Técnica — exigida pela Lei 6.496/1977
        # para qualquer atividade de engenharia
        r"\banota[çc][ãa]o\s+de\s+responsabilidade\s+t[ée]cnica\b",
        r"\bART\b(?:\s+do\s+CREA)?",
    ],
    "RRT": [
        # Registro de Responsabilidade Técnica (CAU - arquitetura)
        r"\bregistro\s+de\s+responsabilidade\s+t[ée]cnica\b",
        r"\bRRT\b(?:\s+do\s+CAU)?",
    ],
    "CREA": [
        r"\bCREA[/\s\-]?\w{0,2}\b",
        r"\bConselho\s+Regional\s+de\s+Engenharia\b",
        r"\bregistro\s+(?:no\s+)?CREA\b",
    ],
    "CAU": [
        r"\bCAU[/\s\-]?\w{0,2}\b",
        r"\bConselho\s+de\s+Arquitetura\s+e\s+Urbanismo\b",
    ],
    "ENGENHEIRO_RESPONSAVEL": [
        r"\bengenheiro\s+respons[áa]vel\b",
        r"\bengenheira?\s+respons[áa]vel\b",
        r"\brespons[áa]vel\s+t[ée]cnico\b",
        r"\bRT\s+do\s+(?:contrato|servi[çc]o|edital)\b",
    ],
    "ANOTACAO_RT": [
        r"\bquadro\s+de\s+respons[áa]veis\s+t[ée]cnicos\b",
        r"\batestado\s+de\s+capacidade\s+t[ée]cnica\b",
        r"\bACT\b(?:\s+do\s+CREA)?",
    ],
    "PROJETO_BASICO": [
        r"\bprojeto\s+b[áa]sico\b",
        r"\banteprojeto\s+de\s+engenharia\b",
        r"\bprojeto\s+executivo\b",
    ],
    "OBRA_SERVICO_ENGENHARIA": [
        r"\bobra\s+de\s+engenharia\b",
        r"\bservi[çc]o\s+(?:comum\s+)?de\s+engenharia\b",
        r"\bservi[çc]o\s+especial\s+de\s+engenharia\b",
    ],
    "LEI_14133_ENGENHARIA": [
        # Artigos da Nova Lei de Licitações que tratam de engenharia
        r"\bart\.?\s*6[°º]?,?\s*(?:inc(?:iso)?\.?\s*)?XII\b",   # serviço de engenharia
        r"\bart\.?\s*6[°º]?,?\s*(?:inc(?:iso)?\.?\s*)?XX(?:I+)?\b", # XX e XXI
    ],
    "ABNT_NORMAS_TECNICAS": [
        r"\bABNT\s+NBR\s*\d+",
        r"\bnorma\s+t[ée]cnica\s+(?:NBR\s*\d+|brasileira)\b",
    ],
}


def _normalizar_texto_pdf(texto: str) -> str:
    """
    Normaliza texto extraído de PDF para detecção robusta de regex.

    Aplica três correções essenciais que o OCR/PyMuPDF deixam:
      1. Hifenizações de quebra de linha:  'engenha-\\nria'  → 'engenharia'
      2. Quebras de linha que cortam palavras:  'res\\nponsável' → 'responsável'
      3. Espaços múltiplos colapsados em um só.

    Sem esta normalização, regex como `\\bengenharia\\b` perdem casos
    onde a palavra está quebrada em duas linhas.
    """
    if not texto:
        return ""
    # 1. Hifenização de quebra de linha
    texto = re.sub(r"-\s*\n\s*", "", texto)
    # 2. Quebras de linha simples → espaço (preserva parágrafos com \n\n)
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)
    # 3. Espaços múltiplos
    texto = re.sub(r"[ \t]+", " ", texto)
    return texto


def detectar_marcadores(texto: str) -> dict:
    """
    Aplica todos os regex de MARCADORES_ENGENHARIA e retorna contagens.

    Retorno: dict com {mk_nome: ocorrências_total} e
             {mk_nome_presente: True/False}, mais o score agregado
             `mk_score_engenharia` (0-9 = nº de tipos de marcadores presentes).

    Importância para o TCC:
    A presença de qualquer marcador ART/RRT/CREA é evidência FORTE de que
    o contrato envolve atividade de engenharia (Lei 6.496/1977 obriga ART
    para qualquer atividade de engenharia, inclusive quando rotulada como
    serviço comum). Contratos rotulados como 'geral' que mencionam ART/RRT
    são candidatos prioritários a subenquadramento.

    Pré-processamento do texto: normaliza hifenizações e quebras de linha
    típicas de PDFs antes de aplicar os regex (evita perder ocorrências
    que estavam quebradas entre linhas).
    """
    texto_norm = _normalizar_texto_pdf(texto).lower()
    resultado = {}
    for nome, padroes in MARCADORES_ENGENHARIA.items():
        total = 0
        for pat in padroes:
            matches = re.findall(pat, texto_norm, flags=re.IGNORECASE)
            total += len(matches)
        resultado[f"mk_{nome}"] = total
        resultado[f"mk_{nome}_presente"] = (total > 0)
    # Score agregado: nº de tipos de marcadores presentes (0-9)
    resultado["mk_score_engenharia"] = sum(
        1 for k, v in resultado.items() if k.endswith("_presente") and v
    )
    return resultado


def construir_features_camada2(df_pdfs: pd.DataFrame) -> pd.DataFrame:
    """
    A partir do df_pdfs (com texto extraído), gera features adicionais
    específicas para a Camada 2:
      • Marcadores ART/RRT/CREA/etc. (regex)
      • Comprimento do texto (chars, tokens, tokens únicos)
      • TTR (Type-Token Ratio)
      • Densidade de keywords de engenharia (KEYWORDS_ENG do Camada 1)
    """
    print(f"\n🔍 Extração de features Camada 2...")
    df = df_pdfs.copy()
    df["texto_extraido"] = df["texto_extraido"].fillna("").astype(str)

    # Marcadores
    print("   Detectando marcadores ART/RRT/CREA/etc...")
    marcadores_lst = []
    for txt in tqdm(df["texto_extraido"].tolist(), desc="🔍 Marcadores"):
        marcadores_lst.append(detectar_marcadores(txt))

    df_mk = pd.DataFrame(marcadores_lst, index=df.index)
    df = pd.concat([df, df_mk], axis=1)

    # Estatísticas textuais
    df["c2_len_chars"]  = df["texto_extraido"].str.len()
    df["c2_len_tokens"] = df["texto_extraido"].apply(
        lambda t: len(tokenizar(str(t)))
    )
    df["c2_n_keywords_eng"] = df["texto_extraido"].apply(
        lambda t: len(set(tokenizar(str(t))) & KEYWORDS_ENG)
    )

    # TTR
    def _ttr(t):
        toks = tokenizar(str(t))
        return len(set(toks)) / max(len(toks), 1)
    df["c2_ttr"] = df["texto_extraido"].apply(_ttr)

    print(f"   ✅ Features Camada 2: {df.shape[1]} colunas | {len(df):,} linhas")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C2.5 — Agregação por contrato e união com Camada 1
# ════════════════════════════════════════════════════════════════════════════

def agregar_pdfs_por_contrato(df_pdfs_features: pd.DataFrame) -> pd.DataFrame:
    """
    Cada contrato pode ter VÁRIOS PDFs (TR + PB + Edital + ...).
    Agregamos para ter UMA linha por contrato:
      • Concatena os textos extraídos
      • Soma os marcadores
      • Soma páginas, tokens etc.
      • Mantém o rótulo
    """
    if df_pdfs_features.empty:
        return pd.DataFrame()

    print(f"\n🔗 Agregando {len(df_pdfs_features)} PDFs por contrato...")

    # Identifica colunas de marcadores para soma
    cols_mk_count    = [c for c in df_pdfs_features.columns
                         if c.startswith("mk_") and not c.endswith("_presente")
                         and c != "mk_score_engenharia"]
    cols_mk_presente = [c for c in df_pdfs_features.columns
                          if c.endswith("_presente")]

    agg = {
        "rotulo":            "first",
        "texto_extraido":    lambda s: "\n\n[NOVO_DOC]\n\n".join(
                                  x for x in s.fillna("") if x.strip()),
        "num_paginas":       "sum",
        "c2_len_chars":      "sum",
        "c2_len_tokens":     "sum",
        "c2_n_keywords_eng": "max",
        "c2_ttr":            "mean",
        "tipoDocumentoNome": lambda s: " | ".join(sorted(set(s.dropna()))),
        "sucesso_extracao":  "any",
    }
    for c in cols_mk_count:
        agg[c] = "sum"
    for c in cols_mk_presente:
        agg[c] = "any"

    df_agg = df_pdfs_features.groupby("numeroControlePNCP", as_index=False).agg(agg)

    # Recalcula score agregado
    if cols_mk_presente:
        df_agg["mk_score_engenharia"] = df_agg[cols_mk_presente].sum(axis=1)

    print(f"   {len(df_agg):,} contratos únicos com pelo menos 1 PDF")
    return df_agg


def juntar_camada1_camada2(df_camada1: pd.DataFrame,
                             df_camada2_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Junta o DataFrame da Camada 1 (objetoCompra curto) com a Camada 2 (textos
    de PDFs agregados por contrato) usando `numeroControlePNCP` como chave.

    Se um contrato não tem PDF, mantém ele com texto Camada 2 vazio
    (left join). Adiciona flag `tem_pdf` para análise.
    """
    df_c1 = df_camada1.copy()

    if df_camada2_agg.empty:
        df_c1["texto_extraido"] = ""
        df_c1["num_paginas"]    = 0
        df_c1["tem_pdf"]        = False
        df_c1["mk_score_engenharia"] = 0
        return df_c1

    # Remove a coluna 'rotulo' da Camada 2 para evitar conflito
    df_c2 = df_camada2_agg.drop(columns=["rotulo"], errors="ignore")

    df_merged = df_c1.merge(df_c2, on="numeroControlePNCP", how="left")
    df_merged["tem_pdf"] = df_merged["texto_extraido"].notna() & \
                           (df_merged["texto_extraido"].fillna("").str.len() > 0)
    df_merged["texto_extraido"] = df_merged["texto_extraido"].fillna("")
    df_merged["num_paginas"]    = df_merged["num_paginas"].fillna(0).astype(int)

    # Texto enriquecido = objeto + texto do PDF
    df_merged["objeto_enriquecido"] = (
        df_merged["objeto"].astype(str) + " \n " +
        df_merged["texto_extraido"].astype(str)
    ).str.strip()

    n_com_pdf = df_merged["tem_pdf"].sum()
    print(f"\n✅ Camadas 1 e 2 unidas: {len(df_merged):,} contratos "
          f"({n_com_pdf:,} com PDF, {len(df_merged)-n_com_pdf:,} só Camada 1)")
    return df_merged


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C2.6 — Pipeline da Camada 2: classificação no texto enriquecido
# ════════════════════════════════════════════════════════════════════════════

def comparar_camada1_vs_camada2(df_unido: pd.DataFrame,
                                  pasta_saida: str,
                                  n_splits: int = 5) -> pd.DataFrame:
    """
    Treina o mesmo classificador em DUAS configurações para comparar:
      • Camada 1 apenas (texto = objeto curto)
      • Camada 1 + 2  (texto = objeto + texto do PDF)

    Usa Logistic Regression com class_weight='balanced' (igual ao baseline).

    Resultado esperado: F1-engenharia mais alto na Camada 1+2 graças ao
    contexto adicional dos TRs e Projetos Básicos. Esse é o resultado-chave
    do TCC.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix

    df = df_unido.copy()
    # Filtra só contratos com PDF para a comparação ser justa
    # (caso contrário, "Camada 1+2" cairia em registros sem texto extra)
    df_com_pdf = df[df["tem_pdf"]].copy()

    if len(df_com_pdf) < 50:
        print(f"\n⚠ Apenas {len(df_com_pdf)} contratos com PDF — amostra muito pequena.")
        print("   Aumente max_contratos na coleta_pdfs para resultado confiável.")
        return pd.DataFrame()

    print(f"\n🔬 Comparação Camada 1 × Camada 1+2  ({len(df_com_pdf):,} contratos com PDF)")

    y = df_com_pdf["rotulo"].map({"geral": 0, "engenharia": 1}).values

    # ── Camada 1: TF-IDF só sobre o objetoCompra ────────────────────────────
    print("   Construindo features Camada 1 (objeto curto)...")
    df_proc = preprocessar_texto(df_com_pdf)
    X_c1, _, _ = construir_features(df_proc, usar_metadados=True)

    # ── Camada 1+2: TF-IDF sobre objeto + texto do PDF ──────────────────────
    print("   Construindo features Camada 1+2 (objeto + PDF)...")
    df_proc2 = df_com_pdf.copy()
    df_proc2["objeto"] = df_proc2["objeto_enriquecido"]
    df_proc2 = preprocessar_texto(df_proc2)

    # Adiciona features dos marcadores Camada 2 ao espaço de features
    cols_marcadores = [c for c in df_com_pdf.columns
                          if c.startswith("mk_") and not c.endswith("_presente")]
    cols_marcadores += ["num_paginas", "c2_n_keywords_eng"]
    cols_marcadores = [c for c in cols_marcadores if c in df_com_pdf.columns]

    X_c12_texto, _, _ = construir_features(df_proc2, usar_metadados=True)
    if cols_marcadores:
        X_marcadores = StandardScaler().fit_transform(
            df_com_pdf[cols_marcadores].fillna(0).values)
        X_c12 = hstack([X_c12_texto, csr_matrix(X_marcadores)])
        print(f"   Features Camada 1+2: {X_c12.shape[1]:,} "
              f"(texto + {len(cols_marcadores)} marcadores)")
    else:
        X_c12 = X_c12_texto

    # ── CV comparativa ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    linhas = []
    for nome, X in [("Camada 1 (objeto)", X_c1),
                     ("Camada 1+2 (objeto + PDF)", X_c12)]:
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=1_000, solver="lbfgs", random_state=42)
        try:
            sc = cross_validate(clf, X, y, cv=cv,
                                 scoring=["accuracy", "f1_macro", "precision",
                                          "recall", "f1", "roc_auc",
                                          "average_precision"],
                                 n_jobs=1)
            linhas.append({
                "Configuração":  nome,
                "Dim":           X.shape[1],
                "Accuracy":      round(sc["test_accuracy"].mean(), 4),
                "F1-macro":      round(sc["test_f1_macro"].mean(), 4),
                "F1-engenharia": round(sc["test_f1"].mean(), 4),
                "Precision-eng": round(sc["test_precision"].mean(), 4),
                "Recall-eng":    round(sc["test_recall"].mean(), 4),
                "ROC-AUC":       round(sc["test_roc_auc"].mean(), 4),
                "Avg-Precision": round(sc["test_average_precision"].mean(), 4),
            })
        except Exception as e:
            print(f"   [aviso] {nome} falhou: {e}")

    if not linhas:
        return pd.DataFrame()

    tab = pd.DataFrame(linhas).set_index("Configuração")
    print("\n── Camada 1 × Camada 1+2 ──")
    print(tab.to_string())

    # Calcula DELTA para destaque
    if len(tab) == 2:
        delta_f1 = tab["F1-engenharia"].iloc[1] - tab["F1-engenharia"].iloc[0]
        print(f"\n   📈 Δ F1-engenharia: {delta_f1:+.4f} "
              f"({'GANHO' if delta_f1 > 0 else 'PERDA'} ao incluir PDFs)")

    # ── Gráfico ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    cols_plot = ["F1-engenharia", "Precision-eng", "Recall-eng", "Avg-Precision"]
    tab[cols_plot].plot(kind="bar", ax=ax, colormap="tab10",
                          edgecolor="white", linewidth=0.5)
    ax.set_title("Camada 1 (objeto) × Camada 1+2 (objeto + PDF)\n"
                 "Mesma LR + mesma CV — só muda a representação textual",
                 fontweight="bold")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_xticklabels(tab.index, rotation=10, ha="right")
    ax.legend(title="Métrica", bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "c2_01_comparacao_camadas.png", pasta_saida)

    # Salva tabela
    tab.to_csv(os.path.join(pasta_saida, "c2_comparacao_camadas.csv"))
    return tab


def g_marcadores_por_rotulo(df_unido: pd.DataFrame, pasta: str) -> pd.DataFrame:
    """
    Análise de presença de marcadores ART/RRT/CREA POR rótulo.

    Insight central do TCC: contratos rotulados como GERAL que contêm
    marcadores de engenharia em seus PDFs são fortes candidatos a
    subenquadramento (a Lei 6.496/1977 exige ART para serviços de eng.).
    """
    cols_mk_pres = [c for c in df_unido.columns if c.endswith("_presente")]
    if not cols_mk_pres:
        return pd.DataFrame()
    sub = df_unido[df_unido["tem_pdf"]].copy()
    if sub.empty:
        return pd.DataFrame()

    res = sub.groupby("rotulo")[cols_mk_pres].mean().mul(100).round(1)
    res.columns = [c.replace("mk_", "").replace("_presente", "")
                     for c in res.columns]
    print("\n── % de contratos COM cada marcador (por rótulo) ──")
    print(res.T.to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    res.T.plot(kind="bar", ax=ax,
                color=[PALETA.get(c, "#aaa") for c in res.index],
                edgecolor="white")
    ax.set_title("Marcadores legais/técnicos por rótulo (% contratos com PDF)\n"
                 "Lei 6.496/1977: ART obrigatória em serviços de engenharia",
                 fontweight="bold")
    ax.set_xlabel("Marcador"); ax.set_ylabel("% de contratos")
    ax.legend(title="Rótulo")
    ax.tick_params(axis="x", rotation=30)
    sns.despine(ax=ax); fig.tight_layout()
    _salvar(fig, "c2_02_marcadores_por_rotulo.png", pasta)

    # ⚠ Contratos GERAL com marcadores — o ouro do TCC
    geral_com_mk = sub[
        (sub["rotulo"] == "geral") &
        (sub.get("mk_score_engenharia", 0) >= 2)   # ≥ 2 marcadores diferentes
    ].sort_values("mk_score_engenharia", ascending=False)
    print(f"\n⚠ Contratos GERAL com ≥2 marcadores de engenharia: "
          f"{len(geral_com_mk):,}")
    if len(geral_com_mk) > 0:
        cols_show = ["numeroControlePNCP", "objeto", "valorTotalEstimado",
                     "mk_score_engenharia", "num_paginas"]
        cols_show = [c for c in cols_show if c in geral_com_mk.columns]
        print(geral_com_mk[cols_show].head(15).to_string(index=False))
        arq = os.path.join(pasta, "c2_subenquadramento_marcadores.csv")
        geral_com_mk[cols_show].to_csv(arq, index=False, encoding="utf-8-sig")
        print(f"\n   💾 {arq}")

    return res


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO C2.7 — Orquestrador
# ════════════════════════════════════════════════════════════════════════════

def executar_camada2(df: pd.DataFrame,
                      pasta_saida: str = None,
                      max_contratos: int = 200,
                      tipos_aceitos: list = None,
                      apenas_engenharia_ou_top_suspeitos: bool = True,
                      ranking_p2: pd.DataFrame = None,
                      n_splits: int = 5) -> dict:
    """
    Pipeline completo da Camada 2: download + extração + features +
    comparação Camada 1 × Camada 1+2.

    Por padrão prioriza contratos onde a Camada 2 vai dar maior valor:
      • TODOS os contratos rotulados como engenharia (rotulo='engenharia')
      • Top suspeitos do ranking da Parte 2 (geral com alta P(eng))
    Isso evita baixar PDF de toda a base inteira (~10.000 contratos = horas).

    Parâmetros
    ──────────
    df            : DataFrame da Camada 1 (output de carregar_e_limpar)
    pasta_saida   : pasta para gráficos e CSVs (default = igual à da Parte 1)
    max_contratos : limite total (engenharia + top suspeitos)
    tipos_aceitos : lista de tipoDocumentoId (default = TIPOS_RELEVANTES_ENGENHARIA)
    ranking_p2    : DataFrame de ranking da Parte 2 (p2['ranking']) — se None,
                     usa apenas contratos rotulo='engenharia'
    n_splits      : folds da CV
    """
    print("\n" + "█"*62)
    print("  CAMADA 2 — TR / Edital / Projeto Básico (PDFs)")
    print("█"*62)

    if pasta_saida is None:
        uf  = df["ufSigla"].mode()[0]       if "ufSigla" in df.columns else "xx"
        ano = df["anoPublicacao"].mode()[0] if "anoPublicacao" in df.columns else "xxxx"
        pasta_saida = f"graficos_pncp_{uf}_{ano}"
    os.makedirs(pasta_saida, exist_ok=True)

    # ── 1. Selecionar contratos alvo ────────────────────────────────────────
    print("\n[1] Selecionando contratos alvo...")

    df_eng = df[df["rotulo"] == "engenharia"].copy()
    print(f"   • Engenharia (todos):          {len(df_eng):,}")

    if ranking_p2 is not None and not ranking_p2.empty:
        suspeitos = ranking_p2.head(max_contratos // 2)
        # Garante que sejam só os 'geral' e exista no df
        suspeitos = suspeitos[suspeitos["numeroControlePNCP"].isin(df["numeroControlePNCP"])]
        df_susp = df[df["numeroControlePNCP"].isin(
            suspeitos["numeroControlePNCP"]
        )].copy()
        print(f"   • Top-{len(df_susp)} suspeitos do ranking P2: {len(df_susp):,}")
    else:
        df_susp = pd.DataFrame()
        print(f"   • Sem ranking P2 fornecido — usa apenas engenharia")

    df_alvo = pd.concat([df_eng, df_susp]).drop_duplicates("numeroControlePNCP")
    if len(df_alvo) > max_contratos:
        # Limita preservando todos de engenharia + suspeitos restantes
        if len(df_eng) >= max_contratos:
            df_alvo = df_eng.head(max_contratos)
        else:
            n_susp = max_contratos - len(df_eng)
            df_alvo = pd.concat([df_eng, df_susp.head(n_susp)]) \
                         .drop_duplicates("numeroControlePNCP")
    print(f"   → Total alvo: {len(df_alvo):,} contratos")

    # ── 2. Coleta de PDFs ───────────────────────────────────────────────────
    print(f"\n[2] Coletando PDFs (cache: {PASTA_CACHE_PDF}/)...")
    df_pdfs = coletar_pdfs_dataframe(
        df_alvo, tipos_aceitos=tipos_aceitos,
        max_contratos=None,    # já foi limitado acima
    )

    if df_pdfs.empty:
        print("\n❌ Nenhum PDF foi baixado. "
              "Pode ser que os contratos selecionados não tenham anexos no PNCP.")
        return {"df_pdfs": df_pdfs}

    df_pdfs.to_parquet(os.path.join(pasta_saida, "c2_pdfs_metadados.parquet"))

    # ── 3. Extração de texto ───────────────────────────────────────────────
    print(f"\n[3] Extração de texto dos PDFs...")
    df_pdfs = extrair_textos_em_lote(df_pdfs, usar_ocr_se_vazio=True)

    # ── 4. Features Camada 2 ───────────────────────────────────────────────
    print(f"\n[4] Features adicionais (ART/RRT/CREA/etc.)...")
    df_pdfs = construir_features_camada2(df_pdfs)

    df_pdfs.to_parquet(os.path.join(pasta_saida, "c2_pdfs_completos.parquet"))

    # ── 5. Agregação por contrato ──────────────────────────────────────────
    print(f"\n[5] Agregando por contrato...")
    df_agg = agregar_pdfs_por_contrato(df_pdfs)

    # ── 6. União com Camada 1 ──────────────────────────────────────────────
    print(f"\n[6] Unindo Camada 1 + Camada 2...")
    df_unido = juntar_camada1_camada2(df_alvo, df_agg)
    df_unido.to_parquet(os.path.join(pasta_saida, "c2_camadas_unidas.parquet"))

    # ── 7. Análises ─────────────────────────────────────────────────────────
    print(f"\n[7] Análise de marcadores por rótulo...")
    g_marcadores_por_rotulo(df_unido, pasta_saida)

    print(f"\n[8] Comparação Camada 1 × Camada 1+2...")
    tab_comp = comparar_camada1_vs_camada2(df_unido, pasta_saida, n_splits=n_splits)

    print("\n" + "█"*62)
    print("  CAMADA 2 CONCLUÍDA ✅")
    print(f"  PDFs baixados:        {len(df_pdfs):,}")
    print(f"  Contratos com PDF:    {df_unido['tem_pdf'].sum():,}")
    print(f"  Métodos de extração:  {df_pdfs['metodo_usado'].value_counts().to_dict()}")
    print("█"*62)

    return {
        "df_pdfs":           df_pdfs,
        "df_agregado":       df_agg,
        "df_unido":          df_unido,
        "comparacao":        tab_comp,
    }


# ════════════════════════════════════════════════════════════════════════════
# REFERÊNCIA RÁPIDA
# ════════════════════════════════════════════════════════════════════════════
#
# Pré-requisito (no Colab, UMA VEZ):
#   !apt-get install -y -qq tesseract-ocr tesseract-ocr-por
#   !pip install -q pymupdf pdfplumber pytesseract
#
# ── Antes de rodar a coleta completa, TESTE com UM contrato ───────────────
#   from pncp_camada2 import descobrir_documentos
#   # Pegue qualquer numeroControlePNCP do seu df de engenharia:
#   num = df[df["rotulo"]=="engenharia"]["numeroControlePNCP"].iloc[0]
#   info = descobrir_documentos(num)
#   # Se imprimir lista de docs com tipo 4 (TR) ou 6 (PB) → ✅ funciona
#
# ── Pipeline completo ────────────────────────────────────────────────────
#   from pncp_camada2 import executar_camada2
#   c2 = executar_camada2(
#       df=df,                              # do pncp_analise (Camada 1)
#       max_contratos=200,                  # ajuste conforme seu tempo
#       ranking_p2=p2["ranking"],           # foca nos suspeitos
#   )
#
# ── Resultados ───────────────────────────────────────────────────────────
#   c2["comparacao"]                # tabela Camada 1 × Camada 1+2
#   c2["df_unido"]                  # contratos com texto enriquecido
#   c2["df_pdfs"]                   # 1 linha por PDF baixado
#
# ── Subenquadramentos por marcadores (Lei 6.496/1977 — ART) ─────────────
#   suspeitos_marcadores = c2["df_unido"][
#       (c2["df_unido"]["rotulo"] == "geral") &
#       (c2["df_unido"]["mk_score_engenharia"] >= 2)
#   ].sort_values("mk_score_engenharia", ascending=False)
#   print(suspeitos_marcadores[["numeroControlePNCP","objeto","mk_score_engenharia"]])
