"""
Análise geográfica e NER de contratos.

Combina:
  - Reconhecimento de Entidades Nomeadas (Spacy PT) sobre objetos
  - Geocoding de municípios (via prefeitura → coordenadas)
  - Mapa Folium HeatMap dos contratos suspeitos

Mostra a DISTRIBUIÇÃO ESPACIAL de subenquadramento. Se um município
concentra muitos suspeitos, vale investigação local.
"""

import json
from pathlib import Path

import pandas as pd

from pncp import config
from pncp.io_disco import ler_parquet, salvar_json
from pncp.ram import com_gc


# ── NER de objetos com Spacy ────────────────────────────────────────────────
def extrair_entidades(amostra=200, modelo="pt_core_news_sm"):
    """
    Roda NER do Spacy sobre os objetos de contratos. Útil para identificar:
      - PER: nome de engenheiros/arquitetos responsáveis citados
      - ORG: empresas/órgãos
      - LOC: locais específicos da obra
      - MISC: normas técnicas, leis citadas

    Default: subset de 200 objetos (rápido). Spacy é leve mas roda em CPU.
    """
    try:
        import spacy
    except ImportError:
        print("[geo] instale spacy: pip install spacy")
        return None
    try:
        nlp = spacy.load(modelo)
    except OSError:
        print(f"[geo] modelo não baixado. Rode: "
              f"python -m spacy download {modelo}")
        return None

    df = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"),
                     colunas=["numeroControlePNCP", "objeto", "rotulo"])
    if df.empty:
        return None
    df = df.dropna(subset=["objeto"]).sample(
        n=min(amostra, len(df)), random_state=config.SEED)

    registros = []
    for _, row in df.iterrows():
        doc = nlp(str(row["objeto"])[:1000])
        ents = [{"texto": e.text, "tipo": e.label_} for e in doc.ents]
        registros.append({
            "numeroControlePNCP": row["numeroControlePNCP"],
            "objeto": str(row["objeto"])[:200],
            "rotulo": row["rotulo"],
            "n_entidades": len(ents),
            "entidades": ents,
        })

    out = pd.DataFrame(registros)
    saida = config.caminho("geografico", "ner_entidades.parquet")
    out.to_parquet(saida, index=False)
    print(f"[geo] NER em {len(out)} contratos → {saida}")

    # Resumo por tipo
    from collections import Counter
    contador = Counter()
    for ents in out["entidades"]:
        for e in ents:
            contador[e["tipo"]] += 1
    print(f"[geo] entidades por tipo: {dict(contador)}")
    return out


# ── Geocoding de municípios ────────────────────────────────────────────────
def geocodificar_municipios(amostra=None):
    """
    Geocodifica municípios via OpenStreetMap (geopy/Nominatim).
    Cuidado: Nominatim impõe rate limit ~1 req/s.

    Cacheia resultados em geografico/cache_geocoding.json.
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
    except ImportError:
        print("[geo] instale geopy: pip install geopy")
        return None
    import time

    df = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"),
                     colunas=["municipioNome", "rotulo"])
    municipios = df["municipioNome"].dropna().unique()
    if amostra:
        municipios = municipios[:amostra]
    print(f"[geo] geocodificando {len(municipios)} municípios "
          f"(rate limit ~1/s — vai demorar)")

    cache_path = config.caminho("geografico", "cache_geocoding.json")
    cache = {}
    if Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())

    geolocator = Nominatim(user_agent="pncp-tcc")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    for i, m in enumerate(municipios, 1):
        if m in cache:
            continue
        try:
            loc = geocode(f"{m}, Brasil", timeout=10)
            cache[m] = ({"lat": loc.latitude, "lng": loc.longitude}
                        if loc else None)
        except Exception:
            cache[m] = None
        if i % 25 == 0:
            print(f"[geo] {i}/{len(municipios)}")
            salvar_json(cache, cache_path)
        time.sleep(0.1)

    salvar_json(cache, cache_path)
    n_ok = sum(1 for v in cache.values() if v)
    print(f"[geo] geocoding: {n_ok}/{len(cache)} OK")
    return cache


# ── Mapa Folium HeatMap ──────────────────────────────────────────────────────
def mapa_suspeitos(html_path=None):
    """
    Mapa interativo (Folium HeatMap) dos contratos suspeitos por município.
    Salva como HTML — abra no navegador para zoom e pop-ups.
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("[geo] instale folium: pip install folium")
        return None

    susp = ler_parquet(config.caminho(config.SUB_P9,
                                       "suspeitos_consolidados.parquet"))
    if susp.empty or "municipioNome" not in susp.columns:
        print("[geo] sem suspeitos consolidados ou sem municipioNome")
        return None

    cache_path = config.caminho("geografico", "cache_geocoding.json")
    if not Path(cache_path).exists():
        print("[geo] rode geocodificar_municipios() primeiro")
        return None
    cache = json.loads(Path(cache_path).read_text())

    # Conta suspeitos por município com coordenada conhecida
    n_sinais = susp.get("n_sinais", 0)
    susp_forte = susp[n_sinais >= 2]
    contagem = susp_forte.groupby("municipioNome").size().to_dict()

    pontos = []
    for m, n in contagem.items():
        coord = cache.get(m)
        if coord and isinstance(coord, dict):
            # HeatMap: peso = quantidade de suspeitos
            pontos.append([coord["lat"], coord["lng"], n])

    if not pontos:
        print("[geo] nenhum município com coordenada — geocodifique primeiro")
        return None

    # Centro: Brasil
    mapa = folium.Map(location=[-15.78, -47.93], zoom_start=4,
                       tiles="cartodbpositron")
    HeatMap(pontos, radius=15, blur=20, min_opacity=0.4).add_to(mapa)

    # Adiciona marcadores nos top-20 municípios
    top = sorted(contagem.items(), key=lambda x: x[1], reverse=True)[:20]
    for m, n in top:
        coord = cache.get(m)
        if coord and isinstance(coord, dict):
            folium.Marker(
                location=[coord["lat"], coord["lng"]],
                popup=f"<b>{m}</b><br>{n} suspeitos fortes",
                icon=folium.Icon(color="red" if n > 5 else "orange",
                                   icon="warning-sign"),
            ).add_to(mapa)

    if html_path is None:
        html_path = config.caminho("geografico", "mapa_suspeitos.html")
    mapa.save(str(html_path))
    print(f"[geo] mapa salvo em {html_path} ({len(pontos)} pontos, "
          f"abra no navegador)")
    return html_path


@com_gc
def executar():
    """Pipeline: NER + Geocoding + Mapa."""
    extrair_entidades()
    geocodificar_municipios(amostra=200)
    return mapa_suspeitos()
