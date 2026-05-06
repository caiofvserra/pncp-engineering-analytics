"""
Smoke test: verifica que o pacote importa e que as funções principais
têm a assinatura esperada. Não baixa dados nem treina modelos.
"""

import sys
from pathlib import Path

import pytest

# Adiciona o pacote ao path se rodando sem instalar
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_principal():
    import pncp
    assert hasattr(pncp, "__version__")


def test_lazy_submodulos():
    import pncp
    for sub in ("config", "io_disco", "ram", "coleta", "texto",
                 "eda", "classificacao", "avancado", "grafos",
                 "cnae", "pdfs", "aditivos", "relatorio"):
        modulo = getattr(pncp, sub)
        assert modulo is not None


def test_config_basico():
    from pncp import config
    assert config.CAT_OBRAS == 7
    assert config.CAT_SERV_GERAIS == 8
    assert config.CAT_SERV_ENG == 9
    assert config.rotular(7) == "obras"
    assert config.rotular(8) == "geral"
    assert config.rotular(9) == "engenharia"


def test_texto_limpar():
    from pncp.texto import limpar
    saida = limpar("Construção de PONTE — manutenção elétrica!")
    assert "construcao" in saida
    assert "ponte" in saida
    assert "—" not in saida


def test_pdfs_decompor_ncp():
    from pncp.pdfs import _decompor_ncp
    r = _decompor_ncp("12345678000199-1-000123/2024")
    assert r is not None
    assert r["cnpj"] == "12345678000199"
    assert r["ano"] == 2024
    assert r["sequencial"] == 123


def test_io_disco_round_trip(tmp_path):
    import pandas as pd
    from pncp.io_disco import salvar_parquet, ler_parquet
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    salvar_parquet(df, tmp_path / "t.parquet")
    out = ler_parquet(tmp_path / "t.parquet")
    assert len(out) == 3


def test_relatorio_glossario(capsys):
    from pncp.relatorio import glossario, GLOSSARIO
    assert "F1" in GLOSSARIO
    glossario("F1")
    captured = capsys.readouterr()
    assert "harm" in captured.out.lower() or "precis" in captured.out.lower()
