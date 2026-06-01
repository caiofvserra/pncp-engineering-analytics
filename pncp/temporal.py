"""
Análise temporal de contratos PNCP.

Baseado nos notebooks 03 (Decomposição) e 11 (Outras tarefas — Change
Point Detection) do MBA Inteligência Analítica (Marcacini, ICMC/USP).

Funções:
  - decompor_serie(rotulo)         — tendência + sazonalidade + ruído
  - detectar_change_points(rotulo) — momentos de quebra estrutural
  - forecast_volume(rotulo, n)     — projeção via Prophet/ARIMA
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pncp import config
from pncp._plot import salvar_e_mostrar
from pncp.io_disco import ler_parquet, salvar_json
from pncp.ram import com_gc


def _serie_mensal(rotulo=None):
    """Constrói série temporal mensal de contagem de contratos."""
    df = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"),
                     colunas=["anoPublicacao", "mesPublicacao", "rotulo"])
    if df.empty:
        return None
    df = df.dropna(subset=["anoPublicacao", "mesPublicacao"]).copy()
    df["anoPublicacao"] = df["anoPublicacao"].astype(int)
    df["mesPublicacao"] = df["mesPublicacao"].astype(int)
    if rotulo:
        df = df[df["rotulo"] == rotulo]
    if df.empty:
        return None
    df["periodo"] = pd.to_datetime(
        df["anoPublicacao"].astype(str) + "-"
        + df["mesPublicacao"].astype(str).str.zfill(2) + "-01"
    )
    s = df.groupby("periodo").size().sort_index()
    s = s.asfreq("MS", fill_value=0)
    return s


@com_gc
def decompor_serie(rotulo="geral", modelo="additive"):
    """
    Decomposição STL/clássica: tendência + sazonalidade + ruído.
    Notebook 03 do curso. Útil para detectar se a sazonalidade dos
    contratos 'geral' segue padrão diferente de 'engenharia' (sinal
    indireto de mudança de comportamento).
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        print("[temporal] instale statsmodels: pip install statsmodels")
        return None

    s = _serie_mensal(rotulo)
    if s is None or len(s) < 24:
        print(f"[temporal] série de '{rotulo}' tem menos de 24 meses — "
              f"não dá pra decompor sazonalidade")
        return None

    decomp = seasonal_decompose(s, model=modelo, period=12, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    s.plot(ax=axes[0], color="#1f77b4")
    axes[0].set_title(f"Série original (contratos 'rotulo={rotulo}' por mês)")
    decomp.trend.plot(ax=axes[1], color="#2ca02c")
    axes[1].set_title("Tendência (média móvel ~12m)")
    decomp.seasonal.plot(ax=axes[2], color="#ff7f0e")
    axes[2].set_title("Sazonalidade (componente mensal)")
    decomp.resid.plot(ax=axes[3], color="#d62728")
    axes[3].set_title("Ruído (diferença não-explicada)")
    plt.suptitle(f"Decomposição STL — {rotulo}", fontweight="bold", y=1.01)

    saida = config.caminho("temporal", f"decomposicao_{rotulo}.png")
    return salvar_e_mostrar(fig, saida)


@com_gc
def detectar_change_points(rotulo="geral", n_pontos=4):
    """
    Detecta momentos de quebra estrutural na série temporal.
    Notebook 11 (Outras tarefas — ruptures).

    Útil para identificar SE houve mudança brusca de critério de
    cadastro (ex: pré e pós algum decreto/portaria).
    """
    try:
        import ruptures as rpt
    except ImportError:
        print("[temporal] instale ruptures: pip install ruptures")
        return None

    s = _serie_mensal(rotulo)
    if s is None or len(s) < 12:
        print("[temporal] série muito curta")
        return None

    algo = rpt.Pelt(model="rbf").fit(s.values.reshape(-1, 1))
    breakpoints = algo.predict(pen=10)[:-1]   # último ponto é o fim da série

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, s.values, color="#1f77b4", marker="o", markersize=3)
    for bp in breakpoints:
        if bp < len(s):
            data_bp = s.index[bp]
            ax.axvline(data_bp, color="#d62728", linestyle="--", alpha=0.7)
            ax.text(data_bp, ax.get_ylim()[1] * 0.95,
                    str(data_bp.date()), rotation=90, fontsize=8,
                    color="#d62728")
    ax.set_title(f"Change Point Detection — {rotulo} "
                 f"({len(breakpoints)} quebras detectadas)")
    ax.set_xlabel("Período"); ax.set_ylabel("Nº contratos/mês")

    saida = config.caminho("temporal", f"change_points_{rotulo}.png")
    salvar_e_mostrar(fig, saida)

    info = {
        "rotulo": rotulo,
        "n_breakpoints": len(breakpoints),
        "datas_breakpoints": [str(s.index[bp].date())
                                for bp in breakpoints if bp < len(s)],
    }
    salvar_json(info, config.caminho("temporal",
                                       f"change_points_{rotulo}.json"))
    return info


@com_gc
def forecast_volume(rotulo="geral", n_meses=12):
    """
    Projeção da quantidade futura de contratos.
    Notebook 07 (Forecasting). Usa Prophet se instalado, senão ARIMA.

    Útil para o TCC mostrar tendência projetada — quantos contratos
    'geral' são esperados nos próximos N meses.
    """
    s = _serie_mensal(rotulo)
    if s is None or len(s) < 12:
        return None

    forecast = None
    metodo = None

    # Tenta Prophet primeiro (mais robusto sazonal)
    try:
        from prophet import Prophet
        df_prophet = pd.DataFrame({"ds": s.index, "y": s.values})
        modelo = Prophet(yearly_seasonality=True, daily_seasonality=False,
                          weekly_seasonality=False)
        modelo.fit(df_prophet)
        futuro = modelo.make_future_dataframe(periods=n_meses, freq="MS")
        pred = modelo.predict(futuro)
        forecast = pred.tail(n_meses)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        forecast.columns = ["periodo", "previsto", "lower", "upper"]
        metodo = "Prophet"
    except ImportError:
        pass

    # Fallback: ARIMA
    if forecast is None:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            modelo = ARIMA(s, order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12)).fit()
            pred = modelo.get_forecast(steps=n_meses)
            ci = pred.conf_int()
            forecast = pd.DataFrame({
                "periodo": pred.predicted_mean.index,
                "previsto": pred.predicted_mean.values,
                "lower": ci.iloc[:, 0].values,
                "upper": ci.iloc[:, 1].values,
            })
            metodo = "ARIMA"
        except Exception as e:
            print(f"[temporal] forecast falhou: {e}")
            return None

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, s.values, label="Histórico", color="#1f77b4")
    ax.plot(forecast["periodo"], forecast["previsto"],
             label="Previsão", color="#d62728", marker="o", markersize=4)
    ax.fill_between(forecast["periodo"], forecast["lower"], forecast["upper"],
                     color="#d62728", alpha=0.15, label="IC")
    ax.set_title(f"Forecast {n_meses} meses — '{rotulo}' ({metodo})")
    ax.set_ylabel("Nº contratos/mês")
    ax.legend()
    salvar_e_mostrar(fig,
                     config.caminho("temporal", f"forecast_{rotulo}.png"))
    return forecast
