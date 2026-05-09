"""
Helper de plotagem — detecta automaticamente se está em Jupyter/Colab
e usa o backend certo para que os gráficos APAREÇAM inline.

Antes: cada módulo fazia `matplotlib.use('Agg')` na importação. Isso
gera o PNG no disco mas NUNCA mostra inline no Colab. Resultado: você
roda o pipeline e não vê gráfico nenhum.

Agora: usamos o backend padrão do ambiente. Em Jupyter/Colab, o
`%matplotlib inline` automático mostra os gráficos. Em script comum,
o backend Agg ainda funciona para salvar.
"""

import matplotlib
import matplotlib.pyplot as plt


def em_notebook():
    """True se estamos rodando dentro de Jupyter/Colab."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in ip.config or "google.colab" in str(type(ip))
    except ImportError:
        return False


def configurar():
    """Configura matplotlib para o ambiente atual.

    Em notebook: backend default + inline magic ativo.
    Fora dele: backend Agg (não tenta abrir janela).
    """
    if em_notebook():
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                ip.run_line_magic("matplotlib", "inline")
        except Exception:
            pass
    else:
        # Script sem display — usa Agg para não tentar abrir janela
        if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
            matplotlib.use("Agg")


def salvar_e_mostrar(fig, caminho_destino):
    """
    Salva a figura em disco E mostra inline (se estiver em notebook).

    Use no fim de cada função que gera gráfico:
        salvar_e_mostrar(fig, pasta / "01_distribuicao.png")
    """
    from pathlib import Path
    caminho_destino = Path(caminho_destino)
    caminho_destino.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho_destino, dpi=120, bbox_inches="tight")
    if em_notebook():
        plt.show()       # mostra inline
    plt.close(fig)
    return caminho_destino


# Auto-configura ao importar
configurar()
