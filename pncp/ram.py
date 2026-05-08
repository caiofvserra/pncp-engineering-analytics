"""
Utilitários de memória.

O Colab Free dá ~12GB de RAM. Cada etapa do pipeline deve liberar
o que não precisa antes de retornar. Aqui estão os helpers para isso.
"""

import gc
import functools


def monitorar_ram(rotulo=""):
    """Imprime uso atual de RAM. Silencioso se psutil não estiver instalado."""
    try:
        import psutil
        proc = psutil.Process()
        usado = proc.memory_info().rss / (1024 ** 3)
        total = psutil.virtual_memory().total / (1024 ** 3)
        print(f"[RAM] {rotulo}: {usado:.2f}GB / {total:.2f}GB")
    except ImportError:
        pass


def liberar(*objetos):
    """Apaga referências e força gc. Use no fim de cada etapa pesada."""
    for _ in objetos:
        del _
    gc.collect()


def com_gc(funcao):
    """
    Decorator: força gc.collect() depois da função, mesmo em erro.
    Útil para envolver etapas pesadas (treino, embeddings, PDFs).
    """
    @functools.wraps(funcao)
    def wrapper(*args, **kwargs):
        try:
            return funcao(*args, **kwargs)
        finally:
            gc.collect()
    return wrapper


def precisa_de(caminho, etapa, hint=None):
    """
    Helper de pré-condição. Retorna True se o arquivo existe.
    Caso contrário imprime aviso amigável e devolve False — a função
    chamadora deve fazer `return None` em seguida.

    Permite "Run all" no notebook sem que uma etapa faltando derrube
    todas as outras.
    """
    from pathlib import Path
    if Path(caminho).exists():
        return True
    print(f"  ⚠ [{etapa}] pulando — arquivo necessário não existe: {caminho}")
    if hint:
        print(f"     {hint}")
    return False
