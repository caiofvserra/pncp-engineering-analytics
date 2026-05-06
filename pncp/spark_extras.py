"""
Extras com PySpark — APENAS para escalas muito grandes (multi-UF, milhões
de linhas). Para 200–300k contratos em Colab single-node, **pandas + parquet
é estritamente melhor** (sem overhead de JVM). Este módulo é um stub para o
caso de o usuário escalar para todo o Brasil no futuro.

Uso (opcional):
    from pncp.spark_extras import sessao, ler_contratos_spark
    spark = sessao()
    sdf = ler_contratos_spark(spark, "dados/coleta/contratos.parquet")

Para instalar (Colab):
    !pip install pyspark
"""

from typing import Optional


def sessao(app_name="pncp-tcc", memoria="4g"):
    """Cria SparkSession local. Erro claro se PySpark não estiver instalado."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise RuntimeError(
            "PySpark não instalado. `pip install pyspark` (~250MB). "
            "Para 300k contratos isso provavelmente é desnecessário — "
            "pandas+parquet do pncp/ basta."
        )
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.driver.memory", memoria)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate())


def ler_contratos_spark(spark, caminho):
    """Lê parquet com Spark."""
    return spark.read.parquet(str(caminho))


def estatisticas_spark(sdf):
    """Exemplo: distribuição por rótulo via Spark."""
    return sdf.groupBy("rotulo").count().toPandas()


def converter_pandas_to_spark(spark, df):
    """Conversão direta (com Arrow ativo no spark)."""
    return spark.createDataFrame(df)
