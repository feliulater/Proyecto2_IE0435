from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


FIGURES_DIR = Path("results/figures")


def detect_anomalies_rolling(
    df,
    column="demanda_real_mw",
    window=12,
    threshold=3.0
):
    """
    Detecta anomalías usando media móvil y desviación estándar móvil.

    Parámetros:
    - df: DataFrame con los datos.
    - column: columna donde se buscarán anomalías.
    - window: tamaño de ventana móvil.
      Como los datos son cada 15 minutos, window=12 equivale a 3 horas.
    - threshold: número de desviaciones estándar permitido.

    Retorna:
    - DataFrame con columnas adicionales:
      media_movil, std_movil, limite_superior, limite_inferior, es_anomalia.
    """

    df_anom = df.copy()

    df_anom["media_movil"] = (
        df_anom[column]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )

    df_anom["std_movil"] = (
        df_anom[column]
        .rolling(window=window, center=True, min_periods=1)
        .std()
        .fillna(0)
    )

    df_anom["limite_superior"] = (
        df_anom["media_movil"] + threshold * df_anom["std_movil"]
    )

    df_anom["limite_inferior"] = (
        df_anom["media_movil"] - threshold * df_anom["std_movil"]
    )

    df_anom["es_anomalia"] = (
        (df_anom[column] > df_anom["limite_superior"])
        | (df_anom[column] < df_anom["limite_inferior"])
    )

    return df_anom


def correct_anomalies(df_anom, column="demanda_real_mw"):
    """
    Corrige anomalías reemplazándolas temporalmente por NaN
    y luego aplicando interpolación.
    """

    df_corrected = df_anom.copy()

    corrected_column = f"{column}_corregida"

    df_corrected[corrected_column] = df_corrected[column]

    df_corrected.loc[
        df_corrected["es_anomalia"],
        corrected_column
    ] = pd.NA

    df_corrected[corrected_column] = (
        df_corrected[corrected_column]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    return df_corrected


def show_anomaly_summary(df_anom):
    """
    Muestra resumen de anomalías detectadas.
    """

    total = len(df_anom)
    anomalies = int(df_anom["es_anomalia"].sum())
    percentage = (anomalies / total) * 100 if total > 0 else 0

    print("\nResumen de detección de anomalías")
    print("-" * 45)
    print(f"Total de datos: {total}")
    print(f"Anomalías detectadas: {anomalies}")
    print(f"Porcentaje de anomalías: {percentage:.2f}%")

    if anomalies > 0:
        print("\nPrimeras anomalías detectadas:")
        print(
            df_anom.loc[
                df_anom["es_anomalia"],
                ["fecha_hora", "demanda_real_mw", "media_movil"]
            ].head()
        )


def plot_anomalies(df_anom, column="demanda_real_mw"):
    """
    Grafica la demanda real y marca las anomalías detectadas.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    anomalies = df_anom[df_anom["es_anomalia"]]

    plt.figure(figsize=(14, 6))

    plt.plot(
        df_anom["fecha_hora"],
        df_anom[column],
        label="Demanda real",
        linewidth=1.8,
    )

    plt.scatter(
        anomalies["fecha_hora"],
        anomalies[column],
        label="Anomalías detectadas",
        marker="o",
        s=50,
    )

    plt.title("Detección de anomalías en demanda eléctrica")
    plt.xlabel("Fecha")
    plt.ylabel("Demanda [MW]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "anomalias_demanda.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"\nFigura de anomalías guardada en: {output_path}")

    return output_path


def plot_corrected_anomalies(df_corrected, column="demanda_real_mw"):
    """
    Compara la demanda original con la demanda corregida.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    corrected_column = f"{column}_corregida"

    plt.figure(figsize=(14, 6))

    plt.plot(
        df_corrected["fecha_hora"],
        df_corrected[column],
        label="Demanda original",
        linewidth=1.5,
    )

    plt.plot(
        df_corrected["fecha_hora"],
        df_corrected[corrected_column],
        label="Demanda corregida",
        linestyle="--",
        linewidth=1.8,
    )

    plt.title("Corrección de anomalías en demanda eléctrica")
    plt.xlabel("Fecha")
    plt.ylabel("Demanda [MW]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "demanda_corregida.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"\nFigura de corrección guardada en: {output_path}")

    return output_path