from pathlib import Path
import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")


def create_time_features(df):
    """
    Crea características temporales a partir de la columna fecha_hora.
    """

    df_features = df.copy()

    df_features["hora"] = df_features["fecha_hora"].dt.hour
    df_features["minuto"] = df_features["fecha_hora"].dt.minute
    df_features["dia_semana"] = df_features["fecha_hora"].dt.dayofweek
    df_features["dia_mes"] = df_features["fecha_hora"].dt.day
    df_features["mes"] = df_features["fecha_hora"].dt.month
    df_features["es_fin_semana"] = (
        df_features["dia_semana"].isin([5, 6]).astype(int)
    )

    return df_features


def create_lag_features(df, column="demanda_real_mw"):
    """
    Crea variables de rezago para usar valores pasados de demanda
    como entradas del modelo.

    Como los datos están cada 15 minutos:
    - lag_1 equivale a 15 minutos antes.
    - lag_4 equivale a 1 hora antes.
    - lag_96 equivale a 1 día antes.
    """

    df_features = df.copy()

    df_features["demanda_lag_1"] = df_features[column].shift(1)
    df_features["demanda_lag_4"] = df_features[column].shift(4)
    df_features["demanda_lag_96"] = df_features[column].shift(96)

    df_features["media_movil_1h"] = (
        df_features[column]
        .shift(1)
        .rolling(window=4, min_periods=1)
        .mean()
    )

    df_features["media_movil_3h"] = (
        df_features[column]
        .shift(1)
        .rolling(window=12, min_periods=1)
        .mean()
    )

    return df_features


def create_model_dataset(df, target_column="demanda_real_mw"):
    """
    Construye el dataset final para entrenamiento.

    Incluye:
    - variables temporales;
    - variables de rezago;
    - promedio móvil;
    - variable objetivo.
    """

    df_model = df.copy()

    df_model = create_time_features(df_model)
    df_model = create_lag_features(df_model, column=target_column)

    df_model = df_model.dropna().reset_index(drop=True)

    feature_columns = [
        "hora",
        "minuto",
        "dia_semana",
        "dia_mes",
        "mes",
        "es_fin_semana",
        "demanda_programada_mw",
        "demanda_lag_1",
        "demanda_lag_4",
        "demanda_lag_96",
        "media_movil_1h",
        "media_movil_3h",
    ]

    X = df_model[feature_columns]
    y = df_model[target_column]

    return df_model, X, y, feature_columns


def show_feature_summary(df_model, feature_columns):
    """
    Muestra un resumen del dataset preparado para el modelo.
    """

    print("\nResumen de ingeniería de características")
    print("-" * 50)

    print(f"Filas disponibles para modelado: {df_model.shape[0]}")
    print(f"Columnas totales: {df_model.shape[1]}")

    print("\nVariables de entrada del modelo:")
    for column in feature_columns:
        print(f"- {column}")

    print("\nPrimeras filas del dataset de modelado:")
    print(df_model.head())


def save_model_dataset(df_model, filename="demanda_features.csv"):
    """
    Guarda el dataset con características en data/processed.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / filename
    df_model.to_csv(output_path, index=False)

    print(f"\nDataset de modelado guardado en: {output_path}")

    return output_path