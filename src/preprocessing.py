from pathlib import Path
import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")


def preprocess_demand_data(df):
    """
    Limpia y prepara los datos de demanda eléctrica.

    Operaciones realizadas:
    - Copia el DataFrame original.
    - Convierte fechaHora a formato datetime.
    - Ordena los datos por fecha y hora.
    - Renombra columnas para mayor claridad.
    - Elimina duplicados.
    - Revisa valores faltantes.
    """

    df_clean = df.copy()

    required_columns = ["fechaHora", "MW", "MW_P"]
    for column in required_columns:
        if column not in df_clean.columns:
            raise ValueError(f"Falta la columna requerida: {column}")

    df_clean["fechaHora"] = pd.to_datetime(df_clean["fechaHora"])

    df_clean = df_clean.rename(
        columns={
            "fechaHora": "fecha_hora",
            "MW": "demanda_real_mw",
            "MW_P": "demanda_programada_mw",
        }
    )

    df_clean = df_clean.sort_values("fecha_hora")
    df_clean = df_clean.drop_duplicates(subset=["fecha_hora"])

    df_clean = df_clean.reset_index(drop=True)

    return df_clean


def show_preprocessing_summary(df_clean):
    """
    Muestra un resumen después del preprocesamiento.
    """

    print("\nResumen después del preprocesamiento")
    print("-" * 45)

    print(f"Filas: {df_clean.shape[0]}")
    print(f"Columnas: {df_clean.shape[1]}")

    print("\nTipos de datos:")
    print(df_clean.dtypes)

    print("\nValores faltantes:")
    print(df_clean.isna().sum())

    print("\nPrimeras filas limpias:")
    print(df_clean.head())

    print("\nÚltimas filas limpias:")
    print(df_clean.tail())


def save_processed_data(df_clean, filename="demanda_limpia.csv"):
    """
    Guarda los datos procesados en data/processed.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / filename
    df_clean.to_csv(output_path, index=False)

    print(f"\nDatos procesados guardados en: {output_path}")

    return output_path