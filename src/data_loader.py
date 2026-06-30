from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")


def get_data_files():
    """
    Busca archivos de datos dentro de la carpeta data/raw.
    Acepta archivos CSV, TXT y Excel.
    """
    valid_extensions = ["*.csv", "*.txt", "*.xlsx", "*.xls"]

    files = []
    for extension in valid_extensions:
        files.extend(RAW_DATA_DIR.glob(extension))

    return files


def load_data(file_path=None):
    """
    Carga un archivo CSV, TXT o Excel.

    Si no se indica file_path, carga automáticamente el primer archivo
    encontrado en data/raw.
    """
    if file_path is None:
        files = get_data_files()

        if not files:
            raise FileNotFoundError(
                "No se encontraron archivos en data/raw. "
                "Coloque ahí un archivo CSV, TXT o Excel con los datos de demanda."
            )

        file_path = files[0]

    file_path = Path(file_path)

    if file_path.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            "Formato no soportado. Use archivos .csv, .txt, .xlsx o .xls."
        )

    return df


def show_data_summary(df):
    """
    Muestra información básica del conjunto de datos.
    """
    print("\nResumen del archivo cargado")
    print("-" * 40)

    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    print("\nColumnas encontradas:")
    for column in df.columns:
        print(f"- {column}")

    print("\nPrimeras filas:")
    print(df.head())

    print("\nInformación general:")
    print(df.info())