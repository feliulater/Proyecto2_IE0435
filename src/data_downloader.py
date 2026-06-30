from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests


RAW_DATA_DIR = Path("data/raw")

URL_BASE = "https://apps.grupoice.com/CenceWeb/data/sen/csv/DemandaMW"
INTERVALO_MINUTOS = 15


def download_cence_day(date):
    """
    Descarga los datos de demanda eléctrica del CENCE para un día específico.

    Parámetro:
    - date: objeto datetime con la fecha deseada.

    Retorna:
    - DataFrame con los datos del día.
    - None si no se pudieron descargar datos.
    """

    date_str = date.strftime("%Y%m%d")

    url = (
        f"{URL_BASE}"
        f"?intervalo={INTERVALO_MINUTOS}"
        f"&inicio={date_str}"
        f"&fin={date_str}"
    )

    print(f"Descargando: {url}")

    try:
        response = requests.get(url, timeout=20)

        if response.status_code != 200:
            print(f"Error HTTP {response.status_code} para la fecha {date_str}")
            return None

        if len(response.text.strip()) < 50:
            print(f"No se encontraron datos para la fecha {date_str}")
            return None

        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            print(f"Archivo vacío para la fecha {date_str}")
            return None

        return df

    except requests.RequestException as error:
        print(f"Error de conexión para la fecha {date_str}: {error}")
        return None


def download_cence_range(start_date, end_date):
    """
    Descarga datos del CENCE para un rango de fechas.

    Parámetros:
    - start_date: string con formato YYYY-MM-DD.
    - end_date: string con formato YYYY-MM-DD.

    Retorna:
    - DataFrame con todos los días unidos.
    """

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if end < start:
        raise ValueError("La fecha final no puede ser menor que la fecha inicial.")

    dataframes = []
    current_date = start

    while current_date <= end:
        df_day = download_cence_day(current_date)

        if df_day is not None:
            dataframes.append(df_day)

        current_date += timedelta(days=1)

    if not dataframes:
        raise ValueError("No se descargaron datos para el rango indicado.")

    df_all = pd.concat(dataframes, ignore_index=True)

    return df_all


def save_raw_data(df, start_date, end_date):
    """
    Guarda los datos descargados en data/raw.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_name = start_date.replace("-", "")
    end_name = end_date.replace("-", "")

    output_path = RAW_DATA_DIR / f"demanda_cence_{start_name}_to_{end_name}.csv"

    df.to_csv(output_path, index=False)

    print(f"\nDatos descargados guardados en: {output_path}")
    print(f"Filas guardadas: {df.shape[0]}")
    print(f"Columnas guardadas: {list(df.columns)}")

    return output_path


def main():
    """
    Descarga un rango de fechas del CENCE.

    Modifique start_date y end_date según el rango deseado.
    """

    start_date = "2026-06-01"
    end_date = "2026-06-07"

    df = download_cence_range(start_date, end_date)
    save_raw_data(df, start_date, end_date)


if __name__ == "__main__":
    main()