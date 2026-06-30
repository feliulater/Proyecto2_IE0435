from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


MODELS_DIR = Path("results/models")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")


def temporal_train_test_split(
    df_model,
    feature_columns,
    target_column="demanda_real_mw",
    test_size=0.2,
):
    """
    Divide los datos respetando el orden temporal.
    """

    df_sorted = df_model.sort_values("fecha_hora").reset_index(drop=True)

    split_index = int(len(df_sorted) * (1 - test_size))

    df_train = df_sorted.iloc[:split_index].copy()
    df_test = df_sorted.iloc[split_index:].copy()

    X_train = df_train[feature_columns]
    y_train = df_train[target_column]

    X_test = df_test[feature_columns]
    y_test = df_test[target_column]

    return df_train, df_test, X_train, X_test, y_train, y_test


def train_decision_tree_model(
    X_train,
    y_train,
    max_depth=8,
    random_state=42,
):
    """
    Entrena un modelo de árbol de regresión.
    """

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model


def create_hourly_historical_baseline(
    df_train,
    df_test,
    target_column="demanda_real_mw",
):
    """
    Crea una línea base usando el promedio histórico por hora y minuto.
    """

    hourly_mean = (
        df_train
        .groupby(["hora", "minuto"])[target_column]
        .mean()
    )

    global_mean = df_train[target_column].mean()

    predictions = []

    for _, row in df_test.iterrows():
        key = (row["hora"], row["minuto"])

        if key in hourly_mean.index:
            predictions.append(hourly_mean.loc[key])
        else:
            predictions.append(global_mean)

    return np.array(predictions)


def calculate_metrics(y_true, y_pred, model_name):
    """
    Calcula MAE, RMSE y MAPE.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)

    mape = np.mean(
        np.abs((y_true_array - y_pred_array) / y_true_array)
    ) * 100

    return {
        "modelo": model_name,
        "MAE_MW": mae,
        "RMSE_MW": rmse,
        "MAPE_%": mape,
    }


def evaluate_models(
    df_train,
    df_test,
    X_test_with_programmed,
    X_test_without_programmed,
    y_test,
    model_with_programmed,
    model_without_programmed,
    target_column="demanda_real_mw",
):
    """
    Evalúa:
    - Árbol de regresión con demanda programada.
    - Árbol de regresión sin demanda programada.
    - Demanda programada CENCE.
    - Promedio histórico horario.
    """

    tree_with_programmed_predictions = model_with_programmed.predict(
        X_test_with_programmed
    )

    tree_without_programmed_predictions = model_without_programmed.predict(
        X_test_without_programmed
    )

    programmed_predictions = df_test["demanda_programada_mw"].values

    hourly_baseline_predictions = create_hourly_historical_baseline(
        df_train,
        df_test,
        target_column=target_column,
    )

    metrics = [
        calculate_metrics(
            y_test,
            tree_with_programmed_predictions,
            "Árbol con demanda programada",
        ),
        calculate_metrics(
            y_test,
            tree_without_programmed_predictions,
            "Árbol sin demanda programada",
        ),
        calculate_metrics(
            y_test,
            programmed_predictions,
            "Demanda programada CENCE",
        ),
        calculate_metrics(
            y_test,
            hourly_baseline_predictions,
            "Promedio histórico horario",
        ),
    ]

    metrics_df = pd.DataFrame(metrics)

    results_df = df_test[
        ["fecha_hora", target_column, "demanda_programada_mw"]
    ].copy()

    results_df = results_df.rename(
        columns={target_column: "demanda_real_mw"}
    )

    results_df["pred_arbol_con_programada_mw"] = (
        tree_with_programmed_predictions
    )

    results_df["pred_arbol_sin_programada_mw"] = (
        tree_without_programmed_predictions
    )

    results_df["pred_promedio_historico_mw"] = (
        hourly_baseline_predictions
    )

    return metrics_df, results_df


def save_metrics(metrics_df, filename="metricas_modelos.csv"):
    """
    Guarda las métricas en results/metrics.
    """

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = METRICS_DIR / filename
    metrics_df.to_csv(output_path, index=False)

    print(f"\nMétricas guardadas en: {output_path}")

    return output_path


def save_model(model, filename):
    """
    Guarda un modelo entrenado.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = MODELS_DIR / filename
    joblib.dump(model, output_path)

    print(f"Modelo guardado en: {output_path}")

    return output_path


def plot_model_predictions(results_df):
    """
    Grafica demanda real contra las predicciones.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))

    plt.plot(
        results_df["fecha_hora"],
        results_df["demanda_real_mw"],
        label="Demanda real",
        linewidth=2,
    )

    plt.plot(
        results_df["fecha_hora"],
        results_df["pred_arbol_con_programada_mw"],
        label="Árbol con demanda programada",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        results_df["fecha_hora"],
        results_df["pred_arbol_sin_programada_mw"],
        label="Árbol sin demanda programada",
        linestyle="-.",
        linewidth=2,
    )

    plt.plot(
        results_df["fecha_hora"],
        results_df["demanda_programada_mw"],
        label="Demanda programada CENCE",
        linestyle=":",
        linewidth=2,
    )

    plt.plot(
        results_df["fecha_hora"],
        results_df["pred_promedio_historico_mw"],
        label="Promedio histórico horario",
        linewidth=1.5,
    )

    plt.title("Comparación de predicciones de demanda eléctrica")
    plt.xlabel("Fecha")
    plt.ylabel("Demanda [MW]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "comparacion_predicciones.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"\nFigura de predicciones guardada en: {output_path}")

    return output_path


def plot_feature_importance(
    model,
    feature_columns,
    title,
    filename,
):
    """
    Grafica la importancia de variables del árbol de regresión.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {
            "variable": feature_columns,
            "importancia": model.feature_importances_,
        }
    ).sort_values("importancia", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["variable"],
        importance_df["importancia"],
    )

    plt.title(title)
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"\nFigura de importancia de variables guardada en: {output_path}")

    return output_path


def show_model_summary(metrics_df):
    """
    Muestra resumen de desempeño de los modelos.
    """

    print("\nResumen de desempeño de modelos")
    print("-" * 50)
    print(metrics_df.to_string(index=False))

    best_model = metrics_df.sort_values("RMSE_MW").iloc[0]

    print("\nMejor modelo según RMSE:")
    print(f"{best_model['modelo']} con RMSE = {best_model['RMSE_MW']:.3f} MW")