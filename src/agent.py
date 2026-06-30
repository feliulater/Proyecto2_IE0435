from pathlib import Path

import pandas as pd

from src.data_downloader import download_cence_range, save_raw_data
from src.data_loader import load_data, show_data_summary
from src.preprocessing import (
    preprocess_demand_data,
    show_preprocessing_summary,
    save_processed_data,
)
from src.visualization import plot_real_vs_programmed
from src.anomaly_detection import (
    detect_anomalies_rolling,
    correct_anomalies,
    show_anomaly_summary,
    plot_anomalies,
    plot_corrected_anomalies,
)
from src.feature_engineering import (
    create_model_dataset,
    show_feature_summary,
    save_model_dataset,
)
from src.models import (
    temporal_train_test_split,
    train_decision_tree_model,
    evaluate_models,
    save_metrics,
    save_model,
    plot_model_predictions,
    plot_feature_importance,
    show_model_summary,
)


METRICS_PATH = Path("results/metrics/metricas_modelos.csv")
PREDICTIONS_PATH = Path("results/metrics/predicciones_modelos.csv")


class DemandForecastAgent:
    """
    Agente orquestador para el proyecto de predicción de demanda eléctrica.

    El agente coordina varias herramientas:
    - descarga de datos del CENCE;
    - carga y preprocesamiento;
    - visualización;
    - detección y corrección de anomalías;
    - ingeniería de características;
    - entrenamiento y evaluación de modelos.
    """

    def download_data_tool(self, start_date, end_date):
        """
        Herramienta para descargar datos del CENCE.
        """

        print("\nHerramienta: descarga de datos CENCE")
        print("-" * 50)

        df = download_cence_range(start_date, end_date)
        output_path = save_raw_data(df, start_date, end_date)

        print("\nDescarga completada.")
        print(f"Archivo generado: {output_path}")

        return output_path

    def prepare_data_tool(
        self,
        show_details=True,
        plot_initial=True,
        detect_anomalies=True,
        plot_anomaly_results=True,
    ):
        """
        Herramienta para cargar, limpiar y preparar datos.
        """

        print("\nHerramienta: preparación de datos")
        print("-" * 50)

        print("\n1. Carga inicial de datos")
        df = load_data()

        if show_details:
            show_data_summary(df)

        print("\n2. Preprocesamiento de datos")
        df_clean = preprocess_demand_data(df)

        if show_details:
            show_preprocessing_summary(df_clean)

        save_processed_data(df_clean)

        if plot_initial:
            print("\n3. Visualización inicial")
            plot_real_vs_programmed(df_clean)

        if detect_anomalies:
            print("\n4. Detección de anomalías")
            df_anom = detect_anomalies_rolling(
                df_clean,
                column="demanda_real_mw",
                window=12,
                threshold=2.0,
            )

            show_anomaly_summary(df_anom)

            if plot_anomaly_results:
                plot_anomalies(df_anom)

            print("\n5. Corrección de anomalías")
            df_corrected = correct_anomalies(
                df_anom,
                column="demanda_real_mw",
            )

            if plot_anomaly_results:
                plot_corrected_anomalies(df_corrected)

        else:
            df_corrected = df_clean.copy()

        # Usar la señal corregida como demanda principal si existe.
        # Si no hubo anomalías, será equivalente a la original.
        corrected_column = "demanda_real_mw_corregida"

        if corrected_column in df_corrected.columns:
            df_for_model = df_corrected.copy()
            df_for_model["demanda_real_mw_original"] = df_for_model["demanda_real_mw"]
            df_for_model["demanda_real_mw"] = df_for_model[corrected_column]
        else:
            df_for_model = df_corrected.copy()

        print("\n6. Ingeniería de características")
        df_model, X, y, feature_columns = create_model_dataset(
            df_for_model,
            target_column="demanda_real_mw",
        )

        show_feature_summary(df_model, feature_columns)
        save_model_dataset(df_model)

        return df_model, feature_columns

    def train_models_tool(self, df_model, feature_columns):
        """
        Herramienta para entrenar y evaluar modelos de predicción.
        """

        print("\nHerramienta: entrenamiento y evaluación de modelos")
        print("-" * 50)

        feature_columns_with_programmed = feature_columns

        feature_columns_without_programmed = [
            column for column in feature_columns
            if column != "demanda_programada_mw"
        ]

        df_train, df_test, X_train_with, X_test_with, y_train, y_test = (
            temporal_train_test_split(
                df_model,
                feature_columns_with_programmed,
                target_column="demanda_real_mw",
                test_size=0.2,
            )
        )

        X_train_without = df_train[feature_columns_without_programmed]
        X_test_without = df_test[feature_columns_without_programmed]

        model_with_programmed = train_decision_tree_model(
            X_train_with,
            y_train,
            max_depth=8,
            random_state=42,
        )

        model_without_programmed = train_decision_tree_model(
            X_train_without,
            y_train,
            max_depth=8,
            random_state=42,
        )

        metrics_df, results_df = evaluate_models(
            df_train,
            df_test,
            X_test_with,
            X_test_without,
            y_test,
            model_with_programmed,
            model_without_programmed,
            target_column="demanda_real_mw",
        )

        show_model_summary(metrics_df)
        save_metrics(metrics_df)

        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Predicciones guardadas en: {PREDICTIONS_PATH}")

        save_model(
            model_with_programmed,
            filename="arbol_con_demanda_programada.joblib",
        )

        save_model(
            model_without_programmed,
            filename="arbol_sin_demanda_programada.joblib",
        )

        plot_model_predictions(results_df)

        plot_feature_importance(
            model_with_programmed,
            feature_columns_with_programmed,
            title="Importancia de variables - Árbol con demanda programada",
            filename="importancia_arbol_con_programada.png",
        )

        plot_feature_importance(
            model_without_programmed,
            feature_columns_without_programmed,
            title="Importancia de variables - Árbol sin demanda programada",
            filename="importancia_arbol_sin_programada.png",
        )

        return metrics_df, results_df

    def full_pipeline_tool(self):
        """
        Herramienta que ejecuta el flujo completo del proyecto.
        """

        print("\nEjecutando flujo completo del agente")
        print("=" * 60)

        df_model, feature_columns = self.prepare_data_tool(
            show_details=True,
            plot_initial=True,
            detect_anomalies=True,
            plot_anomaly_results=True,
        )

        metrics_df, results_df = self.train_models_tool(
            df_model,
            feature_columns,
        )

        print("\nFlujo completo finalizado correctamente.")

        return metrics_df, results_df

    def anomaly_tool_only(self):
        """
        Herramienta enfocada solo en detección y corrección de anomalías.
        """

        print("\nEjecutando herramienta de anomalías")
        print("=" * 60)

        df = load_data()
        df_clean = preprocess_demand_data(df)

        df_anom = detect_anomalies_rolling(
            df_clean,
            column="demanda_real_mw",
            window=12,
            threshold=2.0,
        )

        show_anomaly_summary(df_anom)
        plot_anomalies(df_anom)

        df_corrected = correct_anomalies(
            df_anom,
            column="demanda_real_mw",
        )

        plot_corrected_anomalies(df_corrected)

        print("\nHerramienta de anomalías finalizada.")

        return df_corrected

    def model_tool_only(self):
        """
        Herramienta enfocada en preparar datos y entrenar modelos.
        No genera todas las gráficas iniciales para ahorrar tiempo.
        """

        print("\nEjecutando herramienta de modelado")
        print("=" * 60)

        df_model, feature_columns = self.prepare_data_tool(
            show_details=False,
            plot_initial=False,
            detect_anomalies=True,
            plot_anomaly_results=False,
        )

        metrics_df, results_df = self.train_models_tool(
            df_model,
            feature_columns,
        )

        print("\nHerramienta de modelado finalizada.")

        return metrics_df, results_df

    def show_best_model_tool(self):
        """
        Herramienta para mostrar el mejor modelo según métricas guardadas.
        """

        print("\nHerramienta: consulta de mejor modelo")
        print("-" * 50)

        if not METRICS_PATH.exists():
            print("No existe archivo de métricas todavía.")
            print("Ejecute primero el flujo completo o la herramienta de modelado.")
            return None

        metrics_df = pd.read_csv(METRICS_PATH)

        print("\nMétricas disponibles:")
        print(metrics_df.to_string(index=False))

        best_model = metrics_df.sort_values("RMSE_MW").iloc[0]

        print("\nMejor modelo según RMSE:")
        print(f"Modelo: {best_model['modelo']}")
        print(f"MAE: {best_model['MAE_MW']:.3f} MW")
        print(f"RMSE: {best_model['RMSE_MW']:.3f} MW")
        print(f"MAPE: {best_model['MAPE_%']:.3f} %")

        return best_model


def interactive_agent():
    """
    Interfaz interactiva del agente en consola.
    """

    agent = DemandForecastAgent()

    while True:
        print("\n" + "=" * 60)
        print("AGENTE DE PREDICCIÓN DE DEMANDA ELÉCTRICA")
        print("=" * 60)
        print("1. Ejecutar flujo completo")
        print("2. Descargar datos del CENCE")
        print("3. Ejecutar detección/corrección de anomalías")
        print("4. Entrenar y evaluar modelos")
        print("5. Mostrar mejor modelo guardado")
        print("0. Salir")

        option = input("\nSeleccione una opción: ").strip()

        if option == "1":
            agent.full_pipeline_tool()

        elif option == "2":
            start_date = input("Fecha inicial (YYYY-MM-DD): ").strip()
            end_date = input("Fecha final   (YYYY-MM-DD): ").strip()
            agent.download_data_tool(start_date, end_date)

        elif option == "3":
            agent.anomaly_tool_only()

        elif option == "4":
            agent.model_tool_only()

        elif option == "5":
            agent.show_best_model_tool()

        elif option == "0":
            print("Saliendo del agente.")
            break

        else:
            print("Opción no válida. Intente de nuevo.")