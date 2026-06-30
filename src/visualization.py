from pathlib import Path
import matplotlib.pyplot as plt


FIGURES_DIR = Path("results/figures")


def plot_real_vs_programmed(df):
    """
    Grafica la demanda real y la demanda programada.
    Guarda la figura en results/figures.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["fecha_hora"],
        df["demanda_real_mw"],
        label="Demanda real",
        linewidth=2,
    )

    plt.plot(
        df["fecha_hora"],
        df["demanda_programada_mw"],
        label="Demanda programada",
        linewidth=2,
        linestyle="--",
    )

    plt.title("Demanda eléctrica real vs programada")
    plt.xlabel("Hora")
    plt.ylabel("Demanda [MW]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "demanda_real_vs_programada.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"\nFigura guardada en: {output_path}")

    return output_path