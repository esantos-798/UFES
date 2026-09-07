import matplotlib.pyplot as plt

def plot_model_comparison(df):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC"
    ]

    for metric in metrics:

        plt.figure(figsize=(6,4))

        bars = plt.bar(df["Model"], df[metric])

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.title(metric)

        plt.ylim(
            df[metric].min() - 0.01,
            min(1.0, df[metric].max() + 0.005)
        )

        plt.grid(axis="y")

        plt.tight_layout()

        plt.savefig(
            f"experiments/{metric.lower()}.png",
            dpi=300
        )

        plt.show()