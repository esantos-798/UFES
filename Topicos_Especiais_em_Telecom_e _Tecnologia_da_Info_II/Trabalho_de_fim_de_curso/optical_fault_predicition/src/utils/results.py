import json
from pathlib import Path


def save_results(results, filename):

    Path("results").mkdir(
        exist_ok=True
    )

    with open(
        f"results/{filename}",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\nResults saved to results/{filename}"
    )