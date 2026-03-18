"""Entry point for the AI automation pipeline.

This script orchestrates the end‑to‑end machine‑learning workflow:

* Parse command‑line arguments to optionally accept a custom CSV dataset.
* Load and preprocess the data.
* Train multiple classification models and automatically select the best one.
* Generate evaluation reports and visualisations.

Usage:

```bash
python main.py --csv_path path/to/your/dataset.csv
```

If ``--csv_path`` is omitted, the built‑in Wine dataset is used instead.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_dataset, train_and_evaluate


def save_reports(
    reports_dir: Path,
    model_name: str,
    accuracy: float,
    classification_report: dict,
    confusion: dict,
) -> None:
    """Persist evaluation results to the ``reports/`` directory.

    Parameters
    ----------
    reports_dir : Path
        Directory in which to save report files.  Created if it does not exist.
    model_name : str
        Name of the best model.
    accuracy : float
        Accuracy of the best model.
    classification_report : dict
        A dictionary returned by ``sklearn.metrics.classification_report(..., output_dict=True)``.
    confusion : dict
        Dictionary containing keys ``'labels'`` and ``'matrix'`` representing the confusion matrix.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Save a simple summary
    summary_path = reports_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Best model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    # Save classification report as JSON
    report_path = reports_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(classification_report, f, indent=2)

    # Save confusion matrix as heatmap
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()
    heatmap_path = reports_dir / "confusion_matrix.png"
    fig.savefig(heatmap_path)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    """Parse command‑line arguments.  Returns an ``argparse.Namespace``."""
    parser = argparse.ArgumentParser(description="Run the AI automation pipeline")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "Optional path to a CSV file containing your dataset. "
            "If omitted, the built‑in Wine dataset is used. The target label must "
            "be the last column in the CSV."
        ),
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_arguments()
    X, y = load_dataset(args.csv_path)
    best_model_name, best_accuracy, report, confusion = train_and_evaluate(X, y)
    logging.info("Best model: %s (Accuracy: %.4f)", best_model_name, best_accuracy)
    save_reports(Path(__file__).parent / "reports", best_model_name, best_accuracy, report, confusion)
    logging.info("Reports saved to the 'reports' directory.")


if __name__ == "__main__":
    main()
