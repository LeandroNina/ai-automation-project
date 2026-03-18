# AI Automation: AutoML Pipeline Example

This repository contains a simple example of a data‑science automation pipeline implemented in Python.  
The goal of this project is to demonstrate how you can build an automated machine‑learning workflow that performs the following steps end‑to‑end:

1. **Data ingestion** – Load a tabular dataset from a CSV file.  
   You can pass your own dataset through a command‑line argument or let the script fall back to the built‑in Wine dataset from `scikit‑learn`.
2. **Data preprocessing** – Handle missing values, encode categorical features (if present) and scale numerical features.
3. **Model training and selection** – Train multiple classification models (Logistic Regression and Random Forest) using cross‑validation and select the one with the highest accuracy on a hold‑out test set.
4. **Evaluation and report generation** – Generate a classification report, confusion matrix and simple plots summarising the model’s performance.  
   Results are saved to the `reports/` directory.

## Getting Started

### Prerequisites

* **Python 3.9+**
* The packages listed in `requirements.txt`.  
  You can install them using pip:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

1. Clone this repository or download the contents.
2. (Optional) Prepare your own tabular dataset as a CSV file where the last column contains the target labels.  
   Place the file anywhere on your system and note the path.
3. Execute the pipeline:

```bash
python main.py --csv_path path/to/your/data.csv
```

If you omit the `--csv_path` argument the script automatically uses the Wine dataset included with `scikit‑learn`.

After running, the script prints a summary of the models considered, identifies the best model and writes a report to the `reports/` directory.

## Project Structure

```text
ai_automation_project/
├── README.md          # Project overview and instructions
├── requirements.txt   # Required Python packages
├── main.py            # Entrypoint for running the automated pipeline
├── utils.py           # Helper functions used by the pipeline
└── reports/           # Generated classification reports and plots
```

## Extending the Project

This simple automation pipeline is intended as a starting point.  
Here are a few ideas for extending it:

* Add more models (e.g. Support Vector Machines, Gradient Boosting, XGBoost) and compare them using cross‑validation.
* Integrate hyperparameter optimisation using libraries like `optuna` or `scikit‑optimize`.
* Support regression tasks in addition to classification.
* Replace the built‑in reporting with a richer HTML report using `pandas‑profiling` or `sweetviz`.
* Schedule the pipeline to run periodically (e.g. via a GitHub Action or cron job) to retrain models with new data.

## License

This project is released under the MIT License.  Feel free to fork it and adapt it to your own needs.
