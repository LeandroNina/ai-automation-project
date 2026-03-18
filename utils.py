"""Utility functions for the AI automation pipeline.

This module provides helper functions for loading datasets, preprocessing,
building pipelines and evaluating classification models.  It aims to keep
`main.py` focused on orchestrating the overall workflow.
"""

from __future__ import annotations

import logging
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)

def load_dataset(csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a dataset from a CSV file or use the built‑in Wine dataset.

    Parameters
    ----------
    csv_path : Optional[str]
        Path to a CSV file containing data.  The target label is assumed to
        reside in the last column.  If ``None``, the Wine dataset from
        ``scikit‑learn`` is loaded.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the feature matrix ``X`` and target vector ``y``.
    """
    if csv_path is None:
        wine = load_wine(as_frame=True)
        X = wine.data
        y = wine.target
        logger.info("Loaded built‑in Wine dataset with %d samples and %d features.", X.shape[0], X.shape[1])
    else:
        df = pd.read_csv(csv_path)
        # assume last column is target
        if df.shape[1] < 2:
            raise ValueError("CSV must contain at least two columns (features and target).")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        logger.info("Loaded custom dataset from %s with %d samples and %d features.", csv_path, X.shape[0], X.shape[1])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construct a preprocessing transformer for numeric and categorical features.

    Numeric features are imputed (mean) and scaled, while categorical features
    are imputed (most frequent) and one‑hot encoded.  If no categorical
    features are present, only numeric preprocessing is applied.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix used to infer column dtypes.

    Returns
    -------
    ColumnTransformer
        A transformer ready to be used in a scikit‑learn pipeline.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No valid feature columns found in the dataset.")

    return ColumnTransformer(transformers)


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[str, float, dict, dict]:
    """Train multiple models and select the best based on accuracy.

    This function splits the data into train and test sets, constructs a
    preprocessing pipeline, fits different classifiers and evaluates them on
    the hold‑out test set.  Currently supported models are:

    * Logistic Regression
    * Random Forest Classifier

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, default=0.25
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Seed used by the random number generator.

    Returns
    -------
    Tuple[str, float, dict, dict]
        A tuple containing:
        * Name of the best model
        * Accuracy score of the best model
        * Classification report (dictionary) of the best model
        * Confusion matrix (dictionary with keys 'labels', 'matrix')
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
    }

    best_model_name = None
    best_accuracy = -np.inf
    best_report = None
    best_confusion = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        logger.info("Training %s ...", name)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        logger.info("%s accuracy: %.4f", name, acc)
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_report = classification_report(y_test, predictions, output_dict=True)
            # prepare confusion matrix as a dictionary for easy JSON serialisation
            cm = confusion_matrix(y_test, predictions)
            labels = np.unique(y)
            best_confusion = {
                "labels": labels.tolist(),
                "matrix": cm.tolist(),
            }

    return best_model_name, best_accuracy, best_report, best_confusion
