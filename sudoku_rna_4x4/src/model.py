"""
# Modelo de RNA multicamadas

Este módulo encapsula a criação, o treinamento e a avaliação da rede neural.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def create_mlp(random_seed: int = 42) -> MLPClassifier:
    """
    ## Criação da RNA

    A arquitetura escolhida é uma MLP com três camadas ocultas.
    A entrada possui 64 neurônios, porque cada tabuleiro 4x4 é codificado
    como 16 células vezes 4 símbolos possíveis.
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=random_seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
    )


def train_model(X: np.ndarray, y: np.ndarray, random_seed: int = 42) -> Tuple[MLPClassifier, Dict[str, object]]:
    """
    ## Treinamento e teste

    Divide o dataset em treino e teste, treina a RNA e retorna métricas.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_seed,
        stratify=y,
    )

    model = create_mlp(random_seed=random_seed)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics: Dict[str, object] = {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
        "loss_curve": model.loss_curve_,
        "X_train_size": len(X_train),
        "X_test_size": len(X_test),
    }

    return model, metrics
