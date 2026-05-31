"""
# Geração do dataset

Este módulo cria exemplos positivos e negativos para treinar a RNA.
Os exemplos positivos são soluções Sudoku 4x4 válidas.
Os exemplos negativos são tabuleiros completos que violam alguma regra.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from sudoku_rules import Board, N, SYMBOLS, is_safe, is_valid_complete_board


def board_to_one_hot(board: Board) -> np.ndarray:
    """
    ## Codificação one-hot

    Converte uma matriz 4x4 em um vetor de 64 posições.
    Como a RNA trabalha com números, a representação categórica é convertida
    em vetores binários.
    """
    encoded = []
    for row in range(N):
        for col in range(N):
            value = board[row][col]
            for symbol in sorted(SYMBOLS):
                encoded.append(1 if value == symbol else 0)
    return np.array(encoded, dtype=float)


def generate_all_valid_solutions() -> List[Board]:
    """
    ## Geração de soluções válidas

    Usa backtracking para enumerar todos os tabuleiros Sudoku 4x4 válidos.
    No 4x4 isso é viável e produz o conjunto positivo do treinamento.
    """
    board = [[0 for _ in range(N)] for _ in range(N)]
    solutions: List[Board] = []

    def backtrack(cell_index: int = 0) -> None:
        if cell_index == N * N:
            solutions.append([row[:] for row in board])
            return

        row = cell_index // N
        col = cell_index % N
        values = list(SYMBOLS)
        random.shuffle(values)

        for value in values:
            if is_safe(board, row, col, value):
                board[row][col] = value
                backtrack(cell_index + 1)
                board[row][col] = 0

    backtrack()
    return solutions


def random_complete_board() -> Board:
    """
    ## Tabuleiro completo aleatório

    Cria uma matriz 4x4 preenchida aleatoriamente com valores de S.
    Na maioria das vezes, esse tabuleiro será inválido.
    """
    symbols = sorted(SYMBOLS)
    return [[random.choice(symbols) for _ in range(N)] for _ in range(N)]


def corrupt_valid_board(board: Board) -> Board:
    """
    ## Corrupção de solução válida

    Parte de um tabuleiro correto e altera algumas células para gerar exemplos
    negativos próximos dos positivos. Isso evita que a rede aprenda apenas a
    separar casos obviamente ruins.
    """
    corrupted = [row[:] for row in board]
    number_of_changes = random.randint(1, 4)

    for _ in range(number_of_changes):
        row = random.randrange(N)
        col = random.randrange(N)
        current = corrupted[row][col]
        alternatives = [value for value in SYMBOLS if value != current]
        corrupted[row][col] = random.choice(alternatives)

    return corrupted


def make_dataset(negative_multiplier: int = 6, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[Board]]:
    """
    ## Construção do dataset supervisionado

    Retorna:
    - X: entradas codificadas em one-hot;
    - y: rótulos binários;
    - valid_solutions: lista de soluções válidas usadas também para gerar puzzles.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    valid_solutions = generate_all_valid_solutions()

    X = []
    y = []

    # Classe positiva
    for board in valid_solutions:
        X.append(board_to_one_hot(board))
        y.append(1)

    # Classe negativa: mistura de tabuleiros aleatórios e tabuleiros corrompidos
    target_negatives = len(valid_solutions) * negative_multiplier
    negatives = []

    while len(negatives) < target_negatives:
        if random.random() < 0.5:
            candidate = random_complete_board()
        else:
            candidate = corrupt_valid_board(random.choice(valid_solutions))

        if not is_valid_complete_board(candidate):
            negatives.append(candidate)

    for board in negatives:
        X.append(board_to_one_hot(board))
        y.append(0)

    X_array = np.array(X)
    y_array = np.array(y)

    return X_array, y_array, valid_solutions
