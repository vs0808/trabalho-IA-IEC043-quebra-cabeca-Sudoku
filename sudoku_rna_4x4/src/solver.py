"""
# Geração e resolução de tabuleiros

Este módulo cria tabuleiros iniciais aleatórios e resolve os puzzles usando
busca com restrições. A RNA é usada para reconhecer a solução completa como válida.
"""

from __future__ import annotations

import random
from typing import List, Optional

from sklearn.neural_network import MLPClassifier

from dataset import board_to_one_hot
from sudoku_rules import Board, N, candidates_for_cell, empty_cells, is_valid_complete_board


def make_puzzle_from_solution(solution: Board, clues_to_keep: int = 8, random_seed: Optional[int] = None) -> Board:
    """
    ## Geração do tabuleiro inicial

    Recebe uma solução completa e remove números aleatoriamente.
    O valor 0 representa célula vazia.
    """
    if random_seed is not None:
        random.seed(random_seed)

    puzzle = [row[:] for row in solution]
    positions = [(r, c) for r in range(N) for c in range(N)]
    random.shuffle(positions)

    cells_to_remove = (N * N) - clues_to_keep
    for row, col in positions[:cells_to_remove]:
        puzzle[row][col] = 0

    return puzzle


def solve_with_constraints(board: Board) -> Optional[Board]:
    """
    ## Resolvedor por busca com restrições

    Esta função resolve o puzzle parcial.
    Ela usa a heurística MRV, isto é, escolhe primeiro a célula vazia com
    menor quantidade de candidatos possíveis.
    """
    working_board = [row[:] for row in board]

    def backtrack() -> bool:
        cells = empty_cells(working_board)
        if not cells:
            return is_valid_complete_board(working_board)

        # Heurística: escolher a célula mais restrita primeiro
        cells.sort(key=lambda pos: len(candidates_for_cell(working_board, pos[0], pos[1])))
        row, col = cells[0]
        candidates = candidates_for_cell(working_board, row, col)

        for value in candidates:
            working_board[row][col] = value
            if backtrack():
                return True
            working_board[row][col] = 0

        return False

    if backtrack():
        return working_board

    return None


def neural_validation(model: MLPClassifier, board: Board) -> float:
    """
    ## Validação neural

    Retorna a probabilidade estimada pela RNA de que um tabuleiro completo seja válido.
    """
    encoded = board_to_one_hot(board).reshape(1, -1)
    probabilities = model.predict_proba(encoded)[0]
    class_index = list(model.classes_).index(1)
    return float(probabilities[class_index])
