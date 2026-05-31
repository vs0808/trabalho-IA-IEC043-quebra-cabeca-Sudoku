"""
# Regras do Sudoku 4x4

Este módulo concentra as funções lógicas do problema.
A decisão de separar as regras em um arquivo próprio torna o código mais legível,
facilita testes e ajuda na discussão sobre generalização para NxN.
"""

from typing import List, Set, Tuple

Board = List[List[int]]

N = 4
BLOCK = 2
SYMBOLS: Set[int] = {1, 2, 3, 4}


def is_value_valid(value: int) -> bool:
    """
    ## Verificação de domínio

    Garante que cada célula, quando preenchida, contenha apenas um número de S.
    Para tabuleiros parciais, o valor 0 representa célula vazia.
    """
    return value in SYMBOLS


def is_complete(board: Board) -> bool:
    """
    ## Verificação de completude

    Um tabuleiro completo não possui zeros.
    """
    return all(board[row][col] in SYMBOLS for row in range(N) for col in range(N))


def valid_group(values: List[int]) -> bool:
    """
    ## Verificação de grupo

    Um grupo completo é válido quando contém exatamente os símbolos {1,2,3,4}.
    Isso vale para linhas, colunas e subgrupos 2x2.
    """
    return set(values) == SYMBOLS and len(values) == N


def is_valid_complete_board(board: Board) -> bool:
    """
    ## Validador lógico de tabuleiro completo

    Esta função implementa a definição formal de uma solução Sudoku 4x4 válida.
    Ela é usada para criar o dataset e também para conferir o resultado final.
    """
    if len(board) != N or any(len(row) != N for row in board):
        return False

    if not is_complete(board):
        return False

    # Linhas
    for row in range(N):
        if not valid_group(board[row]):
            return False

    # Colunas
    for col in range(N):
        column = [board[row][col] for row in range(N)]
        if not valid_group(column):
            return False

    # Subgrupos 2x2
    for start_row in range(0, N, BLOCK):
        for start_col in range(0, N, BLOCK):
            block_values = []
            for row in range(start_row, start_row + BLOCK):
                for col in range(start_col, start_col + BLOCK):
                    block_values.append(board[row][col])
            if not valid_group(block_values):
                return False

    return True


def is_safe(board: Board, row: int, col: int, value: int) -> bool:
    """
    ## Teste de jogada segura

    Verifica se um valor pode ser colocado em uma célula sem violar linha,
    coluna ou subgrupo. Essa função permite resolver tabuleiros parciais.
    """
    if value not in SYMBOLS:
        return False

    # Linha
    if value in board[row]:
        return False

    # Coluna
    if value in [board[r][col] for r in range(N)]:
        return False

    # Subgrupo 2x2
    start_row = (row // BLOCK) * BLOCK
    start_col = (col // BLOCK) * BLOCK
    for r in range(start_row, start_row + BLOCK):
        for c in range(start_col, start_col + BLOCK):
            if board[r][c] == value:
                return False

    return True


def empty_cells(board: Board) -> List[Tuple[int, int]]:
    """
    ## Lista de células vazias

    Retorna todas as posições cujo valor é 0.
    """
    return [(r, c) for r in range(N) for c in range(N) if board[r][c] == 0]


def candidates_for_cell(board: Board, row: int, col: int) -> List[int]:
    """
    ## Candidatos possíveis

    Retorna todos os números de S que podem ocupar uma célula sem violar
    imediatamente as restrições do Sudoku.
    """
    if board[row][col] != 0:
        return []
    return [value for value in sorted(SYMBOLS) if is_safe(board, row, col, value)]
