"""
# Visualização dos tabuleiros e do treinamento

Este módulo gera imagens para documentação do experimento.
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt

from sudoku_rules import Board, N, BLOCK


def plot_board(board: Board, title: str, output_path: str) -> None:
    """
    ## Imagem do tabuleiro

    Salva uma figura do Sudoku 4x4. Células vazias aparecem em branco.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(title)
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(N + 1):
        line_width = 2.5 if i % BLOCK == 0 else 1.0
        ax.plot([0, N], [i, i], linewidth=line_width)
        ax.plot([i, i], [0, N], linewidth=line_width)

    for row in range(N):
        for col in range(N):
            value = board[row][col]
            if value != 0:
                ax.text(
                    col + 0.5,
                    N - row - 0.5,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=22,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_loss_curve(loss_curve: List[float], output_path: str) -> None:
    """
    ## Curva de treinamento

    Salva a curva de perda da RNA ao longo das épocas.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_curve)
    ax.set_title("Curva de perda da RNA")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
