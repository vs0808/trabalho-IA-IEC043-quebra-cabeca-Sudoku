"""
# Código principal

Este arquivo integra todas as etapas do projeto:
1. geração do dataset;
2. treinamento da RNA;
3. geração de tabuleiro inicial;
4. resolução do Sudoku;
5. validação neural;
6. geração de imagens.
"""

from __future__ import annotations

import os
import random

from dataset import make_dataset
from model import train_model
from solver import make_puzzle_from_solution, neural_validation, solve_with_constraints
from sudoku_rules import is_valid_complete_board
from visualization import plot_board, plot_loss_curve


def print_board(title: str, board: list[list[int]]) -> None:
    """
    ## Impressão em formato de matriz

    Exibe o tabuleiro no terminal exatamente como uma matriz Python.
    """
    print(f"\n{title}")
    for row in board:
        print(row)


def main() -> None:
    """
    ## Execução completa do experimento
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("Gerando dataset...")
    X, y, valid_solutions = make_dataset(negative_multiplier=6, random_seed=42)

    print(f"Total de exemplos: {len(X)}")
    print(f"Exemplos positivos: {sum(y)}")
    print(f"Exemplos negativos: {len(y) - sum(y)}")
    print(f"Soluções Sudoku 4x4 válidas geradas: {len(valid_solutions)}")

    print("\nTreinando RNA multicamadas...")
    model, metrics = train_model(X, y, random_seed=42)

    print("\nMétricas de teste")
    print(f"Acurácia: {metrics['accuracy']:.4f}")
    print("Matriz de confusão:")
    print(metrics["confusion_matrix"])
    print("Relatório de classificação:")
    print(metrics["classification_report"])

    # Imagem da curva de treinamento
    plot_loss_curve(metrics["loss_curve"], os.path.join(output_dir, "curva_treinamento.png"))

    # Geração de puzzle inicial aleatório
    random.seed(7)
    complete_solution = random.choice(valid_solutions)
    puzzle = make_puzzle_from_solution(complete_solution, clues_to_keep=8, random_seed=7)

    print_board("Tabuleiro inicial aleatório", puzzle)
    plot_board(puzzle, "Tabuleiro inicial aleatório", os.path.join(output_dir, "tabuleiro_inicial.png"))

    # Resolução
    solved = solve_with_constraints(puzzle)

    if solved is None:
        print("\nNenhuma solução foi encontrada para o tabuleiro inicial.")
        return

    print_board("Solução final encontrada", solved)
    plot_board(solved, "Solução final encontrada", os.path.join(output_dir, "tabuleiro_resolvido.png"))

    logical_validity = is_valid_complete_board(solved)
    neural_probability = neural_validation(model, solved)

    print("\nValidação final")
    print(f"Validação lógica: {logical_validity}")
    print(f"Probabilidade estimada pela RNA para classe 'válido': {neural_probability:.4f}")

    if neural_probability >= 0.5 and logical_validity:
        print("Resultado: a RNA reconheceu o tabuleiro final como válido.")
    else:
        print("Resultado: a solução lógica foi encontrada, mas a RNA não a reconheceu com confiança suficiente.")

    print("\nImagens salvas em:")
    print(output_dir)


if __name__ == "__main__":
    main()
