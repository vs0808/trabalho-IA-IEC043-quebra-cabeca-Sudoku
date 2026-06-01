# Sudoku 4x4 com Rede Neural Artificial Multicamadas

Este projeto apresenta uma solução em Python para o problema do Sudoku 4x4 utilizando uma Rede Neural Artificial multicamadas como mecanismo de reconhecimento de tabuleiros válidos.

O trabalho foi desenvolvido a partir de uma grade 4x4, com subgrupos 2x2 e valores pertencentes ao conjunto S = {1, 2, 3, 4}. A proposta combina geração de dados, validação lógica, treinamento supervisionado e busca de solução para tabuleiros iniciais parcialmente preenchidos.

## Objetivo

O objetivo principal é demonstrar como uma RNA pode ser treinada para reconhecer configurações válidas de Sudoku 4x4 e como essa rede pode ser integrada a um processo de geração e solução de tabuleiros.

Além da implementação do caso 4x4, o projeto também discute as dificuldades envolvidas na generalização da solução para grades NxN.

## O que o projeto faz

- Gera automaticamente soluções válidas de Sudoku 4x4.
- Cria exemplos inválidos para compor o conjunto de treinamento.
- Treina uma Rede Neural Artificial multicamadas.
- Testa a capacidade da RNA de reconhecer tabuleiros válidos.
- Gera um tabuleiro inicial aleatório parcialmente preenchido.
- Encontra uma solução final para o tabuleiro gerado.
- Produz imagens do tabuleiro inicial, do tabuleiro resolvido e da curva de treinamento.
- Apresenta documentação explicando a estrutura do projeto, o funcionamento do código e as imagens geradas.

## Estrutura geral

```text
sudoku_rna_4x4/
│
├── README.md
├── requirements.txt
├── main.py
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── solver.py
│   ├── sudoku_rules.py
│   └── visualization.py
│
├── docs/
│   ├── ESTRUTURA_DO_PROJETO.md
│   └── EXPLICACAO_DAS_IMAGENS.md
│
└── outputs/
    ├── tabuleiro_inicial.png
    ├── tabuleiro_resolvido.png
    └── curva_treinamento.png

## Membros

Domynique Alexandrina Monteiro de Souza
João Vitor de Aguiar Ribeiro
Thales Araújo de Souza
Vinicius de Souza Menezes