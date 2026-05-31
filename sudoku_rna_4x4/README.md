# Sudoku 4x4 com RNA Multicamadas em Python

## 1. Objetivo do trabalho

Este projeto propõe uma solução de **Rede Neural Artificial (RNA) multicamadas**, implementada em Python, para reconhecer e auxiliar a solução de um Sudoku **4x4**, com subgrupos **2x2** e símbolos `S = {1, 2, 3, 4}`.

A proposta atende aos pontos solicitados:

1. Garante que cada célula contenha apenas um número pertencente a `S`.
2. Garante que não haja repetição em linhas, colunas e subgrupos 2x2.
3. Garante que cada linha e coluna da grade principal 4x4 tenha exatamente uma ocorrência de cada número de `S`.
4. Apresenta um conjunto de dados para treinamento e teste.
5. Gera tabuleiros iniciais aleatórios em formato de matriz, usa a RNA treinada para reconhecer soluções válidas e gera uma solução final.
6. Discute as dificuldades de generalizar a solução de 4x4 para Sudoku NxN.
7. Discute o problema de tratar “gerar amostras e testá-las” como raciocínio.

> Observação importante: a RNA aqui não substitui completamente o raciocínio lógico simbólico. Ela atua como um **classificador neural de validade** para tabuleiros completos. A geração da solução a partir de um tabuleiro inicial usa busca guiada por restrições, e a RNA valida/reconhece a solução final como válida.

---

## 2. Por que usar uma RNA multicamadas nesse problema?

O Sudoku é, por natureza, um problema de **satisfação de restrições**. Isso significa que uma solução correta precisa obedecer a regras exatas:

- cada célula deve conter um único valor;
- cada linha deve conter os números 1, 2, 3 e 4 sem repetição;
- cada coluna deve conter os números 1, 2, 3 e 4 sem repetição;
- cada subgrupo 2x2 deve conter os números 1, 2, 3 e 4 sem repetição.

Uma RNA multicamadas pode aprender padrões de validade a partir de exemplos, mas ela não possui, por si só, a mesma garantia formal de um algoritmo lógico. Por isso, este projeto usa uma arquitetura híbrida:

1. **Gerador lógico de soluções válidas 4x4**: produz exemplos positivos.
2. **Gerador de tabuleiros inválidos**: produz exemplos negativos.
3. **RNA MLP**: aprende a distinguir tabuleiros completos válidos e inválidos.
4. **Gerador de tabuleiros iniciais**: remove alguns números de uma solução válida para criar um puzzle.
5. **Resolvedor por busca com restrições**: preenche as células vazias respeitando as regras do Sudoku.
6. **Validador neural final**: a RNA reconhece se o tabuleiro final é válido.

Essa decisão foi tomada porque exigir que uma RNA densa gere diretamente um Sudoku solucionado a partir de qualquer tabuleiro parcial seria pouco robusto, especialmente com poucos dados e sem mecanismos explícitos de restrição.

---

## 3. Representação dos dados

Cada tabuleiro 4x4 é representado como uma matriz:

```python
[
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1]
]
```

Para a RNA, esse tabuleiro é convertido para **one-hot encoding**.

Como existem 16 células e cada célula pode assumir 4 valores, cada tabuleiro vira um vetor de dimensão:

```text
16 células × 4 possibilidades = 64 entradas
```

Exemplo conceitual:

- valor 1 vira `[1, 0, 0, 0]`
- valor 2 vira `[0, 1, 0, 0]`
- valor 3 vira `[0, 0, 1, 0]`
- valor 4 vira `[0, 0, 0, 1]`

A saída da rede é binária:

```text
1 = tabuleiro completo válido
0 = tabuleiro completo inválido
```

---

## 4. Modelo neural escolhido

Foi usado um **MLPClassifier** com camadas ocultas densas:

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)
```

### Justificativa das escolhas

- `hidden_layer_sizes=(128, 64, 32)`: camadas suficientes para aprender combinações entre células, linhas, colunas e subgrupos.
- `relu`: função de ativação simples, eficiente e comum em redes multicamadas.
- `adam`: otimizador robusto para treinamento supervisionado.
- `max_iter=1000`: permite convergência com o conjunto pequeno de dados.
- `random_state=42`: garante reprodutibilidade dos resultados.

---

## 5. Como o dataset foi construído

### 5.1 Exemplos positivos

Todos os tabuleiros 4x4 válidos são gerados por backtracking. Para Sudoku 4x4, há poucas soluções possíveis, então podemos gerar todas as soluções válidas.

### 5.2 Exemplos negativos

Os exemplos negativos são criados de três formas:

1. Gerando matrizes aleatórias com valores de 1 a 4.
2. Pegando uma solução válida e trocando posições para provocar violações.
3. Gerando linhas embaralhadas que nem sempre obedecem colunas/subgrupos.

Isso ensina a RNA a diferenciar padrões válidos de padrões quase válidos ou completamente inválidos.

---

## 6. Como a solução final é gerada

O programa executa os seguintes passos:

1. Treina a RNA com tabuleiros válidos e inválidos.
2. Sorteia uma solução válida completa.
3. Remove alguns números para formar um tabuleiro inicial.
4. Resolve o tabuleiro inicial por busca com restrições.
5. Passa a solução encontrada pela RNA.
6. Mostra se a RNA reconheceu a solução como válida.
7. Gera imagens do tabuleiro inicial e do tabuleiro resolvido.

---

## 7. Por que gerar amostras e testá-las é problemático se o problema é de raciocínio?

Gerar amostras e testá-las funciona para Sudoku 4x4 porque o espaço de busca é pequeno. Porém, isso não representa raciocínio lógico completo.

Em um Sudoku 4x4, há 16 células e 4 valores possíveis, então o espaço bruto de combinações é:

```text
4^16 = 4.294.967.296 combinações possíveis
```

Apesar disso parecer grande, as restrições reduzem bastante o espaço, e a geração por backtracking torna o problema administrável.

Mas, para um Sudoku 9x9 tradicional, temos:

```text
9^81 combinações possíveis
```

Esse número é gigantesco. Testar amostras aleatórias não garante que uma solução será encontrada em tempo razoável. O problema é que amostragem não entende o motivo lógico de uma escolha ser necessária; ela apenas tenta possibilidades.

Em problemas de raciocínio, espera-se que o sistema:

- use restrições explicitamente;
- elimine possibilidades impossíveis;
- preserve coerência entre decisões;
- justifique escolhas;
- tenha garantias de validade.

Uma RNA treinada apenas por exemplos pode reconhecer padrões, mas pode falhar em casos fora da distribuição do treinamento. Assim, amostrar e testar pode parecer resolver o problema em escala pequena, mas não necessariamente aprende o raciocínio geral do Sudoku.

---

## 8. Dificuldades de generalizar de 4x4 para NxN

A generalização para NxN exige mudanças importantes.

### 8.1 Mudança no tamanho da entrada

No 4x4, a entrada da RNA tem 64 posições:

```text
4 × 4 × 4 = 64
```

Para um Sudoku 9x9, a entrada teria:

```text
9 × 9 × 9 = 729
```

Para um Sudoku 16x16, teria:

```text
16 × 16 × 16 = 4096
```

Uma MLP tradicional depende de tamanho fixo de entrada. Portanto, uma rede treinada para 4x4 não recebe naturalmente um tabuleiro 9x9.

### 8.2 Mudança nas restrições

No Sudoku 4x4, os subgrupos são 2x2. No 9x9, são 3x3. Em um Sudoku 16x16, seriam 4x4.

O algoritmo precisa ser parametrizado por:

```python
N = tamanho da grade
B = tamanho do bloco
S = conjunto de símbolos
```

com a condição:

```text
N = B²
```

### 8.3 Crescimento explosivo do espaço de busca

A complexidade aumenta de forma exponencial. Mesmo usando backtracking, resolver instâncias maiores exige heurísticas melhores, como:

- escolher primeiro a célula com menor número de candidatos;
- propagação de restrições;
- forward checking;
- poda de estados inválidos;
- uso de técnicas específicas de Sudoku.

### 8.4 Dataset muito maior

Gerar exemplos positivos e negativos para NxN é mais difícil. No 4x4, podemos gerar todos os tabuleiros válidos. No 9x9, o número de soluções é gigantesco. Não é viável enumerar tudo.

### 8.5 A rede pode memorizar em vez de generalizar

Em 4x4, como há poucas soluções, a RNA pode acabar decorando padrões. Isso não significa que ela aprendeu uma regra geral de Sudoku.

Para generalizar, seria melhor usar arquiteturas que representem relações entre células, como:

- Graph Neural Networks;
- Transformers com atenção entre células;
- redes com camadas de restrição diferenciável;
- modelos híbridos com programação por restrições.

---

## 9. Como executar
O ambiente de desenvolvimento foi o Linux/Ubuntu, por padrão, essa distro
já vêm com o python instalado.
O processo é bastante familiar em outros sistemas operacionais.
Crie um ambiente Python e instale as dependências:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Execute:

```bash
python src/main.py
```

O programa irá:

- gerar o dataset;
- treinar a RNA;
- avaliar no conjunto de teste;
- gerar um tabuleiro inicial aleatório;
- resolver o tabuleiro;
- validar a solução com a RNA;
- salvar imagens na pasta `outputs/`.

---

## 10. Estrutura do projeto

```text
sudoku_rna_4x4/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── sudoku_rules.py
│   ├── dataset.py
│   ├── model.py
│   ├── solver.py
│   └── visualization.py
└── outputs/
    ├── tabuleiro_inicial.png
    ├── tabuleiro_resolvido.png
    └── curva_treinamento.png
```

