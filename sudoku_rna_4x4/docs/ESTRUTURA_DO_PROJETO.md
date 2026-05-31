# Estrutura detalhada do projeto — Sudoku 4x4 com RNA Multicamadas

Este documento complementa o `README.md` principal do repositório. Enquanto o README apresenta a proposta geral do trabalho, a justificativa teórica e a forma de execução, este arquivo explica com mais profundidade a **estrutura interna do projeto**, o papel de cada arquivo Python, o fluxo de dados entre os módulos e as principais funções implementadas.

A ideia central do projeto é resolver e validar um Sudoku 4x4 utilizando uma abordagem híbrida:

1. regras lógicas explícitas para representar as restrições do Sudoku;
2. geração automática de dados positivos e negativos;
3. treinamento de uma Rede Neural Artificial multicamadas para reconhecer tabuleiros completos válidos;
4. geração de tabuleiros iniciais aleatórios;
5. resolução do puzzle por busca com restrições;
6. validação final lógica e neural;
7. geração de imagens para documentação do resultado.

---

## 1. Visão geral da estrutura de pastas

A estrutura esperada do projeto é a seguinte:

```text
sudoku_rna_4x4/
├── README.md
├── requirements.txt
├── docs/
│   └── ESTRUTURA_DO_PROJETO.md
├── outputs/
│   ├── curva_treinamento.png
│   ├── tabuleiro_inicial.png
│   └── tabuleiro_resolvido.png
└── src/
    ├── sudoku_rules.py
    ├── dataset.py
    ├── model.py
    ├── solver.py
    ├── visualization.py
    └── main.py
```

Cada parte tem uma responsabilidade específica.

| Caminho | Função no projeto |
|---|---|
| `README.md` | Documento principal do trabalho. Explica o objetivo, a solução proposta, a discussão sobre RNA, raciocínio e generalização NxN. |
| `requirements.txt` | Lista as bibliotecas necessárias para executar o projeto. |
| `docs/ESTRUTURA_DO_PROJETO.md` | Documento detalhado sobre a organização do código e o papel das principais funções. |
| `outputs/` | Pasta onde são salvas as imagens geradas pela execução. |
| `src/` | Pasta com o código-fonte Python. |
| `src/sudoku_rules.py` | Implementa as regras formais do Sudoku 4x4. |
| `src/dataset.py` | Gera o dataset supervisionado para treinar e testar a RNA. |
| `src/model.py` | Define, treina e avalia a Rede Neural Artificial. |
| `src/solver.py` | Gera puzzles iniciais e resolve tabuleiros parciais por busca com restrições. |
| `src/visualization.py` | Gera imagens dos tabuleiros e da curva de treinamento. |
| `src/main.py` | Arquivo principal que integra e executa todo o experimento. |

---

## 2. Decisão de arquitetura

O projeto foi separado em módulos para evitar que todo o código ficasse concentrado em um único arquivo. Essa separação facilita a leitura, a manutenção e a explicação acadêmica da solução.

A divisão segue a seguinte lógica:

```text
Regras do problema     → sudoku_rules.py
Geração dos dados      → dataset.py
Modelo neural          → model.py
Resolução do puzzle    → solver.py
Visualização           → visualization.py
Execução geral         → main.py
```

Essa organização também ajuda a evidenciar uma questão importante do trabalho: o Sudoku não é apenas um problema de classificação, mas um problema de **satisfação de restrições**. Por isso, a RNA não trabalha sozinha. Ela é usada como parte de uma solução híbrida.

A RNA aprende a reconhecer tabuleiros completos válidos, enquanto o resolvedor simbólico por restrições é responsável por preencher o tabuleiro inicial. Essa escolha é importante porque uma MLP comum não garante, sozinha, que a saída gerada obedecerá a todas as regras do Sudoku.

---

## 3. Fluxo geral de execução

O fluxo completo do programa acontece dentro de `src/main.py`.

De forma resumida, a execução segue esta sequência:

```text
1. Gerar dataset
2. Treinar a RNA
3. Avaliar a RNA no conjunto de teste
4. Gerar curva de treinamento
5. Sortear uma solução Sudoku 4x4 válida
6. Remover alguns valores para formar um puzzle inicial
7. Resolver o puzzle por busca com restrições
8. Validar a solução com regras lógicas
9. Validar a solução com a RNA treinada
10. Gerar imagens do tabuleiro inicial e do tabuleiro resolvido
```

Em termos de módulos, o fluxo pode ser visto assim:

```text
main.py
│
├── chama make_dataset() em dataset.py
│   └── usa regras de sudoku_rules.py
│
├── chama train_model() em model.py
│   └── treina MLPClassifier com scikit-learn
│
├── chama make_puzzle_from_solution() em solver.py
│   └── cria tabuleiro inicial com células vazias
│
├── chama solve_with_constraints() em solver.py
│   └── usa candidates_for_cell() e is_safe() de sudoku_rules.py
│
├── chama neural_validation() em solver.py
│   └── usa board_to_one_hot() de dataset.py
│
└── chama plot_board() e plot_loss_curve() em visualization.py
    └── salva imagens na pasta outputs/
```

---

# 4. Arquivo `src/sudoku_rules.py`

Este é um dos arquivos mais importantes do projeto, pois concentra a definição formal das regras do Sudoku 4x4.

Ele define:

```python
N = 4
BLOCK = 2
SYMBOLS = {1, 2, 3, 4}
```

Essas três constantes determinam o tamanho da grade, o tamanho dos blocos e o conjunto de símbolos permitidos.

## 4.1 Responsabilidade do arquivo

O papel de `sudoku_rules.py` é responder perguntas como:

- este valor pertence ao conjunto permitido `S = {1, 2, 3, 4}`?
- o tabuleiro está completamente preenchido?
- uma linha contém exatamente os símbolos esperados?
- uma coluna contém exatamente os símbolos esperados?
- um subgrupo 2x2 contém exatamente os símbolos esperados?
- é seguro colocar determinado valor em determinada célula?
- quais células ainda estão vazias?
- quais candidatos são possíveis para uma célula vazia?

Este módulo é usado por quase todos os outros arquivos, porque as regras do Sudoku são a base tanto para gerar dados quanto para resolver o puzzle.

---

## 4.2 Funções principais de `sudoku_rules.py`

### `is_value_valid(value: int) -> bool`

Verifica se um valor pertence ao conjunto de símbolos permitido.

```python
def is_value_valid(value: int) -> bool:
    return value in SYMBOLS
```

No problema do trabalho, as células devem conter apenas valores de `S = {1, 2, 3, 4}`. Portanto, essa função representa a primeira regra básica do Sudoku: não aceitar símbolos fora do domínio.

Ela retorna:

- `True`, se o valor estiver em `{1, 2, 3, 4}`;
- `False`, se o valor for qualquer outro número.

Observação: no projeto, o valor `0` é usado para representar célula vazia em tabuleiros parciais. Portanto, `0` não é considerado um valor válido para um tabuleiro completo.

---

### `is_complete(board: Board) -> bool`

Verifica se o tabuleiro está completamente preenchido.

```python
def is_complete(board: Board) -> bool:
    return all(board[row][col] in SYMBOLS for row in range(N) for col in range(N))
```

Essa função percorre todas as 16 células do Sudoku 4x4 e verifica se todas contêm valores válidos.

Ela é importante porque a RNA foi treinada para classificar **tabuleiros completos**, não tabuleiros parciais. O resolvedor pode trabalhar com zeros, mas a validação final exige que não haja células vazias.

---

### `valid_group(values: List[int]) -> bool`

Verifica se um grupo contém exatamente os símbolos `{1, 2, 3, 4}`.

```python
def valid_group(values: List[int]) -> bool:
    return set(values) == SYMBOLS and len(values) == N
```

Um grupo pode ser:

- uma linha;
- uma coluna;
- um subgrupo 2x2.

Para ser válido, esse grupo precisa conter todos os números de 1 a 4, sem repetição e sem ausência.

Exemplo válido:

```python
[1, 2, 3, 4]
```

Exemplo inválido por repetição:

```python
[1, 1, 3, 4]
```

Exemplo inválido por ausência:

```python
[1, 2, 3, 3]
```

Essa função é uma das mais importantes do projeto, pois traduz diretamente a regra de não repetição.

---

### `is_valid_complete_board(board: Board) -> bool`

Valida se um tabuleiro completo é uma solução Sudoku 4x4 correta.

Essa função verifica:

1. se o tabuleiro tem dimensão 4x4;
2. se todas as células estão preenchidas;
3. se todas as linhas são válidas;
4. se todas as colunas são válidas;
5. se todos os subgrupos 2x2 são válidos.

Ela é usada em três momentos fundamentais:

- na geração do dataset, para separar exemplos válidos e inválidos;
- no resolvedor, para confirmar que a solução encontrada é correta;
- no final do programa, para comparar validação lógica com validação neural.

Essa função oferece uma validação determinística. Diferente da RNA, ela não estima probabilidade. Ela simplesmente retorna `True` ou `False` com base nas regras formais.

Por isso, ela funciona como uma espécie de “juiz lógico” do projeto.

---

### `is_safe(board: Board, row: int, col: int, value: int) -> bool`

Verifica se um valor pode ser colocado em uma posição sem violar as regras do Sudoku.

A função testa três restrições:

1. o valor não pode aparecer na mesma linha;
2. o valor não pode aparecer na mesma coluna;
3. o valor não pode aparecer no mesmo subgrupo 2x2.

Esse teste é usado tanto para gerar soluções válidas quanto para resolver puzzles parciais.

Exemplo de raciocínio:

```text
Queremos colocar o número 3 na célula (linha 1, coluna 2).
A função verifica:
- já existe 3 na linha 1?
- já existe 3 na coluna 2?
- já existe 3 no bloco 2x2 dessa célula?
Se alguma resposta for sim, a jogada não é segura.
```

Essa função é uma peça central do projeto porque implementa a noção de restrição local.

---

### `empty_cells(board: Board) -> List[Tuple[int, int]]`

Retorna uma lista com as posições das células vazias.

No projeto, célula vazia é representada por `0`.

Exemplo:

```python
[
    [4, 1, 3, 0],
    [0, 0, 4, 0],
    [1, 0, 2, 0],
    [2, 0, 0, 3]
]
```

A função retornaria posições como:

```python
[(0, 3), (1, 0), (1, 1), ...]
```

Ela é usada pelo resolvedor para saber quais células ainda precisam ser preenchidas.

---

### `candidates_for_cell(board: Board, row: int, col: int) -> List[int]`

Retorna todos os valores possíveis para uma célula vazia.

A função testa cada valor de `SYMBOLS` usando `is_safe()`.

Exemplo:

```python
return [value for value in sorted(SYMBOLS) if is_safe(board, row, col, value)]
```

Se uma célula só puder receber o número 2 sem violar regras, a função retorna:

```python
[2]
```

Se puder receber 1 ou 4, retorna:

```python
[1, 4]
```

Essa função é essencial para a heurística MRV usada no resolvedor.

---

# 5. Arquivo `src/dataset.py`

Este arquivo constrói o conjunto de dados usado para treinar e testar a RNA.

A RNA precisa de exemplos rotulados. Portanto, o projeto precisa criar dois tipos de tabuleiro:

- exemplos positivos: tabuleiros completos válidos;
- exemplos negativos: tabuleiros completos inválidos.

## 5.1 Responsabilidade do arquivo

O papel de `dataset.py` é:

1. converter tabuleiros em vetores numéricos;
2. gerar todas as soluções válidas do Sudoku 4x4;
3. gerar tabuleiros completos aleatórios inválidos;
4. corromper soluções válidas para criar negativos difíceis;
5. montar o dataset final `X` e `y`.

---

## 5.2 Funções principais de `dataset.py`

### `board_to_one_hot(board: Board) -> np.ndarray`

Converte uma matriz 4x4 em um vetor one-hot de 64 posições.

O Sudoku 4x4 tem:

```text
16 células × 4 valores possíveis = 64 entradas
```

Cada célula é convertida da seguinte forma:

| Valor da célula | Representação one-hot |
|---|---|
| 1 | `[1, 0, 0, 0]` |
| 2 | `[0, 1, 0, 0]` |
| 3 | `[0, 0, 1, 0]` |
| 4 | `[0, 0, 0, 1]` |

Essa codificação foi escolhida porque os números do Sudoku são categorias, não grandezas contínuas. Ou seja, o número 4 não é “maior” que o número 1 no sentido matemático relevante para a rede; ele é apenas outro símbolo.

Usar one-hot evita que a RNA interprete incorretamente relações numéricas artificiais entre os símbolos.

---

### `generate_all_valid_solutions() -> List[Board]`

Gera todas as soluções válidas do Sudoku 4x4 usando backtracking.

O funcionamento é:

1. começa com um tabuleiro 4x4 vazio;
2. percorre as células em sequência;
3. tenta preencher cada célula com valores de `{1, 2, 3, 4}`;
4. usa `is_safe()` para verificar se o valor pode ser colocado;
5. se puder, continua para a próxima célula;
6. se chegar ao final do tabuleiro, salva uma solução válida;
7. se nenhum valor funcionar, volta para a célula anterior.

Essa função gera o conjunto positivo do treinamento.

No Sudoku 4x4, isso é viável porque o número de soluções válidas é pequeno. Na execução de exemplo, foram geradas 288 soluções válidas.

Essa função é importante também para a discussão sobre generalização: em um Sudoku maior, como 9x9, enumerar todas as soluções válidas não é viável.

---

### `random_complete_board() -> Board`

Cria um tabuleiro 4x4 completamente preenchido com valores aleatórios de 1 a 4.

Exemplo possível:

```python
[
    [1, 4, 2, 2],
    [3, 1, 4, 4],
    [2, 2, 1, 3],
    [4, 3, 3, 1]
]
```

A maioria desses tabuleiros será inválida, porque dificilmente respeitará simultaneamente linhas, colunas e subgrupos.

Essa função ajuda a gerar exemplos negativos simples.

---

### `corrupt_valid_board(board: Board) -> Board`

Cria exemplos negativos a partir de uma solução válida.

Ela recebe uma solução correta e altera aleatoriamente algumas células. Isso gera tabuleiros que podem parecer próximos de uma solução real, mas que violam alguma regra.

Esse tipo de negativo é importante porque evita que a RNA aprenda apenas a separar casos muito fáceis.

Exemplo:

Solução válida:

```python
[
    [4, 1, 3, 2],
    [3, 2, 4, 1],
    [1, 3, 2, 4],
    [2, 4, 1, 3]
]
```

Após corrupção:

```python
[
    [4, 1, 3, 2],
    [3, 2, 4, 1],
    [1, 1, 2, 4],
    [2, 4, 1, 3]
]
```

Nesse exemplo, a repetição do número 1 em uma linha ou coluna tornaria o tabuleiro inválido.

---

### `make_dataset(negative_multiplier: int = 6, random_seed: int = 42)`

Monta o dataset supervisionado completo.

Retorna três objetos:

```python
X, y, valid_solutions
```

Onde:

- `X` contém os tabuleiros codificados em one-hot;
- `y` contém os rótulos, sendo `1` para válido e `0` para inválido;
- `valid_solutions` contém as soluções válidas originais, usadas depois para gerar puzzles.

O parâmetro `negative_multiplier=6` indica que o dataset terá seis vezes mais exemplos negativos do que positivos. Isso foi usado para expor a RNA a uma variedade maior de violações.

Na execução de exemplo:

```text
Exemplos positivos: 288
Exemplos negativos: 1728
Total: 2016
```

O `random_seed=42` foi usado para tornar os resultados reprodutíveis.

---

# 6. Arquivo `src/model.py`

Este arquivo define e treina a Rede Neural Artificial multicamadas.

Foi usado o `MLPClassifier` da biblioteca `scikit-learn`.

## 6.1 Responsabilidade do arquivo

O papel de `model.py` é:

1. criar a arquitetura da RNA;
2. dividir o dataset em treino e teste;
3. treinar o modelo;
4. calcular métricas de avaliação;
5. retornar o modelo treinado e os resultados.

---

## 6.2 Funções principais de `model.py`

### `create_mlp(random_seed: int = 42) -> MLPClassifier`

Cria a RNA multicamadas.

A arquitetura usada foi:

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=random_seed,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=40,
)
```

Interpretação da arquitetura:

| Parâmetro | Significado |
|---|---|
| `hidden_layer_sizes=(128, 64, 32)` | Três camadas ocultas, com 128, 64 e 32 neurônios. |
| `activation="relu"` | Função de ativação usada entre as camadas. |
| `solver="adam"` | Algoritmo de otimização usado no treinamento. |
| `max_iter=1000` | Número máximo de épocas de treinamento. |
| `early_stopping=True` | Interrompe o treinamento se a validação parar de melhorar. |
| `validation_fraction=0.15` | Reserva 15% do treino para validação interna. |
| `n_iter_no_change=40` | Espera 40 épocas sem melhora antes de parar. |

A entrada da rede possui 64 posições, pois cada tabuleiro 4x4 é transformado em um vetor one-hot.

A saída é binária:

```text
0 = tabuleiro inválido
1 = tabuleiro válido
```

---

### `train_model(X: np.ndarray, y: np.ndarray, random_seed: int = 42)`

Treina e avalia a RNA.

A função executa estas etapas:

1. divide o dataset em treino e teste usando `train_test_split`;
2. usa `stratify=y` para manter a proporção entre classes válidas e inválidas;
3. cria a RNA chamando `create_mlp()`;
4. treina o modelo com `model.fit(X_train, y_train)`;
5. faz predições no conjunto de teste;
6. calcula acurácia, matriz de confusão e relatório de classificação;
7. retorna o modelo treinado e um dicionário de métricas.

O uso de `stratify=y` é importante porque o dataset é desbalanceado: há mais exemplos negativos do que positivos. A estratificação evita que o conjunto de teste fique com uma distribuição muito diferente do conjunto total.

As principais métricas retornadas são:

```python
metrics = {
    "accuracy": ...,
    "confusion_matrix": ...,
    "classification_report": ...,
    "loss_curve": ...,
    "X_train_size": ...,
    "X_test_size": ...,
}
```

A `loss_curve` é usada depois para gerar a imagem da curva de treinamento.

---

# 7. Arquivo `src/solver.py`

Este arquivo cuida da geração do tabuleiro inicial e da resolução do Sudoku.

Embora o trabalho peça uma solução com RNA, foi adotada uma solução híbrida. A RNA reconhece soluções válidas, mas a resolução do tabuleiro parcial é feita por busca com restrições.

Essa decisão é importante porque uma MLP tradicional não é naturalmente uma máquina de raciocínio lógico passo a passo. Ela classifica padrões, mas não garante que consegue construir uma solução válida a partir de um tabuleiro parcial.

---

## 7.1 Responsabilidade do arquivo

O papel de `solver.py` é:

1. criar um puzzle inicial aleatório a partir de uma solução válida;
2. resolver o puzzle parcial usando backtracking;
3. usar heurística MRV para reduzir o espaço de busca;
4. validar neuralmente a solução final.

---

## 7.2 Funções principais de `solver.py`

### `make_puzzle_from_solution(solution: Board, clues_to_keep: int = 8, random_seed: Optional[int] = None) -> Board`

Gera um tabuleiro inicial aleatório.

Ela recebe uma solução completa e remove parte dos números. As células removidas são preenchidas com `0`, que representa vazio.

Exemplo de solução completa:

```python
[
    [4, 1, 3, 2],
    [3, 2, 4, 1],
    [1, 3, 2, 4],
    [2, 4, 1, 3]
]
```

Exemplo de puzzle gerado:

```python
[
    [4, 1, 3, 0],
    [0, 0, 4, 0],
    [1, 0, 2, 0],
    [2, 0, 0, 3]
]
```

O parâmetro `clues_to_keep=8` significa que 8 células permanecem preenchidas e as outras 8 são apagadas.

Essa função garante que o puzzle gerado tem pelo menos uma solução, porque ele é derivado de uma solução completa válida.

Observação importante: a função não verifica se a solução é única. Para um Sudoku 4x4 acadêmico, isso é aceitável, mas em um gerador profissional de Sudoku seria necessário verificar unicidade.

---

### `solve_with_constraints(board: Board) -> Optional[Board]`

Resolve o tabuleiro parcial.

Essa é uma das funções mais importantes do projeto.

Ela usa backtracking com a heurística MRV.

MRV significa **Minimum Remaining Values**, ou seja, “menor número de valores restantes”. A ideia é escolher primeiro a célula vazia com menos candidatos possíveis.

Exemplo:

```text
Célula A pode receber [1, 2, 3]
Célula B pode receber [4]
Célula C pode receber [2, 4]
```

A heurística escolhe primeiro a célula B, porque ela tem apenas uma possibilidade. Isso reduz o número de tentativas erradas e acelera a busca.

Fluxo interno da função:

```text
1. Copia o tabuleiro recebido
2. Procura células vazias
3. Se não houver células vazias, valida o tabuleiro completo
4. Ordena as células vazias pela quantidade de candidatos
5. Escolhe a célula mais restrita
6. Testa cada candidato possível
7. Continua recursivamente
8. Se uma escolha levar a erro, desfaz a jogada
9. Retorna o tabuleiro quando encontrar solução
```

Essa função retorna:

- um tabuleiro resolvido, se encontrar solução;
- `None`, se nenhuma solução for possível.

---

### `neural_validation(model: MLPClassifier, board: Board) -> float`

Usa a RNA treinada para estimar a probabilidade de um tabuleiro ser válido.

A função:

1. converte o tabuleiro para one-hot usando `board_to_one_hot()`;
2. ajusta o formato para uma amostra com `reshape(1, -1)`;
3. chama `model.predict_proba()`;
4. localiza a probabilidade associada à classe `1`, isto é, “válido”;
5. retorna essa probabilidade como número decimal.

Exemplo de saída:

```text
0.9923
```

Isso significa que, segundo a RNA, o tabuleiro tem aproximadamente 99,23% de probabilidade de pertencer à classe dos tabuleiros válidos.

É importante destacar que essa probabilidade não substitui a validação lógica. A validação neural é uma estimativa aprendida a partir dos dados.

---

# 8. Arquivo `src/visualization.py`

Este arquivo gera as imagens usadas para documentar visualmente o experimento.

Ele usa a biblioteca `matplotlib`.

## 8.1 Responsabilidade do arquivo

O papel de `visualization.py` é:

1. gerar uma imagem do tabuleiro inicial;
2. gerar uma imagem do tabuleiro resolvido;
3. gerar uma imagem da curva de perda do treinamento da RNA.

As imagens são salvas na pasta `outputs/`.

---

## 8.2 Funções principais de `visualization.py`

### `plot_board(board: Board, title: str, output_path: str) -> None`

Gera uma imagem do tabuleiro Sudoku.

A função:

1. cria uma figura 4x4;
2. remove os eixos tradicionais;
3. desenha as linhas da grade;
4. usa linhas mais grossas para separar os blocos 2x2;
5. escreve os valores preenchidos no centro das células;
6. deixa células com `0` em branco;
7. salva a imagem no caminho informado.

Essa função é usada duas vezes no fluxo principal:

```python
plot_board(puzzle, "Tabuleiro inicial aleatório", "outputs/tabuleiro_inicial.png")
plot_board(solved, "Solução final encontrada", "outputs/tabuleiro_resolvido.png")
```

---

### `plot_loss_curve(loss_curve: List[float], output_path: str) -> None`

Gera o gráfico da curva de perda da RNA.

A curva de perda mostra como o erro do modelo evoluiu durante o treinamento.

Em geral:

- se a perda diminui, o modelo está aprendendo;
- se a perda fica estável, o treinamento convergiu;
- se a perda oscila muito, pode haver dificuldade de treinamento.

A função salva o gráfico em:

```text
outputs/curva_treinamento.png
```

Esse gráfico é útil para apresentar evidência visual do processo de treinamento.

---

# 9. Arquivo `src/main.py`

Este é o arquivo de entrada do projeto.

Ele conecta todos os módulos e executa o experimento completo.

Para rodar o projeto, o comando principal é:

```bash
python src/main.py
```

---

## 9.1 Responsabilidade do arquivo

O papel de `main.py` é:

1. preparar a pasta `outputs/`;
2. gerar o dataset;
3. treinar a RNA;
4. mostrar métricas de avaliação;
5. gerar a curva de treinamento;
6. criar um tabuleiro inicial aleatório;
7. salvar imagem do tabuleiro inicial;
8. resolver o tabuleiro;
9. salvar imagem do tabuleiro resolvido;
10. validar o resultado por regras lógicas;
11. validar o resultado pela RNA;
12. imprimir a conclusão no terminal.

---

## 9.2 Funções principais de `main.py`

### `print_board(title: str, board: list[list[int]]) -> None`

Imprime um tabuleiro no terminal em formato de matriz.

Exemplo de saída:

```text
Tabuleiro inicial aleatório
[4, 1, 3, 0]
[0, 0, 4, 0]
[1, 0, 2, 0]
[2, 0, 0, 3]
```

Essa função foi criada para atender ao requisito do trabalho de apresentar o tabuleiro em formato de matriz.

---

### `main() -> None`

Executa todo o projeto.

O início da função prepara a pasta de saída:

```python
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)
```

Depois, gera o dataset:

```python
X, y, valid_solutions = make_dataset(negative_multiplier=6, random_seed=42)
```

Em seguida, treina a RNA:

```python
model, metrics = train_model(X, y, random_seed=42)
```

Depois, gera o puzzle inicial:

```python
complete_solution = random.choice(valid_solutions)
puzzle = make_puzzle_from_solution(complete_solution, clues_to_keep=8, random_seed=7)
```

Então, resolve o Sudoku:

```python
solved = solve_with_constraints(puzzle)
```

Por fim, valida o resultado:

```python
logical_validity = is_valid_complete_board(solved)
neural_probability = neural_validation(model, solved)
```

Essa função mostra claramente a integração entre raciocínio simbólico e validação neural.

---

# 10. Arquivo `requirements.txt`

Este arquivo lista as bibliotecas externas necessárias.

```text
numpy
scikit-learn
matplotlib
```

## 10.1 Bibliotecas usadas

### `numpy`

Usada para trabalhar com vetores e arrays numéricos. No projeto, aparece principalmente na codificação one-hot e na montagem do dataset.

### `scikit-learn`

Usada para criar e treinar a RNA multicamadas por meio do `MLPClassifier`. Também fornece as métricas de avaliação e a divisão treino/teste.

### `matplotlib`

Usada para gerar as imagens dos tabuleiros e da curva de treinamento.

---

# 11. Saídas geradas pelo projeto

Após executar `python src/main.py`, a pasta `outputs/` recebe os arquivos:

```text
outputs/
├── curva_treinamento.png
├── tabuleiro_inicial.png
└── tabuleiro_resolvido.png
```

## 11.1 `curva_treinamento.png`

Mostra a evolução da perda da RNA ao longo do treinamento.

Essa imagem ajuda a verificar se o treinamento convergiu.

## 11.2 `tabuleiro_inicial.png`

Mostra o puzzle criado aleatoriamente a partir de uma solução válida.

As células removidas aparecem em branco.

## 11.3 `tabuleiro_resolvido.png`

Mostra o tabuleiro final encontrado pelo resolvedor.

Esse tabuleiro é então avaliado por duas formas:

- validação lógica, com `is_valid_complete_board()`;
- validação neural, com `neural_validation()`.

---

# 12. Funções mais importantes do projeto

Embora todos os módulos sejam relevantes, algumas funções são centrais para compreender a solução.

## 12.1 `is_valid_complete_board()`

É o validador lógico principal. Ele define o que significa um Sudoku correto.

Sem essa função, o projeto não teria uma referência confiável para dizer se um tabuleiro é válido ou inválido.

---

## 12.2 `is_safe()`

É a função que impede jogadas inválidas durante a geração e a resolução.

Ela representa a aplicação local das restrições do Sudoku.

---

## 12.3 `generate_all_valid_solutions()`

Gera todas as soluções válidas 4x4.

Ela é essencial para criar os exemplos positivos do dataset.

---

## 12.4 `board_to_one_hot()`

Transforma a matriz Sudoku em entrada numérica para a RNA.

Sem essa transformação, a rede não conseguiria processar corretamente os símbolos do tabuleiro.

---

## 12.5 `make_dataset()`

Monta o conjunto completo de treinamento e teste.

É a ponte entre as regras lógicas do Sudoku e o aprendizado supervisionado da RNA.

---

## 12.6 `create_mlp()`

Define a arquitetura da rede neural.

É onde são escolhidas as camadas, a ativação, o otimizador e os critérios de parada.

---

## 12.7 `train_model()`

Executa o treinamento e retorna as métricas.

Essa função permite avaliar se a RNA aprendeu a distinguir tabuleiros válidos de inválidos.

---

## 12.8 `solve_with_constraints()`

Resolve o tabuleiro inicial usando busca com restrições.

É a função que efetivamente transforma o puzzle parcial em uma solução completa.

---

## 12.9 `neural_validation()`

Usa a RNA treinada para reconhecer se a solução final parece válida.

Essa função evidencia o papel da RNA dentro do projeto: classificar/reconhecer validade, não substituir completamente o raciocínio lógico.

---

## 12.10 `plot_board()`

Gera imagens dos tabuleiros.

Ela ajuda a deixar a entrega mais visual e adequada para documentação acadêmica.

---

# 13. Por que a solução foi feita de forma híbrida?

A solução não usa apenas RNA porque o Sudoku exige garantias formais de validade.

Uma MLP pode aprender padrões, mas não possui, por si só, mecanismos explícitos para garantir que:

- cada número apareça exatamente uma vez por linha;
- cada número apareça exatamente uma vez por coluna;
- cada número apareça exatamente uma vez por subgrupo;
- uma decisão parcial não comprometa decisões futuras.

Por isso, o projeto separa as tarefas:

| Tarefa | Técnica usada |
|---|---|
| Definir validade | Regras lógicas |
| Gerar exemplos positivos | Backtracking |
| Gerar exemplos negativos | Aleatoriedade e corrupção de soluções |
| Aprender padrão de validade | RNA multicamadas |
| Resolver puzzle parcial | Busca com restrições |
| Reconhecer solução final | RNA treinada |

Essa abordagem é defensável academicamente porque reconhece a limitação da RNA e usa cada técnica onde ela faz mais sentido.

---

# 14. Relação entre os requisitos do trabalho e o código

| Requisito do enunciado | Onde é atendido |
|---|---|
| Garantir que cada célula tenha apenas um número de `S` | `is_value_valid()`, `is_complete()`, `board_to_one_hot()` |
| Evitar repetição em linhas, colunas e subgrupos | `valid_group()`, `is_valid_complete_board()`, `is_safe()` |
| Garantir que cada linha e coluna contenha os números de `S` | `valid_group()` aplicado em linhas e colunas |
| Apresentar dataset de treinamento e teste | `make_dataset()` e `train_model()` |
| Gerar tabuleiros iniciais aleatórios | `make_puzzle_from_solution()` |
| Gerar solução final | `solve_with_constraints()` |
| Usar RNA multicamadas | `create_mlp()` e `train_model()` |
| Validar solução com a RNA treinada | `neural_validation()` |
| Gerar imagens | `plot_board()` e `plot_loss_curve()` |
| Discutir dificuldade de generalização NxN | README e este documento |

---

# 15. Observações sobre generalização para NxN

O código foi construído para Sudoku 4x4, mas algumas decisões já facilitam a discussão sobre generalização.

As constantes principais estão concentradas em `sudoku_rules.py`:

```python
N = 4
BLOCK = 2
SYMBOLS = {1, 2, 3, 4}
```

Para generalizar, seria necessário transformar essas constantes em parâmetros.

Por exemplo, para Sudoku 9x9:

```python
N = 9
BLOCK = 3
SYMBOLS = {1, 2, 3, 4, 5, 6, 7, 8, 9}
```

No entanto, apenas alterar essas variáveis não basta.

Seriam necessárias mudanças em vários pontos:

1. adaptar a codificação one-hot para tamanho `N × N × N`;
2. evitar enumerar todas as soluções válidas, pois isso se torna inviável;
3. melhorar a geração de exemplos negativos;
4. usar heurísticas mais fortes no resolvedor;
5. possivelmente trocar a MLP por uma arquitetura mais adequada a relações entre células;
6. tratar custo computacional muito maior;
7. criar um gerador de puzzles com unicidade de solução.

Assim, a generalização é mais conceitual do que simplesmente trocar `4` por `9`.

---

# 16. Limitação central da RNA no projeto

A RNA aprende a classificar tabuleiros completos como válidos ou inválidos.

Ela não aprende, neste projeto, a preencher passo a passo um tabuleiro parcial.

Isso foi uma decisão consciente, porque gerar diretamente uma solução final a partir de um puzzle parcial exigiria uma modelagem mais complexa, por exemplo:

- saída com 16 posições categóricas;
- tratamento de células fixas que não podem ser alteradas;
- função de perda que penalize violações de linha, coluna e bloco;
- maior volume de pares entrada/solução;
- arquitetura que represente relações espaciais e simbólicas.

Portanto, o papel da RNA aqui é reconhecer validade, enquanto a resolução propriamente dita é feita por raciocínio simbólico.

Essa escolha também ajuda a responder à pergunta do professor sobre o problema de “gerar amostras e testá-las” quando o problema é tratado como raciocínio.

---

# 17. Conclusão técnica

O projeto foi estruturado para mostrar que uma RNA multicamadas pode ser usada em um problema lógico como o Sudoku, mas com uma função bem delimitada.

A solução final não tenta esconder a dificuldade do problema. Pelo contrário, ela evidencia que:

- Sudoku é um problema de restrições;
- RNA é útil para reconhecer padrões;
- backtracking é adequado para garantir soluções válidas;
- a combinação das duas abordagens é mais robusta do que usar apenas amostragem aleatória;
- a generalização para NxN exige mudanças estruturais, não apenas aumento de tamanho.

Assim, o projeto atende ao Sudoku 4x4 e também apresenta uma discussão honesta sobre as dificuldades de generalizar a solução para dimensões maiores.
