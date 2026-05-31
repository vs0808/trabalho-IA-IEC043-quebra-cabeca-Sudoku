# Explicação detalhada das imagens geradas

Este documento explica, de forma detalhada, as imagens produzidas pela execução do projeto de resolução de Sudoku 4x4 com Rede Neural Artificial multicamadas. As imagens ficam armazenadas na pasta `outputs/` e servem como apoio visual para demonstrar três momentos importantes da solução:

1. o tabuleiro inicial parcial gerado aleatoriamente;
2. o tabuleiro final resolvido pelo algoritmo de busca com restrições;
3. a curva de perda da RNA durante o treinamento.

As imagens foram geradas automaticamente pelo módulo `src/visualization.py`, chamado pelo arquivo principal `src/main.py`. Elas não são apenas ilustrações: cada uma documenta uma etapa do experimento computacional e ajuda a justificar o comportamento do sistema.

---

## 1. Visão geral das imagens do projeto

Durante a execução do arquivo `src/main.py`, três arquivos de imagem são criados dentro da pasta `outputs/`:

```text
outputs/
├── tabuleiro_inicial.png
├── tabuleiro_resolvido.png
└── curva_treinamento.png
```

Cada imagem tem uma função específica dentro do trabalho:

| Imagem | Arquivo | Função dentro do projeto |
|---|---|---|
| Tabuleiro inicial | `tabuleiro_inicial.png` | Mostra o Sudoku 4x4 parcial que será resolvido. As células vazias aparecem sem número. |
| Tabuleiro resolvido | `tabuleiro_resolvido.png` | Mostra a solução completa encontrada pelo resolvedor lógico e reconhecida pela RNA como válida. |
| Curva de treinamento | `curva_treinamento.png` | Mostra a evolução da função de perda da RNA ao longo das épocas de treinamento. |

Essas três imagens representam, juntas, o caminho completo do experimento:

```text
Dataset -> Treinamento da RNA -> Geração de puzzle -> Resolução -> Validação -> Imagens
```

A imagem do tabuleiro inicial mostra a entrada parcial do problema. A imagem do tabuleiro resolvido mostra a saída final. A imagem da curva de treinamento mostra o processo de aprendizagem da RNA usada para reconhecer tabuleiros válidos.

---

## 2. Onde as imagens são geradas no código

As imagens são geradas pelo módulo:

```text
src/visualization.py
```

Esse módulo possui duas funções principais:

```python
plot_board(board, title, output_path)
plot_loss_curve(loss_curve, output_path)
```

A função `plot_board` é responsável por desenhar tabuleiros Sudoku 4x4, enquanto `plot_loss_curve` é responsável por desenhar o gráfico da perda da RNA ao longo do treinamento.

No arquivo `src/main.py`, essas funções são chamadas em três momentos:

```python
plot_loss_curve(metrics["loss_curve"], os.path.join(output_dir, "curva_treinamento.png"))
```

Essa chamada gera a imagem da curva de treinamento.

```python
plot_board(puzzle, "Tabuleiro inicial aleatório", os.path.join(output_dir, "tabuleiro_inicial.png"))
```

Essa chamada gera a imagem do tabuleiro inicial.

```python
plot_board(solved, "Solução final encontrada", os.path.join(output_dir, "tabuleiro_resolvido.png"))
```

Essa chamada gera a imagem do tabuleiro resolvido.

Portanto, as imagens são consequência direta da execução completa do experimento. Elas não foram desenhadas manualmente, mas sim produzidas pelo próprio código, a partir dos dados gerados e processados pelo sistema.

---

# 3. Imagem `tabuleiro_inicial.png`

## 3.1. O que a imagem representa

A imagem `tabuleiro_inicial.png` representa o tabuleiro inicial do Sudoku 4x4, isto é, o problema parcial que será entregue ao resolvedor.

No projeto, o tabuleiro inicial é gerado a partir de uma solução completa válida. O código escolhe uma solução válida dentre as soluções geradas para o Sudoku 4x4 e remove alguns números aleatoriamente. As posições removidas passam a ser representadas internamente pelo valor `0`, indicando célula vazia.

Exemplo de tabuleiro inicial gerado na execução:

```python
[
    [4, 1, 3, 0],
    [0, 0, 4, 0],
    [1, 0, 2, 0],
    [2, 0, 0, 3]
]
```

Na imagem, os valores diferentes de zero aparecem como números visíveis. Já os zeros não são desenhados; por isso, as células vazias aparecem em branco.

---

## 3.2. Relação entre a matriz e a imagem

A matriz do tabuleiro inicial possui quatro linhas e quatro colunas:

```text
Linha 1: 4  1  3  _
Linha 2: _  _  4  _
Linha 3: 1  _  2  _
Linha 4: 2  _  _  3
```

Na visualização:

- cada linha da matriz corresponde a uma linha horizontal da imagem;
- cada coluna da matriz corresponde a uma coluna vertical da imagem;
- o caractere `_`, usado apenas na explicação, corresponde a uma célula vazia;
- no código, a célula vazia é representada por `0`;
- na imagem, a célula vazia aparece sem número.

Essa distinção é importante: o valor `0` não faz parte do conjunto de símbolos do Sudoku. O conjunto válido do problema é:

```python
S = {1, 2, 3, 4}
```

Portanto, o `0` é apenas uma marca computacional para indicar ausência temporária de valor.

---

## 3.3. Como o tabuleiro inicial é gerado

A geração do tabuleiro inicial ocorre na função `make_puzzle_from_solution`, localizada no arquivo `src/solver.py`.

A função recebe uma solução completa válida e remove alguns valores:

```python
def make_puzzle_from_solution(solution: Board, clues_to_keep: int = 8, random_seed: Optional[int] = None) -> Board:
```

O parâmetro mais importante dessa função é:

```python
clues_to_keep=8
```

Isso significa que, de um total de 16 células do Sudoku 4x4, o tabuleiro inicial manterá 8 números preenchidos e terá 8 células vazias.

Como o Sudoku 4x4 possui 16 células:

```text
4 linhas × 4 colunas = 16 células
```

E como são mantidas 8 pistas:

```text
16 células - 8 pistas = 8 células removidas
```

O código faz isso da seguinte forma:

```python
cells_to_remove = (N * N) - clues_to_keep
```

Como `N = 4`, temos:

```python
cells_to_remove = (4 * 4) - 8
cells_to_remove = 16 - 8
cells_to_remove = 8
```

Depois, o código sorteia posições do tabuleiro e substitui os valores dessas posições por zero:

```python
for row, col in positions[:cells_to_remove]:
    puzzle[row][col] = 0
```

Esse processo cria um tabuleiro parcial, mas derivado de uma solução real. Isso aumenta a chance de o tabuleiro possuir solução, pois ele não é montado de forma totalmente aleatória a partir do zero.

---

## 3.4. Por que gerar o puzzle a partir de uma solução válida

A decisão de gerar o tabuleiro inicial a partir de uma solução válida foi tomada por razões técnicas e pedagógicas.

Se o projeto gerasse um tabuleiro parcial completamente aleatório, poderia acontecer de o tabuleiro:

- não ter nenhuma solução;
- ter contradições já nas pistas iniciais;
- ter múltiplas soluções;
- não representar adequadamente um Sudoku bem-formado;
- dificultar a avaliação da RNA e do resolvedor.

Ao partir de uma solução válida e remover números, o projeto garante que existe pelo menos uma solução possível: a solução original antes da remoção.

Isso não significa, obrigatoriamente, que o puzzle gerado terá solução única. Para garantir unicidade, seria necessário adicionar uma etapa extra de verificação, testando se apenas uma solução é possível. Como o foco do trabalho é a construção da RNA e a discussão sobre resolução/generalização, o projeto opta por uma estratégia mais simples e controlada.

---

## 3.5. Interpretação visual das linhas grossas e finas

Na imagem do tabuleiro inicial, é possível perceber que algumas linhas são mais grossas que outras.

Essa escolha visual representa a estrutura do Sudoku 4x4:

- a grade principal tem tamanho 4x4;
- os subgrupos internos têm tamanho 2x2;
- as linhas grossas delimitam os blocos 2x2;
- as linhas finas separam células individuais dentro da grade.

O código que define a espessura das linhas está em `plot_board`:

```python
for i in range(N + 1):
    line_width = 2.5 if i % BLOCK == 0 else 1.0
    ax.plot([0, N], [i, i], linewidth=line_width)
    ax.plot([i, i], [0, N], linewidth=line_width)
```

Como `BLOCK = 2`, as linhas de índice 0, 2 e 4 são desenhadas com maior espessura. Isso cria a divisão visual dos quatro subgrupos:

```text
┌─────┬─────┐
│ 2x2 │ 2x2 │
├─────┼─────┤
│ 2x2 │ 2x2 │
└─────┴─────┘
```

Essa representação é importante porque o Sudoku não exige apenas que linhas e colunas sejam válidas. Ele também exige que cada subgrupo 2x2 contenha os símbolos `{1, 2, 3, 4}` sem repetição.

---

## 3.6. Relação da imagem com os requisitos do trabalho

A imagem `tabuleiro_inicial.png` está diretamente relacionada ao requisito 5 do enunciado:

> Gerar tabuleiros iniciais aleatórios que forneçam, em formato de matriz, o tabuleiro treinado que a RNA treinada reconheça como válido e gere uma solução final.

A imagem mostra visualmente esse tabuleiro inicial aleatório. O mesmo tabuleiro também é impresso no terminal em formato de matriz, por meio da função `print_board`, em `src/main.py`.

Assim, o projeto apresenta o tabuleiro inicial em duas formas:

1. matriz textual no terminal;
2. imagem visual na pasta `outputs/`.

Essa dupla representação ajuda tanto na avaliação técnica quanto na apresentação do trabalho.

---

# 4. Imagem `tabuleiro_resolvido.png`

## 4.1. O que a imagem representa

A imagem `tabuleiro_resolvido.png` representa a solução final encontrada para o tabuleiro inicial.

Exemplo de solução gerada na execução:

```python
[
    [4, 1, 3, 2],
    [3, 2, 4, 1],
    [1, 3, 2, 4],
    [2, 4, 1, 3]
]
```

Diferentemente do tabuleiro inicial, essa matriz não possui zeros. Todas as células estão preenchidas com valores pertencentes ao conjunto:

```python
S = {1, 2, 3, 4}
```

A imagem mostra, portanto, um Sudoku 4x4 completo.

---

## 4.2. Validação das linhas

Para que a solução seja válida, cada linha deve conter exatamente os símbolos `{1, 2, 3, 4}`, sem repetição.

No exemplo:

```text
Linha 1: 4  1  3  2
Linha 2: 3  2  4  1
Linha 3: 1  3  2  4
Linha 4: 2  4  1  3
```

Verificando cada linha:

```text
Linha 1 → {4, 1, 3, 2} = {1, 2, 3, 4}
Linha 2 → {3, 2, 4, 1} = {1, 2, 3, 4}
Linha 3 → {1, 3, 2, 4} = {1, 2, 3, 4}
Linha 4 → {2, 4, 1, 3} = {1, 2, 3, 4}
```

Logo, nenhuma linha possui repetição e nenhuma linha deixa de conter algum símbolo do conjunto `S`.

Esse comportamento atende ao requisito de que cada linha da grade principal 4x4 contenha os números do conjunto permitido sem repetição.

---

## 4.3. Validação das colunas

Além das linhas, cada coluna também deve conter exatamente os símbolos `{1, 2, 3, 4}`.

No exemplo resolvido:

```text
Coluna 1: 4  3  1  2
Coluna 2: 1  2  3  4
Coluna 3: 3  4  2  1
Coluna 4: 2  1  4  3
```

Verificando cada coluna:

```text
Coluna 1 → {4, 3, 1, 2} = {1, 2, 3, 4}
Coluna 2 → {1, 2, 3, 4} = {1, 2, 3, 4}
Coluna 3 → {3, 4, 2, 1} = {1, 2, 3, 4}
Coluna 4 → {2, 1, 4, 3} = {1, 2, 3, 4}
```

Assim, a solução também respeita a regra das colunas.

---

## 4.4. Validação dos subgrupos 2x2

O Sudoku 4x4 é dividido em quatro subgrupos 2x2. Cada subgrupo também precisa conter os símbolos `{1, 2, 3, 4}` sem repetição.

A solução pode ser visualmente dividida assim:

```text
[4, 1] | [3, 2]
[3, 2] | [4, 1]
------+------
[1, 3] | [2, 4]
[2, 4] | [1, 3]
```

Subgrupo superior esquerdo:

```text
4  1
3  2
```

Valores:

```text
{4, 1, 3, 2} = {1, 2, 3, 4}
```

Subgrupo superior direito:

```text
3  2
4  1
```

Valores:

```text
{3, 2, 4, 1} = {1, 2, 3, 4}
```

Subgrupo inferior esquerdo:

```text
1  3
2  4
```

Valores:

```text
{1, 3, 2, 4} = {1, 2, 3, 4}
```

Subgrupo inferior direito:

```text
2  4
1  3
```

Valores:

```text
{2, 4, 1, 3} = {1, 2, 3, 4}
```

Portanto, todos os subgrupos 2x2 estão corretos.

---

## 4.5. Como a solução final é encontrada

A imagem `tabuleiro_resolvido.png` não é produzida diretamente pela RNA. Ela é produzida após a resolução do puzzle pela função:

```python
solve_with_constraints(puzzle)
```

Essa função está localizada em `src/solver.py` e usa uma técnica de busca com restrições. O método utilizado é o backtracking com uma heurística chamada MRV.

MRV significa `Minimum Remaining Values`, ou, em português, “menor número de valores restantes”. A ideia é escolher primeiro a célula vazia que possui menos candidatos possíveis.

Isso reduz o espaço de busca, porque o algoritmo tenta resolver primeiro as partes mais restritas do tabuleiro.

Exemplo conceitual:

```text
Célula A possui candidatos {1, 2, 3}
Célula B possui candidatos {4}
```

A heurística MRV escolhe a célula B antes da célula A, porque ela tem apenas uma possibilidade. Isso torna a busca mais eficiente e diminui a quantidade de tentativas erradas.

No código, essa escolha aparece em:

```python
cells.sort(key=lambda pos: len(candidates_for_cell(working_board, pos[0], pos[1])))
row, col = cells[0]
```

A função `candidates_for_cell` calcula os valores possíveis para uma célula sem violar imediatamente as regras de linha, coluna e subgrupo.

---

## 4.6. Papel da RNA na solução final

Uma decisão importante do projeto é separar duas tarefas:

1. resolver o Sudoku;
2. reconhecer se um tabuleiro completo é válido.

A resolução é feita por busca com restrições. A validação probabilística é feita pela RNA.

Isso significa que a RNA não está sendo usada como único mecanismo de raciocínio lógico para preencher o Sudoku. Ela é usada como classificador supervisionado, treinado para distinguir tabuleiros completos válidos de tabuleiros completos inválidos.

Depois que o resolvedor encontra a solução final, o código calcula a probabilidade de a RNA considerar aquele tabuleiro válido:

```python
neural_probability = neural_validation(model, solved)
```

A função `neural_validation` transforma o tabuleiro resolvido em codificação one-hot e consulta a RNA treinada:

```python
encoded = board_to_one_hot(board).reshape(1, -1)
probabilities = model.predict_proba(encoded)[0]
```

No exemplo executado, a RNA estimou alta probabilidade para a classe válida:

```text
Probabilidade estimada pela RNA para classe 'válido': 0.9923
```

Isso indica que a RNA reconheceu a solução final como pertencente à classe dos tabuleiros válidos.

---

## 4.7. Relação da imagem resolvida com os requisitos do trabalho

A imagem `tabuleiro_resolvido.png` atende diretamente aos requisitos relacionados à validade do Sudoku:

1. cada célula possui apenas um número de `S`;
2. não há repetição em linhas, colunas e subgrupos;
3. cada linha e cada coluna da grade 4x4 contém todos os números de `S`;
4. o tabuleiro final é reconhecido pela RNA como válido.

Além disso, ela ajuda a demonstrar que o sistema não apenas classifica tabuleiros, mas também parte de um tabuleiro inicial parcial e chega a uma solução final completa.

---

# 5. Imagem `curva_treinamento.png`

## 5.1. O que a imagem representa

A imagem `curva_treinamento.png` mostra a curva de perda da Rede Neural Artificial durante o treinamento.

A curva de perda é um gráfico que indica como o erro do modelo evoluiu ao longo das épocas de treinamento.

No eixo horizontal aparece:

```text
Época
```

No eixo vertical aparece:

```text
Loss
```

A época representa uma iteração de treinamento do modelo sobre os dados. A perda, ou `loss`, representa o erro do modelo em relação ao objetivo de classificação.

De forma simplificada:

```text
Loss alta  → a rede está errando mais
Loss baixa → a rede está errando menos
```

Portanto, uma curva decrescente indica que a RNA está aprendendo a separar os tabuleiros válidos dos inválidos.

---

## 5.2. Como a curva é obtida

A RNA utilizada no projeto é um `MLPClassifier`, da biblioteca `scikit-learn`.

A arquitetura está definida no arquivo `src/model.py`, na função `create_mlp`:

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

Após o treinamento, o próprio modelo armazena os valores da perda em:

```python
model.loss_curve_
```

No projeto, essa curva é guardada dentro do dicionário de métricas:

```python
"loss_curve": model.loss_curve_
```

Depois, o arquivo `src/main.py` chama:

```python
plot_loss_curve(metrics["loss_curve"], os.path.join(output_dir, "curva_treinamento.png"))
```

Essa chamada gera a imagem final da curva de treinamento.

---

## 5.3. Interpretação da perda no contexto do Sudoku

No contexto deste trabalho, a RNA recebe como entrada um tabuleiro completo codificado em one-hot e deve prever se ele é válido ou inválido.

Cada tabuleiro 4x4 possui 16 células. Cada célula pode receber um dos quatro símbolos `{1, 2, 3, 4}`. Por isso, a entrada da rede tem:

```text
16 células × 4 símbolos = 64 posições
```

A codificação one-hot transforma cada célula em quatro posições binárias.

Por exemplo, considerando os símbolos `{1, 2, 3, 4}`:

```text
Valor 1 → [1, 0, 0, 0]
Valor 2 → [0, 1, 0, 0]
Valor 3 → [0, 0, 1, 0]
Valor 4 → [0, 0, 0, 1]
```

Assim, um tabuleiro completo é transformado em um vetor numérico de 64 posições. Esse vetor é usado como entrada para a RNA.

A saída esperada é binária:

```text
1 → tabuleiro válido
0 → tabuleiro inválido
```

Durante o treinamento, a rede ajusta seus pesos para reduzir a diferença entre a classe prevista e a classe correta. A curva de perda mostra esse processo de ajuste.

---

## 5.4. Por que a curva geralmente diminui

Se o treinamento está funcionando, espera-se que a perda diminua com o passar das épocas.

Isso acontece porque a RNA começa com pesos aleatórios. No início, ela ainda não aprendeu os padrões que diferenciam um Sudoku válido de um inválido. Ao observar exemplos do dataset, a rede ajusta os pesos internos para melhorar sua classificação.

No projeto, o dataset possui exemplos positivos e negativos:

```text
Exemplos positivos: tabuleiros Sudoku 4x4 válidos
Exemplos negativos: tabuleiros completos inválidos
```

Os exemplos positivos são gerados pela função:

```python
generate_all_valid_solutions()
```

Os exemplos negativos são gerados por duas estratégias:

1. tabuleiros completos aleatórios;
2. tabuleiros válidos corrompidos.

Essa segunda estratégia é importante porque cria exemplos negativos parecidos com soluções reais. Sem isso, a rede poderia aprender apenas a rejeitar tabuleiros muito desorganizados, mas teria dificuldade com erros sutis.

A curva de perda reflete justamente o esforço da RNA para aprender essa separação.

---

## 5.5. Relação entre perda e acurácia

A perda e a acurácia não são a mesma coisa.

A perda mede o erro contínuo do modelo durante o treinamento. A acurácia mede a proporção de acertos em um conjunto de teste.

No exemplo executado, o projeto obteve:

```text
Acurácia no teste: 0.9901
```

Isso significa que, no conjunto de teste, a RNA classificou corretamente cerca de 99% dos tabuleiros.

A curva de perda ajuda a interpretar se essa acurácia foi resultado de um treinamento coerente. Uma curva de perda que diminui indica que a rede realmente ajustou seus parâmetros para reduzir o erro.

Por outro lado, se a curva estivesse instável, oscilando muito ou crescendo, isso poderia indicar problemas como:

- taxa de aprendizado inadequada;
- arquitetura mal dimensionada;
- dataset ruim;
- dados mal codificados;
- treinamento insuficiente;
- excesso de ruído nos exemplos negativos.

---

## 5.6. Early stopping e parada do treinamento

A RNA foi configurada com `early_stopping=True`.

Isso significa que o treinamento pode ser interrompido antes de atingir o número máximo de épocas, caso o modelo pare de melhorar em um conjunto de validação interno.

No código:

```python
early_stopping=True,
validation_fraction=0.15,
n_iter_no_change=40,
```

Esses parâmetros significam:

- `validation_fraction=0.15`: 15% dos dados de treinamento são separados internamente para validação;
- `n_iter_no_change=40`: se o modelo não melhorar por 40 épocas consecutivas, o treinamento pode parar;
- `early_stopping=True`: ativa esse mecanismo de parada antecipada.

Por isso, a imagem da curva pode ter menos de 1000 épocas, mesmo que `max_iter=1000` esteja configurado. O limite de 1000 é apenas o máximo permitido, não necessariamente o número real de épocas executadas.

---

## 5.7. O que a curva não prova sozinha

A curva de treinamento é importante, mas ela não prova sozinha que o modelo “raciocina” como um ser humano ou como um resolvedor lógico.

Ela mostra que a RNA reduziu o erro em um problema de classificação supervisionada. Isso é diferente de afirmar que a rede aprendeu uma regra geral perfeita do Sudoku.

No projeto, essa diferença é fundamental.

A RNA aprende padrões estatísticos presentes no dataset. Ela pode reconhecer com alta probabilidade tabuleiros válidos semelhantes aos exemplos de treinamento. Porém, isso não garante, por si só, que ela seja capaz de resolver qualquer Sudoku parcial ou generalizar para qualquer dimensão `N x N`.

Por esse motivo, o projeto usa a RNA como reconhecedora de validade e mantém um resolvedor lógico separado para construir a solução.

---

# 6. Relação entre as três imagens

As três imagens devem ser lidas como uma sequência narrativa do experimento.

## 6.1. Primeira etapa: aprendizagem

A imagem `curva_treinamento.png` mostra que a RNA passou por uma etapa de treinamento.

Nessa etapa, ela aprendeu a classificar tabuleiros completos em duas classes:

```text
válido
inválido
```

A curva de perda documenta visualmente esse processo.

---

## 6.2. Segunda etapa: problema inicial

A imagem `tabuleiro_inicial.png` mostra o problema parcial que será resolvido.

Esse tabuleiro contém algumas pistas fixas e algumas células vazias.

As pistas funcionam como restrições. O resolvedor não pode alterar esses valores iniciais. Ele deve preencher apenas as células vazias, respeitando as regras do Sudoku.

---

## 6.3. Terceira etapa: solução final

A imagem `tabuleiro_resolvido.png` mostra o resultado final.

Esse resultado é verificado de duas formas:

1. validação lógica, por meio da função `is_valid_complete_board`;
2. validação neural, por meio da função `neural_validation`.

A validação lógica garante que o tabuleiro cumpre formalmente as regras do Sudoku. A validação neural mostra que a RNA treinada também reconhece o tabuleiro como válido.

---

# 7. Relação das imagens com o problema de raciocínio

O enunciado questiona a dificuldade de gerar amostras e testá-las quando o problema é tratado como raciocínio.

As imagens ajudam a explicar essa diferença.

A imagem do tabuleiro inicial mostra um problema parcial. Resolver esse problema exige escolher valores que satisfaçam restrições simultâneas.

A imagem do tabuleiro resolvido mostra uma configuração final válida. Porém, reconhecer uma configuração final válida não é o mesmo que descobrir como chegar até ela.

A imagem da curva de treinamento mostra que a RNA foi treinada para classificar exemplos. Ela aprende com amostras.

Portanto, há uma diferença entre:

```text
Gerar e testar amostras completas
```

e

```text
Raciocinar passo a passo sobre um tabuleiro parcial
```

No Sudoku, o número de combinações possíveis cresce rapidamente. Mesmo no 4x4, há muitas configurações possíveis. Em dimensões maiores, como 9x9 ou 16x16, gerar amostras aleatórias e testá-las se torna cada vez menos viável.

Por isso, o projeto adota uma solução híbrida:

- usa regras lógicas para resolver o puzzle;
- usa a RNA para reconhecer se o resultado completo parece válido;
- usa imagens para documentar entrada, treinamento e saída.

---

# 8. Detalhamento técnico da função `plot_board`

A função responsável pelas imagens dos tabuleiros é:

```python
def plot_board(board: Board, title: str, output_path: str) -> None:
```

Ela recebe três parâmetros:

| Parâmetro | Significado |
|---|---|
| `board` | Matriz 4x4 que será desenhada. |
| `title` | Título exibido na parte superior da figura. |
| `output_path` | Caminho onde a imagem será salva. |

Internamente, a função cria uma figura com Matplotlib:

```python
fig, ax = plt.subplots(figsize=(4, 4))
```

Depois configura o espaço visual:

```python
ax.set_xlim(0, N)
ax.set_ylim(0, N)
ax.set_xticks([])
ax.set_yticks([])
```

Essas linhas removem os eixos numéricos, porque a imagem deve parecer um tabuleiro, não um gráfico cartesiano.

Em seguida, a função desenha as linhas da grade:

```python
for i in range(N + 1):
    line_width = 2.5 if i % BLOCK == 0 else 1.0
    ax.plot([0, N], [i, i], linewidth=line_width)
    ax.plot([i, i], [0, N], linewidth=line_width)
```

Por fim, escreve os números nas células preenchidas:

```python
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
```

A expressão:

```python
N - row - 0.5
```

é usada porque o sistema de coordenadas do Matplotlib cresce de baixo para cima, enquanto matrizes em Python são lidas de cima para baixo. Essa transformação garante que a primeira linha da matriz apareça no topo da imagem, como esperado em um tabuleiro de Sudoku.

---

# 9. Detalhamento técnico da função `plot_loss_curve`

A função responsável pela curva de treinamento é:

```python
def plot_loss_curve(loss_curve: List[float], output_path: str) -> None:
```

Ela recebe:

| Parâmetro | Significado |
|---|---|
| `loss_curve` | Lista de valores de perda registrados durante o treinamento. |
| `output_path` | Caminho onde a imagem será salva. |

A função cria uma figura:

```python
fig, ax = plt.subplots(figsize=(6, 4))
```

Depois desenha a curva:

```python
ax.plot(loss_curve)
```

Configura título e eixos:

```python
ax.set_title("Curva de perda da RNA")
ax.set_xlabel("Época")
ax.set_ylabel("Loss")
```

Ativa a grade de leitura:

```python
ax.grid(True)
```

E salva a imagem:

```python
plt.savefig(output_path, dpi=160)
```

O parâmetro `dpi=160` melhora a resolução da imagem salva, tornando-a mais adequada para apresentação em relatório ou slides.

---

# 10. Por que as imagens são úteis para a apresentação do trabalho

As imagens ajudam a transformar um experimento computacional em uma explicação visual.

Sem elas, o avaliador teria apenas números, matrizes e código. Com elas, é possível enxergar:

- o problema antes da resolução;
- o resultado depois da resolução;
- o comportamento do treinamento da RNA;
- a divisão dos subgrupos 2x2;
- a ausência ou presença de células vazias;
- a redução do erro durante o treinamento.

Além disso, as imagens mostram que o projeto não se limita a treinar uma RNA abstrata. Ele possui uma entrada concreta, um processo de resolução e uma saída verificável.

---

# 11. Observações sobre generalização das imagens para Sudoku NxN

As imagens atuais foram feitas para Sudoku 4x4 com subgrupos 2x2.

Para generalizar a visualização para Sudoku `N x N`, seria necessário alterar alguns pontos do projeto.

Atualmente, as constantes são:

```python
N = 4
BLOCK = 2
SYMBOLS = {1, 2, 3, 4}
```

Para um Sudoku 9x9 tradicional, teríamos:

```python
N = 9
BLOCK = 3
SYMBOLS = {1, 2, 3, 4, 5, 6, 7, 8, 9}
```

A função `plot_board` já foi escrita de forma parcialmente parametrizada, pois usa `N` e `BLOCK` para desenhar a grade. Portanto, a lógica visual poderia ser adaptada.

No entanto, algumas dificuldades surgiriam:

1. a imagem precisaria comportar mais células;
2. o tamanho da fonte teria que ser ajustado;
3. a figura precisaria aumentar de dimensão;
4. o número de símbolos poderia passar de um dígito em Sudokus maiores;
5. a codificação one-hot da RNA cresceria muito;
6. o dataset seria muito mais difícil de gerar exaustivamente;
7. a curva de treinamento poderia exigir mais épocas e mais exemplos.

Assim, a visualização é uma das partes mais simples de generalizar. A dificuldade principal está no crescimento combinatório do problema, no dataset e na arquitetura da RNA.

---

# 12. Conclusão

As imagens geradas pelo projeto documentam três dimensões fundamentais da solução:

1. **Entrada do problema:** representada por `tabuleiro_inicial.png`.
2. **Saída resolvida:** representada por `tabuleiro_resolvido.png`.
3. **Processo de aprendizagem da RNA:** representado por `curva_treinamento.png`.

O tabuleiro inicial mostra o desafio parcial a ser resolvido. O tabuleiro resolvido mostra que as restrições foram satisfeitas. A curva de treinamento mostra que a RNA passou por um processo de aprendizagem supervisionada para reconhecer tabuleiros válidos e inválidos.

Em conjunto, essas imagens reforçam a proposta do trabalho: resolver o Sudoku 4x4, usar uma RNA multicamadas como reconhecedora de validade, gerar exemplos de treinamento e teste, produzir tabuleiros iniciais aleatórios e discutir as limitações de generalizar a abordagem para `N x N`.

