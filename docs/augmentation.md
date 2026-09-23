# Data Augmentation de Landmarks

## Por que augmentation?
O Data Augmentation ajuda na capacidade de generalização dos modelos,
gerando amostras sintéticas para gerar uma maior variação nos dados

## O que são os landmarks

O MediaPipe retorna 21 pontos de referência para cada mão detectada, cada um com três coordenadas (x, y, z). Após a normalização pelo pulso, cada frame é representado por um vetor de **63 valores** (21 × 3).

```
Landmark 0  → pulso (origem após normalização: 0, 0, 0)
Landmarks 1–4  → polegar
Landmarks 5–8  → indicador
Landmarks 9–12 → médio
Landmarks 13–16 → anelar
Landmarks 17–20 → mínimo
```

## Transformações feitas

### 1. Ruído Gaussiano

Adiciona uma perturbação aleatória pequena a cada coordenada, seguindo uma distribuição normal com desvio padrão de 0.005 (na escala normalizada dos landmarks).

**O que simula:** o tremor natural da mão durante a execução do sinal e a imprecisão inerente ao sensor do MediaPipe.


### 2. Escala

Multiplica todos os landmarks por um fator aleatório entre 0.85 e 1.15, mantendo a origem (pulso) no mesmo lugar.

**O que simula:** a mão filmada mais perto ou mais longe da câmera. Como já normalizamos pelo pulso, escalar em torno da origem é geometricamente correto — os dedos se afastam ou se aproximam do pulso proporcionalmente, como na vida real.


### 3. Rotação 2D

Rotaciona todos os landmarks no plano XY em torno da origem (pulso), com um ângulo aleatório entre -15° e +15°. O eixo Z não é rotacionado para não distorcer artificialmente a informação de profundidade.

**O que simula:** o pulso levemente inclinado para a esquerda ou direita, ou um ângulo ligeiramente diferente de filmagem.


### 4. Espelhamento Horizontal

Inverte o sinal do eixo X de todos os landmarks (multiplica por -1).

**O que simula:** a mão esquerda realizando o mesmo sinal. É a transformação mais valiosa do conjunto porque cria amostras genuinamente diferentes — um gesto feito com a mão esquerda é o espelho exato do mesmo gesto feito com a mão direita — e por isso é **sempre gerada** para cada amostra original, não apenas como variação aleatória.


## Como as transformações são combinadas

Para cada amostra original, o módulo gera `n_aumentos` variações:

- **1ª variação:** sempre o espelhamento (a mais valiosa)
- **Demais variações:** combinações aleatórias de ruído, escala e rotação — o número e a escolha das transformações muda a cada geração

Isso garante diversidade entre as amostras sintéticas, evitando que sejam cópias umas das outras.

## Aplicação das Transformações no Dataset

Cada amostra é uma **sequência de frames consecutivos**. Ao aplicar augmentation em uma sequência, a **mesma transformação é usada em todos os frames** da sequência.


## Impacto no tamanho do dataset

Com `n_aumentos=5` (padrão), cada amostra original gera 5 variações sintéticas, totalizando 6× mais dados:

| Cenário | Amostras originais | Após augmentation (n=5) |
|---|---|---|
| 3 vídeos, ~2 sequências/vídeo (LSTM) | ~6 sequências/gesto | ~36 sequências/gesto |

## Como o Data Augmentation é aplicado
Após a separação dos dados de treino em Treino/Validação,
os dados retornados de treino passam pelo processo de Data Augmentation,
o qual é usado para o treino do modelo caso a opção de utilizar
`usar_augmentation` no pipeline seja verdadeira.


```
Leitura dos dados de treinamento via CSV.
      ↓
Separação de Dados de Treinamento e Dados de Validação
      ↓
Augmentation nos Dados de Treinamento
      ↓
Treinamento do modelo
```