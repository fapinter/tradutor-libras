# Data Augmentation de Landmarks

## Por que augmentation?

Modelos de machine learning precisam de exemplos variados para aprender a **generalizar** — ou seja, reconhecer um gesto mesmo quando ele é feito de forma ligeiramente diferente do que foi visto no treino. Com poucos vídeos por gesto, o modelo tende a **memorizar** os exemplos exatos em vez de aprender o padrão real do sinal.

A solução é criar variações sintéticas dos dados que já temos.

## Por que nos landmarks e não nas imagens?

A abordagem mais comum de augmentation (usada por bibliotecas como Albumentations) opera nos **pixels da imagem** — girando, iluminando ou distorcendo o frame antes de passá-lo pelo modelo de detecção.

Aqui fazemos diferente: aplicamos as transformações diretamente nos **landmarks já extraídos pelo MediaPipe** (os 21 pontos xyz da mão). Isso tem três vantagens:

- **Muito mais rápido** — não reprocessa nenhuma imagem com o MediaPipe
- **Sem dependências extras** — usa apenas numpy
- **Controle geométrico exato** — sabemos precisamente o que estamos transformando

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

## As quatro transformações

### 1. Ruído Gaussiano

Adiciona uma perturbação aleatória pequena a cada coordenada, seguindo uma distribuição normal com desvio padrão de 0.005 (na escala normalizada dos landmarks).

**O que simula:** o tremor natural da mão durante a execução do sinal e a imprecisão inerente ao sensor do MediaPipe.

```
coordenada_nova = coordenada_original + N(0, 0.005)
```

### 2. Escala

Multiplica todos os landmarks por um fator aleatório entre 0.85 e 1.15, mantendo a origem (pulso) no mesmo lugar.

**O que simula:** a mão filmada mais perto ou mais longe da câmera. Como já normalizamos pelo pulso, escalar em torno da origem é geometricamente correto — os dedos se afastam ou se aproximam do pulso proporcionalmente, como na vida real.

```
landmarks_novos = landmarks_originais × fator,  fator ∈ [0.85, 1.15]
```

### 3. Rotação 2D

Rotaciona todos os landmarks no plano XY em torno da origem (pulso), com um ângulo aleatório entre -15° e +15°. O eixo Z não é rotacionado para não distorcer artificialmente a informação de profundidade.

**O que simula:** o pulso levemente inclinado para a esquerda ou direita, ou um ângulo ligeiramente diferente de filmagem.

```
x_novo =  cos(θ) × x - sin(θ) × y
y_novo =  sin(θ) × x + cos(θ) × y
z_novo =  z  (inalterado)
```

### 4. Espelhamento Horizontal

Inverte o sinal do eixo X de todos os landmarks (multiplica por -1).

**O que simula:** a mão esquerda realizando o mesmo sinal. É a transformação mais valiosa do conjunto porque cria amostras genuinamente diferentes — um gesto feito com a mão esquerda é o espelho exato do mesmo gesto feito com a mão direita — e por isso é **sempre gerada** para cada amostra original, não apenas como variação aleatória.

```
x_novo = -x_original
y_novo =  y_original  (inalterado)
z_novo =  z_original  (inalterado)
```

## Como as transformações são combinadas

Para cada amostra original, o módulo gera `n_aumentos` variações:

- **1ª variação:** sempre o espelhamento (a mais valiosa)
- **Demais variações:** combinações aleatórias de ruído, escala e rotação — o número e a escolha das transformações muda a cada geração

Isso garante diversidade entre as amostras sintéticas, evitando que sejam cópias umas das outras.

## Comportamento para sequências LSTM

Para o modo LSTM, cada amostra é uma **sequência de frames** (ex: 20 frames consecutivos). Ao aplicar augmentation em uma sequência, a **mesma transformação é usada em todos os frames** da sequência.

Isso é fundamental: se cada frame fosse rotacionado com um ângulo diferente, o modelo aprenderia um "movimento fantasma" que nunca existiu nos dados reais, corrompendo a informação temporal que o LSTM precisa aprender.

```
Sequência original:  [frame_0,  frame_1,  ..., frame_19]
                         ↓          ↓               ↓
   mesma rotação de θ° aplicada em todos os frames
                         ↓          ↓               ↓
Sequência aumentada: [frame_0', frame_1', ..., frame_19']
```

## Impacto no tamanho do dataset

Com `n_aumentos=5` (padrão), cada amostra original gera 5 variações sintéticas, totalizando 6× mais dados:

| Cenário | Amostras originais | Após augmentation (n=5) |
|---|---|---|
| 3 vídeos, ~30 frames/vídeo (RF/KNN) | ~90 frames/gesto | ~540 frames/gesto |
| 3 vídeos, ~2 sequências/vídeo (LSTM) | ~6 sequências/gesto | ~36 sequências/gesto |

## Onde a augmentation é aplicada

A augmentation ocorre **somente nos dados de treino**, após a extração dos landmarks e antes do treinamento do modelo. Dados de teste e validação nunca passam por augmentation — eles representam condições reais de uso e não devem ser modificados.

```
Vídeos de treino
      ↓
Extração de frames (data_preprocessing.py)
      ↓
Detecção de landmarks + normalização (feature_extraction.py)
      ↓
Augmentation ← aqui (landmark_augmentation.py)
      ↓
Treinamento do modelo (model_training.py)
```

## Como ativar

Na opção `[1] Treinar modelo` do menu principal, o sistema pergunta:

```
Usar data augmentation nos landmarks? (s/n)
Quantas variacoes por amostra? [padrao: 5]
```

Também é possível ativar diretamente no código:

```python
features, labels = extract_features_from_directory(
    dataset_root_dir="dataset/frames_treino",
    mode="rf",
    augmentar=True,
    n_aumentos=5
)
```