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

### Formato da extração com o Holistic

Na extração com o MediaPipe Holistic (`feature_extraction.py`), cada frame tem **142 valores**: mão direita (63) + mão esquerda (63) + pose (ombros, cotovelos, pulsos e quadris: 8 pontos, apenas x e y = 16). Todos os pontos são relativos ao corpo: a origem é o centro dos ombros e a unidade é a largura dos ombros (o z das mãos é relativo ao próprio pulso). Por isso:

- a origem das transformações de escala e rotação passa a ser o centro dos ombros (e não o pulso), e as mãos **não** são recentralizadas no pulso depois das transformações, pois isso apagaria a posição delas em relação ao corpo;
- no espelhamento, além da troca entre mão direita e esquerda, cada ponto do Pose troca de lugar com o seu par do outro lado do corpo (ombros, cotovelos, pulsos e quadris).

O módulo identifica o formato pelo número de features do frame (126 ou 142).

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


### 5. Dropout de mão (sequência)

Zera, em todos os frames, a mão **menos ativa** da sequência (a de menor variação das coordenadas nos frames em que aparece). Sequências com apenas uma mão não são alteradas. No dataset de treino do Holistic, a mão removida é a esquerda (não dominante) em ~90% das sequências com as duas mãos.

**O que simula:** a mão não dominante ausente ou fora do quadro. Ensina o modelo a não depender dela nos sinais feitos com uma mão — a principal diferença entre o treino (MINDS, braço solto ao lado do corpo) e o teste (mãos juntas na frente do corpo).


### 6. Dropout de mão (frame)

Zera cada mão, de forma independente, em frames aleatórios, com probabilidade `AUG_DROPOUT_FRAME_PROB` (15%) por mão e por frame.

**O que simula:** as falhas de detecção das mãos pelo Holistic, que não encontra a mão em ~30–40% dos frames.

Nos dois dropouts apenas o bloco da mão é zerado e os pontos do Pose são mantidos, como nos dados reais: quando o Holistic perde a mão, o corpo continua detectado (pose em 100% dos frames). Os dropouts são aplicados depois das transformações geométricas.


## Como as transformações são combinadas

Para cada amostra original, o módulo gera `n_aumentos` variações:

- **1ª variação:** sempre o espelhamento (a mais valiosa)
- **Demais variações:** combinações aleatórias de ruído, escala, rotação e dropouts de mão — o número e a escolha das transformações muda a cada geração

Isso garante diversidade entre as amostras sintéticas, evitando que sejam cópias umas das outras.

## Aplicação das Transformações no Dataset

Cada amostra é uma **sequência de frames consecutivos**. Ao aplicar augmentation em uma sequência, a **mesma transformação é usada em todos os frames** da sequência.


## Impacto no tamanho do dataset

Com `n_aumentos=10` (`N_AUMENTOS`), cada amostra original gera 10 variações (1 espelhamento + 9 combinações), totalizando 11× mais dados; cada transformação aparece em média em ~60% das combinações.

| Cenário | Amostras originais | Após augmentation (n=10) |
|---|---|---|
| Treino Holistic sem a validação (7 intérpretes) | ~1990 sequências | ~21900 sequências |

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
## Transformações planejadas

Levantadas após a análise do dataset do Holistic, em que a principal diferença entre treino (MINDS) e teste é a mão
não dominante em repouso: no teste ela aparece em mais frames e mais alta (mãos juntas na frente do corpo), enquanto
no MINDS o braço fica solto ao lado do corpo.

| Transformação | O que faz | Situação |
|---|---|---|
| Dropout de mão (sequência) | Zera a mão menos ativa em toda a sequência (o Pose é mantido) | **Implementada** (transformação 5) |
| Dropout de mão (frame) | Zera cada mão em 15% dos frames | **Implementada** (transformação 6) |
| Proporções do corpo | Escala x e y de forma independente (±10%) e altera o comprimento dos braços | A discutir |
| Mais janelas reais | Reduz o `DEFAULT_STEP` (5 → 2 ou 3) recortando de novo as séries dos vídeos a partir do CSV | A discutir — gera mais amostras, mas sobre os mesmos frames (mais redundância, não mais informação) |
| Mão sintética em repouso | Insere uma mão estática na frente do abdômen | Descartada — dado sintético sem base suficiente |
| Deslocamento por mão | Desloca cada mão/braço separadamente (±0,1–0,2 largura de ombro) | Descartada — o treino já tem a variação de 8 intérpretes |
| Temporal (velocidade, frames repetidos/pulados) | Reamostra a sequência entre 0,8× e 1,2× | Descartada — os vídeos de treino já têm velocidades diferentes |
| Rotação vertical (câmera mais acima/abaixo) | Gira o frame em torno do eixo x (±15°) | Descartada — depende do z do Pose, que piorou o modelo (benchmark: F1 66,1 → 59,9 ao adicioná-lo) |
| Estratégia balanceada | Espelho + cada transformação aplicada isoladamente na original e no espelho (14×) | Descartada — F1 56,1 contra 59,9 da combinação aleatória, com mais variação entre sementes |

Com os dropouts implementados, `N_AUMENTOS` passou de 5 para 10.
