# Sistema de Reconhecimento de Sinais em LIBRAS

Sistema de visão computacional e aprendizado de máquina para tradução automática da Língua Brasileira de Sinais (LIBRAS). O sistema detecta e reconhece gestos via webcam, utilizando pontos de referência anatômicos das duas mãos e do corpo (ombros, cotovelos, pulsos e quadris) para classificar o gesto com redes neurais (LSTM, BiLSTM e TCN, com ou sem atenção temporal) e modelos clássicos (Random Forest, XGBoost e K-Means + Random Forest).

---

## Artefatos entregues

Esta seção descreve todos os arquivos e diretórios do projeto, seu papel e, quando aplicável, os parâmetros de execução.

### Código-fonte

**`live_testing.py`** — Teste ao vivo com a webcam e um modelo treinado salvo em disco. Uso: `python live_testing.py <modelo .keras|.pkl>` (único argumento; LabelEncoder, câmera, Holistic e tamanho da janela vêm de `utils/constants.py`). Como os modelos não têm classe de repouso, a predição só ocorre sobre um gesto gravado: ESPAÇO inicia uma contagem de 3 s (em que o ruído do Pose parado calibra o limiar de movimento), a gravação começa no primeiro movimento e termina quando os braços e as mãos ficam parados por `GESTO_TEMPO_PARADO_S`. Um único Holistic (modo VIDEO) processa os frames da câmera, com a mesma extração e normalização do treino (142 features, janelas de `SEQUENCE_LENGTH`). Detalhes, parâmetros e limitações em `docs/live_testing.md`.

**`feature_extraction.py`** — Extrai os landmarks com o MediaPipe Holistic (`utils/holistic_landmarker.task`): as duas mãos completas e, do Pose, os ombros, cotovelos, pulsos e quadris (sem os pontos do rosto e sem os dedos estimados pelo Pose; a malha da face também não é usada). Varre o diretório dataset/frames em processos paralelos, frame a frame: cada frame é processado de forma independente por um Holistic novo no modo IMAGE (o Holistic guarda estado entre chamadas mesmo no modo IMAGE, o que alteraria a pose de um frame conforme os anteriores). Depois monta sequências de `SEQUENCE_LENGTH` frames, pulando de `DEFAULT_STEP` em `DEFAULT_STEP` frames, armazenadas em `dataset/(treino|teste).csv`. Os pontos são normalizados em relação ao corpo (`normalizar_holistic` em `utils/utils.py`): origem no centro dos ombros e unidade igual à largura dos ombros, em pixels (sem distorção pela proporção da imagem). As mãos têm x, y e z (z relativo ao próprio pulso) e a pose apenas x e y, pois o z do Pose estimado pelo MediaPipe não é confiável. Mão não detectada vira um bloco de zeros; frame sem pose detectada é preenchido com o frame anterior. Caso um vídeo não possua `SEQUENCE_LENGTH` frames, um padding é adicionado ao final 
(frames vazios com valores de landmarks zerados).

**`pipeline.py`** — Pipeline de treinamento e avaliação dos modelos. Lê os datasets de treino e teste, aplica o Data Augmentation uma única vez nos dados de treino e separa a validação (intérpretes de `INTERPRETES_VALIDACAO`, atualmente o 1 e o 6, sempre sem augmentation), usada no Early Stopping das redes (F1-Macro da validação, paciência `EARLY_STOPPING_PATIENCE` = 15 épocas, restaurando os pesos da melhor época). Os hiperparâmetros dos modelos da lista são escolhidos com **GridSearchCV** e validação cruzada Leave-One-Group-Out (um intérprete por fold). O modelo final é retreinado com todos os intérpretes (as redes por `EPOCAS_TREINO_FINAL` épocas fixas) e avaliado no conjunto de teste; as análises ficam em `outputs/` e os modelos em `models/`. Detalhes em `docs/pipeline.md`; o protocolo e os experimentos que o justificam estão em `docs/protocolo_validacao.md`.

**`extracao_frames.py`** — Realiza o processamento dos vídeos dos datasets, capturando seus frames, salvos no diretório
`dataset/frames/{treinamento|teste}/{gesto}/{id_video}/{id_frame}.jpg`.

**`landmark_augmentation.py`** — Aplica transformações nos vetores de landmarks sem reprocessar imagens: ruído gaussiano (σ=0,005), escala uniforme (±15%), rotação 2D (±15°), espelhamento horizontal e dropout de mão (a mão menos ativa na sequência inteira, ou cada mão em 15% dos frames). Para cada amostra são geradas `N_AUMENTOS` variações: o espelho e `N_AUMENTOS - 1` combinações sorteadas das demais transformações. Os parâmetros aleatórios são pré-gerados uma vez e aplicados igualmente a todos os frames, preservando a coerência temporal. O espelhamento também troca direita e esquerda (mãos e pontos do Pose). Suporta os datasets apenas das mãos (126 features) e os do Holistic (142 features); detalhes em `docs/augmentation.md`.

**`split_minds.py`** — Separa o dataset de treino (MINDS, 8 intérpretes) em treino e teste por intérprete (`INTERPRETES_TESTE_MINDS`, atualmente o 5 e o 8; podem ser passados na linha de comando, ex.: `python split_minds.py 2 3`), gerando `dataset/treino_minds.csv` e `dataset/teste_minds.csv` para avaliar o modelo apenas no MINDS, com intérpretes nunca vistos no treino.

**`preprocessing_tests.py`** — Experimento de pré-processamento dos frames: testa combinações de resolução, sharpening, CLAHE e tipo e tamanho de blur, medindo a taxa de frames com ao menos uma mão detectada pelo MediaPipe HandLandmarker. Os resultados vão para `logs/resultado_preprocessamento_(treino|teste)_blur_clahe.csv`.

**`utils/`** - Diretório com as constantes do projeto (`constants.py`, `constants_cv.py`), a normalização dos landmarks e o registro das métricas (`utils.py`), as arquiteturas das redes (`redes.py`), os pipelines do K-Means, do Random Forest e do XGBoost (`kmeans.py`, `xgboost.py`), a barra de progresso do joblib (`tqdm.py`), o script de análise das resoluções dos frames (`resolution_distribution.py`) e os modelos `.task` do MediaPipe.

---

### Dados

**`dataset/videos`** — Vídeos originais de treinamento e teste, não adicionados ao repositório devido ao tamanho dos
vidos (80MB cada, ~60GB o dataset como um todo)


**`dataset/frames`** — Frames extraídos dos vídeos, separados no subdiretório `treinamento e teste`. Cada vídeo gera uma subpasta numerada (`01`,`02`,`03`, ...) dentro da pasta do gesto correspondente, não adicionados ao repositório devido ao tamanho total de todos os frames (6 a 7 GB).


**`dataset/(treino|teste).csv`** — Dataset de features em 3D. Colunas: `target`(Nome do gesto), `video_id`(ID do vídeo, `{gesto}_{id_video}`; no `pipeline.py` é convertido no intérprete, `(id_video - 1) // 5 + 1`, usado como grupo no split de validação e no GridSearch), `sample_idx` (índice da sequência), `frame_idx` (posição dentro da sequência) + colunas de coordenadas + `duplicado` (Booleano se o frame foi duplicado ou não). São 142 colunas: 126 das mãos (`d_*` mão direita e `e_*` mão esquerda, x/y/z de 21 landmarks) + `p_x_<ponto>`/`p_y_<ponto>` dos 8 pontos do Pose (ombros, cotovelos, pulsos e quadris, ex.: `p_x_ombro_esquerdo`).

**`dataset/(treino|teste)_minds.csv`** — Mesmo formato, gerados pelo `split_minds.py`: o treino do MINDS dividido por intérprete (1761 e 636 sequências).

---

### Modelos treinados


**`models/<modelo>_<configuração>.(keras|pkl)`** — Melhor modelo treinado de cada configuração: as redes `lstm`, `bilstm` e `tcn` (Keras/TensorFlow, sufixo `_at` quando usam atenção temporal) em `.keras`, e `kmeans`, `rf` e `xgboost` (scikit-learn) em `.pkl`. O sufixo `_aug` indica Data Augmentation no treinamento. `final` identifica a avaliação no teste cross-dataset e `minds` a avaliação apenas no MINDS (`split_minds.py`).

**`models/label_encoder.pkl`** — `LabelEncoder` do scikit-learn que mapeia nomes de gestos para índices inteiros.

---

### Resultados e logs


**`logs/resultados_treino.csv`** — Histórico de todos os treinamentos realizados. Colunas: `data_hora`, `modelo`, `acuracia_%`, `f1_macro_%`, `amostras_treino`, `augmentation`, `n_aumentos`, `epoca_parada`, `tempo_min`, `hiperparametros`.

**`logs/(treino|teste).txt`** - Relatório da extração das landmarks nos frames, mostrando a quantidade de amostras
geradas, quantidade de frames duplicado, quantidade de vídeos com padding e quantidade de frames de padding no gesto, além da porcentagem de frames com pose e com cada mão detectada. 

**`outputs/`** — Diretório contendo os relatórios dos resultados do treinamento dos modelos, como matrizes
de confusão e results_{model} mostrando as métricas do modelo treinado e, para as redes neurais (LSTM, BiLSTM e TCN), os gráficos `acc_{model}.png` e `loss_{model}.png` do treino final por época (sem validação, pois o treino final usa todos os intérpretes).

**`best/<modelo>.txt`** — Métricas do melhor treino de cada modelo (`f1-macro` na primeira linha e `acuracia` na segunda, em %). Ao fim de cada treino, o `pipeline.py` só substitui o modelo em `models/` e os arquivos em `outputs/` se o novo treino tiver F1-Macro maior (acurácia como desempate).


---

### Arquivos auxiliares e documentação

**`docs/`** — Diretório contendo a documentação técnica complementar (`augmentation.md`, `live_testing.md`, `pipeline.md`, `protocolo_validacao.md`).

**`utils/holistic_landmarker.task`** — Modelo pré-treinado do MediaPipe Holistic (mãos e pose). Usado na extração de features e no teste ao vivo.

**`utils/hand_landmarker.task`** — Modelo pré-treinado do MediaPipe para landmarks das mãos (HandLandmarker). Usado apenas pelo `preprocessing_tests.py`.

**`requirements.txt`** — Lista de dependências Python. Instalar pacotes com `pip install -r requirements.txt`.

---

## Requisitos e instalação

Tensorflow não possui suporte para o python 3.13, apenas para python 3.9-3.12

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar Ambiente Virtual
Windows - `.\venv\Scripts\Activate.ps1`
Linux - `source venv/bin/activate`

# Instalar dependências
pip install -r requirements.txt
```

---

## Como o sistema funciona

```
Vídeos de exemplo
      ↓
Extração de frames  (extracao_frames.py)
      ↓
Detecção de landmarks via MediaPipe  (feature_extraction.py)
      ↓
Treinamento do modelo  (pipeline.py)
      ↓
Reconhecimento ao vivo pela webcam  (live_testing.py)
```

**Landmarks:** o MediaPipe Holistic detecta, por frame, as duas mãos (21 pontos cada, com x, y e z) e o Pose. Do Pose são usados ombros, cotovelos, pulsos e quadris, apenas com x e y → **142 features por frame**: mão direita (63) + mão esquerda (63) + pose (16). Mão não detectada → bloco de 63 zeros; frame sem pose → repete o último frame válido.

**Normalização:** feita em relação ao corpo (`normalizar_holistic`). Os pontos vão para pixels, a origem passa a ser o centro dos ombros e a unidade é a largura dos ombros, removendo posição, distância da câmera e tamanho da pessoa. O z das mãos é relativo ao próprio pulso, preservando a profundidade dos dedos; o z do Pose não é usado, pois o MediaPipe o estima de forma instável.

**Modelos:** todos recebem sequências de `SEQUENCE_LENGTH` (15) frames.

| Modelo | Entrada |
|---|---|
| LSTM, BiLSTM e TCN (com ou sem atenção) | Sequências contínuas `(N, 15, 142)` |
| Random Forest e XGBoost | Sequência achatada em um único vetor `(N, 2130)` (15 × 142) |
| K-Means + Random Forest | Frames agrupados em poses pelo K-Means; cada sequência vira um histograma de poses e de transições entre poses |
