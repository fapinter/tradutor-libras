# Sistema de Reconhecimento de Sinais em LIBRAS

Sistema de visão computacional e aprendizado de máquina para tradução automática da Língua Brasileira de Sinais (LIBRAS). O sistema detecta e reconhece gestos via webcam, utilizando pontos de referência anatômicos de ambas as mãos para classificação com três modelos distintos disponíveis: LSTM, K-Means temporal e um Ensemble de ambos.

---

## Artefatos entregues

Esta seção descreve todos os arquivos e diretórios do projeto, seu papel e, quando aplicável, os parâmetros de execução.

### Código-fonte

**`live_testing.py`** — Testes na vida real utilizando a Webcam e um modelo treinado salvo em disco.

**`feature_extraction.py`** — Extrai os landmarks com o MediaPipe Holistic (`utils/holistic_landmarker.task`): as duas mãos completas e, do Pose, os ombros, cotovelos, pulsos e quadris (sem os pontos do rosto e sem os dedos estimados pelo Pose; a malha da face também não é usada). Varre o diretório dataset/frames em processos paralelos, frame a frame: cada frame é processado de forma independente por um Holistic novo no modo IMAGE (o Holistic guarda estado entre chamadas mesmo no modo IMAGE, o que alteraria a pose de um frame conforme os anteriores). Depois monta sequências de `SEQUENCE_LENGTH` frames, pulando de `DEFAULT_STEP` em `DEFAULT_STEP` frames, armazenadas em `dataset/(treino|teste)_holistic.csv`. Os pontos são normalizados em relação ao corpo (`normalizar_holistic` em `utils/utils.py`): origem no centro dos ombros e unidade igual à largura dos ombros, em pixels (sem distorção pela proporção da imagem). As mãos têm x, y e z (z relativo ao próprio pulso) e a pose apenas x e y, pois o z do Pose estimado pelo MediaPipe não é confiável. Mão não detectada vira um bloco de zeros; frame sem pose detectada é preenchido com o frame anterior. Caso um vídeo não possua `SEQUENCE_LENGTH` frames, um padding é adicionado ao final 
(frames vazios com valores de landmarks zerados).

**`test_crop.py`** e **`tests_extraction.py`** — Extrações experimentais anteriores, apenas das mãos (126 features, normalizadas pelo pulso): com recortes das mãos localizadas pelo MediaPipe Pose e com o HandLandmarker na imagem inteira, respectivamente.

**`pipeline.py`** — Pipeline de treinamento dos modelos, coleta os dados de treino e teste, separa o treino entre
treino e validação (a validação, usada no Early Stopping das redes e sem augmentation, recebe os intérpretes
de `INTERPRETES_VALIDACAO`, atualmente o 5 e o 8), aplica Data Augmentation nos dados de treino, depois treina os modelos inseridos na lista usando
**GridSearchCV**, para garantir os melhores hiperparâmetros para os dados utilizados, por fim os modelos são avaliados
no conjunto de teste, análises do seu desempenho são armazenados em `outputs/` e os modelos são armazenados em disco
para uso posterior.

**`extracao_frames.py`** — Realiza o processamento dos vídeos dos datasets, capturando seus frames, salvos no diretório
`dataset/frames/{treinamento|teste}/{gesto}/{id_video}/{id_frame}.jpg`.

**`landmark_augmentation.py`** — Aplica transformações nos vetores de landmarks sem reprocessar imagens: ruído gaussiano (σ=0,005), escala uniforme (±15%), rotação 2D (±15°), espelhamento horizontal e dropout de mão (a mão menos ativa na sequência inteira, ou cada mão em 15% dos frames). Para cada amostra são geradas `N_AUMENTOS` variações: o espelho e `N_AUMENTOS - 1` combinações sorteadas das demais transformações. Os parâmetros aleatórios são pré-gerados uma vez e aplicados igualmente a todos os frames, preservando a coerência temporal. O espelhamento também troca direita e esquerda (mãos e pontos do Pose). Suporta os datasets apenas das mãos (126 features) e os do Holistic (142 features); detalhes em `docs/augmentation.md`.

**`utils/`** - Diretório contendo constantes do projeto, e funções auxiliares para o pipeline.

---

### Dados

**`dataset/videos`** — Vídeos originais de treinamento e teste, não adicionados ao repositório devido ao tamanho dos
vidos (80MB cada, ~60GB o dataset como um todo)


**`dataset/frames`** — Frames extraídos dos vídeos, separados no subdiretório `treinamento e teste`. Cada vídeo gera uma subpasta numerada (`01`,`02`,`03`, ...) dentro da pasta do gesto correspondente, não adicionados ao repositório devido ao tamanho total de todos os frames (6 a 7 GB).


**`dataset/(treinamento|teste).csv`** — Dataset de features em 3D. Colunas: `target`(Nome do gesto), `video_id`(ID do vídeo, `{gesto}_{id_video}`; no `pipeline.py` é convertido no intérprete, `(id_video - 1) // 5 + 1`, usado como grupo no split de validação e no GridSearch), `sample_idx` (índice da sequência), `frame_idx` (posição dentro da sequência) + colunas de coordenadas + `duplicado` (Booleano se o frame foi duplicado ou não). Nos datasets apenas das mãos são 126 colunas (`d_*` mão direita e `e_*` mão esquerda, x/y/z de 21 landmarks); nos `*_holistic.csv` são 142 colunas: as mesmas 126 das mãos + `p_x_<ponto>`/`p_y_<ponto>` dos 8 pontos do Pose (ombros, cotovelos, pulsos e quadris, ex.: `p_x_ombro_esquerdo`).

---

### Modelos treinados


**`models/<modelo>_<configuração>.(keras|pkl)`** — Melhor modelo treinado de cada configuração: redes LSTM/TCN (Keras/TensorFlow) em `.keras` e K-Means, Random Forest e XGBoost (scikit-learn) em `.pkl`. O sufixo `aug` indica Data Augmentation no treinamento.

**`models/label_encoder.pkl`** — `LabelEncoder` do scikit-learn que mapeia nomes de gestos para índices inteiros.

---

### Resultados e logs


**`logs/resultados_treino.csv`** — Histórico de todos os treinamentos realizados. Colunas: `data_hora`, `modelo`, `acuracia`, `f1_macro` `amostras_treino`, `augmentation`, `n_aumentos`.

**`logs/(treino|teste).txt`** - Relatório da extração das landmarks nos frames, mostrando a quantidade de amostras
geradas, quantidade de frames duplicado, quantidade de vídeos com padding e quantidade de frames de padding no gesto. 

**`outputs/`** — Diretório contendo os relatórios dos resultados do treinamento dos modelos, como matrizes
de confusão e results_{model} mostrando as métricas do modelo treinado. Também contém `predicoes_{model}.csv`,
com a predição de cada amostra do teste (colunas `id` = `{nome_gesto}_{id_video}_{sample_idx}`, `predicao` e `diagnostico` = `certo|errado`),
e, para as redes neurais (LSTM e TCN), os gráficos `acc_{model}.png` e `loss_{model}.png` de treino vs validação por época.

**`best/<modelo>.txt`** — Métricas do melhor treino de cada modelo (`f1-macro` na primeira linha e `acuracia` na segunda, em %). Ao fim de cada treino, o `pipeline.py` só substitui o modelo em `models/` e os arquivos em `outputs/` se o novo treino tiver F1-Macro maior (acurácia como desempate).


---

### Arquivos auxiliares e documentação

**`docs/`** — Diretório contendo a documentação técnica complementar (`augmentation.md`, `pipeline.md`).

**`hand_landmarker.task`** — Modelo pré-treinado do MediaPipe para detecção de landmarks de mãos. Deve estar na raiz do projeto. Necessário para qualquer etapa que use o MediaPipe (extração de features e inferência).

**`requirements.txt`** — Lista de dependências Python. Instalar pacotes com `pip install -r requirements.txt`.

---

## Requisitos e instalação

Tensorflow não possui suporte para o python 3.13, apenas para python 3.9-3.12

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar Ambiente Virtual
Windows - `.\venv\Scripts\Activate.ps1`
Linux - `venv/bin/activate`

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
Reconhecimento em novos vídeos  (live_testing.py)
```

**Landmarks:** MediaPipe detecta até 2 mãos por frame, retornando 21 pontos anatômicos por mão → **126 coordenadas por frame**. Mão não detectada → bloco de 63 zeros.

**Normalização:** X e Y divididos pela largura da palma (lm5→lm17, eixo XY), removendo variação de câmera. Z apenas centralizado no pulso, preservando profundidade relativa entre dedos.

**Modelos:**

| Modelo | Entrada |
|---|---|
| LSTM | Sequências Contínuas `(N, 20/15, 126)` |
| TCN | Sequências Contínuas `(N, 20/15, 126)` |
| Random Forest K-Means | Sequências representadas em um único vetor (2520) |
| XGBoosting | Sequências representadas em um único vetor (2520) |
