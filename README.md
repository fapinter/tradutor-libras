# Sistema de Reconhecimento de Sinais em LIBRAS

Sistema de visão computacional e aprendizado de máquina para tradução automática da Língua Brasileira de Sinais (LIBRAS). O sistema detecta e reconhece gestos via webcam, utilizando pontos de referência anatômicos de ambas as mãos para classificação com três modelos distintos disponíveis: LSTM, K-Means temporal e um Ensemble de ambos.

---

## Artefatos entregues

Esta seção descreve todos os arquivos e diretórios do projeto, seu papel e, quando aplicável, os parâmetros de execução.

### Código-fonte

**`live_testing.py`** — Testes na vida real utilizando a Webcam e um modelo treinado salvo em disco.

**`feature_extraction.py`** — Extrai e normaliza os landmarks de ambas as mãos usando o MediaPipe Hand Landmarker. Varre o diretório dataset/frames e coleta as landmarks, montando N sequências de 20 frames, pulando de 5 em 5 frames
até não ser mais possível, depois os dados são armazenados em um CSV. Caso as mãos não sejam detectadas em algum frame, 
o frame anterior eh duplicado para evitar 
pontos vazios em meio a uma sequência. Caso um vídeo não possua 20 frames, um padding é adicionado ao final 
(frames vazios com valores de landmarks zerados).

**`pipeline.py`** — Pipeline de treinamento dos modelos, coleta os dados de treino e teste, separa o treino entre
treino e validação, aplica Data Augmentation nos dados de treino, depois treina os modelos inseridos na lista usando
**GridSearchCV**, para garantir os melhores hiperparâmetros para os dados utilizados, por fim os modelos são avaliados
no conjunto de teste, análises do seu desempenho são armazenados em `outputs/` e os modelos são armazenados em disco
para uso posterior.

**`extracao_frames.py`** — Realiza o processamento dos vídeos dos datasets, capturando seus frames, salvos no diretório
`dataset/frames/{treinamento|teste}/{gesto}/{id_video}/{id_frame}.jpg`.

**`landmark_augmentation.py`** — Aplica transformações geométricas nos vetores de landmarks sem reprocessar imagens: ruído gaussiano (σ=0,005), escala uniforme (±15%), rotação 2D (±15°) e espelhamento horizontal. Os parâmetros aleatórios são pré-gerados uma vez e aplicados igualmente a todos os frames, preservando a coerência temporal. Para gestos de duas mãos, o espelhamento também troca direita e esquerda.

**`utils/`** - Diretório contendo constantes do projeto, e funções auxiliares para o pipeline.

---

### Dados

**`dataset/videos`** — Vídeos originais de treinamento e teste, não adicionados ao repositório devido ao tamanho dos
vidos (80MB cada, ~60GB o dataset como um todo)


**`dataset/frames`** — Frames extraídos dos vídeos, separados no subdiretório `treinamento e teste`. Cada vídeo gera uma subpasta numerada (`01`,`02`,`03`, ...) dentro da pasta do gesto correspondente, não adicionados ao repositório devido ao tamanho total de todos os frames (6 a 7 GB).


**`dataset/(treinamento|teste).csv`** — Dataset de features em 3D. Colunas: `target`(Nome do gesto), `video_id`(ID do vídeo, usado para definir grupos), `sample_idx` (índice da sequência), `frame_idx` (posição dentro da sequência) + 126 colunas de coordenadas + `duplicado` (Booleano se o frame foi duplicado ou não).

---

### Modelos treinados


**`models/lstm_sign_model.keras`** — Rede LSTM (Keras/TensorFlow).

**`models/lstm_sign_model_aug.keras`** — Rede LSTM (Keras/TensorFlow) com Data Augmentation no treinamento.

**`models/label_encoder.pkl`** — `LabelEncoder` do scikit-learn que mapeia nomes de gestos para índices inteiros.

---

### Resultados e logs


**`logs/resultados_treino.csv`** — Histórico de todos os treinamentos realizados. Colunas: `data_hora`, `modelo`, `acuracia`, `f1_macro` `amostras_treino`, `augmentation`, `n_aumentos`.

**`logs/(treino|teste).txt`** - Relatório da extração das landmarks nos frames, mostrando a quantidade de amostras
geradas, quantidade de frames duplicado, quantidade de vídeos com padding e quantidade de frames de padding no gesto. 

**`outputs/`** — Diretório contendo os relatórios dos resultados do treinamento dos modelos, como matrizes
de confusão, predicoes_{modelo} e results_{model} mostrando as métricas do modelo treinado.


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
|---|---|---|
| LSTM | Sequências `(N, 20, 126)` |
| KNN + K-Means | Sequências compactadas |