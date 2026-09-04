# Sistema de Reconhecimento de Sinais em LIBRAS

Sistema de visão computacional e aprendizado de máquina para tradução automática da Língua Brasileira de Sinais (LIBRAS). O sistema detecta e reconhece gestos via webcam, utilizando pontos de referência anatômicos de ambas as mãos para classificação com três modelos distintos disponíveis: LSTM, K-Means temporal e um Ensemble de ambos.

---

## Artefatos entregues

Esta seção descreve todos os arquivos e diretórios do projeto, seu papel e, quando aplicável, os parâmetros de execução.

### Código-fonte

**`main.py`** — Ponto de entrada do sistema. Exibe um menu interativo numerado (0–7) que orquestra todo o fluxo de trabalho. Não recebe argumentos de linha de comando; todas as escolhas são feitas durante a execução. Para iniciar: `python main.py`.

**`feature_extraction.py`** — Extrai e normaliza os landmarks de ambas as mãos usando o MediaPipe Hand Landmarker. Varre recursivamente o diretório de frames e retorna features em formato 2D para RF/KNN ou 3D para LSTM.
- Parâmetros principais: `dataset_root_dir` (pasta raiz com subpastas por gesto), `mode` (`"rf"` ou `"lstm"`), `sequence_length` (tamanho da janela temporal, padrão 20 frames), `step` (passo da janela deslizante, padrão 1), `export_dataframe` (se `True`, salva CSV em `dataset/`).

**`model_training.py`** — Treina os três modelos. Cada função (`train_random_forest`, `train_lstm`, `train_knn`) realiza internamente o split treino/teste, aplica data augmentation apenas no conjunto de treino (evitando data leakage) e salva o modelo em `models/`.
- Parâmetros comuns: `features`, `labels`, `LSTM_PATH`, `augmentar` (bool), `n_aumentos` (int, variações sintéticas por amostra), `return_accuracy` (bool).
- `train_lstm` adiciona: `encoder_path` (caminho do `LabelEncoder`).
- `train_knn` adiciona: `n_clusters` (centróides K-Means, padrão 10).

**`sign_recognition.py`** — Motor de inferência. Abre um vídeo ou webcam, processa os frames com MediaPipe e retorna o gesto com maior frequência de predições (voto majoritário). Para modelos sequenciais (LSTM, KNN+K-Means), amostra 1 frame a cada 5 do vídeo original — mesmo intervalo do treinamento — garantindo correspondência temporal.
- Parâmetros: `video_path` (caminho do arquivo ou `0` para webcam), `tipo_modelo` (`"1"` RF / `"2"` LSTM / `"3"` KNN).

**`extracao_frames.py`** — Realiza o processamento dos vídeos dos datasets, capturando seus frames, salvos no diretório
`dataset/frames/{treinamento|teste}/{gesto}/{id_video}/{id_frame}.jpg`.

**`landmark_augmentation.py`** — Aplica transformações geométricas nos vetores de landmarks sem reprocessar imagens: ruído gaussiano (σ=0,005), escala uniforme (±15%), rotação 2D (±15°) e espelhamento horizontal. Para sequências LSTM, os parâmetros aleatórios são pré-gerados uma vez e aplicados igualmente a todos os frames, preservando a coerência temporal. Para gestos de duas mãos, o espelhamento também troca direita e esquerda.

**`import_from_csv.py`** — Carrega o dataset de um arquivo CSV e reconstrói o formato correto para cada modelo: `(N, 126)` para RF/KNN ou `(N, T, 126)` para LSTM.
- Parâmetros: `filepath` (caminho do CSV), `mode` (`"rf"` ou `"lstm"`).

---

### Dados

**`videos/treino/<gesto>/`** — Vídeos originais de treinamento, organizados em subpastas por gesto (ex: `videos/treino/abelha/`, `videos/treino/abraco/`). Cada subpasta corresponde a uma classe reconhecida pelo sistema.

**`videos/teste/<gesto>/`** — Vídeos originais para avaliação final, com mesma organização.

**`dataset/frames_treino/<gesto>/v000N/`** — Frames extraídos dos vídeos de treino. Cada vídeo gera uma subpasta numerada (`v0000/`, `v0001/`, ...) dentro da pasta do gesto correspondente.

**`dataset/frames_teste/<gesto>/v000N/`** — Frames extraídos dos vídeos de teste.

**`dataset/dataset_completo_rf.csv`** — Dataset de features para Random Forest e KNN. Colunas: `target` (nome do gesto) + 126 colunas de coordenadas normalizadas (`d_x_1`…`d_z_21` para mão direita, `e_x_1`…`e_z_21` para mão esquerda). Gerado pela opção `[7]` com modo RF.

**`dataset/dataset_completo_lstm.csv`** — Dataset de features para LSTM. Colunas: `target`, `sample_idx` (índice da sequência), `frame_idx` (posição dentro da sequência) + 126 colunas de coordenadas. Cada amostra ocupa `sequence_length` linhas. Gerado pela opção `[7]` com modo LSTM.

---

### Modelos treinados

**`models/sign_model.pkl`** — Modelo Random Forest serializado. Configuração: 200 estimadores, `class_weight='balanced'`, `random_state=42`.

**`models/lstm_sign_model.h5`** — Rede LSTM (Keras/TensorFlow). Arquitetura: `LSTM(64, recurrent_dropout=0.1)` → `Dropout(0.3)` → `LSTM(128, recurrent_dropout=0.1)` → `Dropout(0.3)` → `Dense(64, relu)` → `Dense(n_classes, softmax)`. Treinada com `class_weight` balanceado e `EarlyStopping(patience=10)`.

**`models/knn_sign_model.pkl`** — Pacote KNN serializado contendo: modelo `KNeighborsClassifier(weights='distance')`, `StandardScaler`, número de centróides K-Means e metadados de configuração (tamanho da janela temporal).

**`models/label_encoder.pkl`** — `LabelEncoder` do scikit-learn que mapeia nomes de gestos para índices inteiros. Usado exclusivamente pelo modelo LSTM.

---

### Resultados e logs

**`outputs/predicoes_modelo_1.txt`** — Predições do Random Forest para os vídeos de teste, uma por linha, na ordem de processamento. Gerado pela opção `[3]`.

**`outputs/predicoes_modelo_2.txt`** — Predições do LSTM para os vídeos de teste.

**`outputs/predicoes_modelo_3.txt`** — Predições do KNN para os vídeos de teste.

**`outputs/mediapipe_mundo_real.txt`** — Relatório diagnóstico gerado pelo teste do MediaPipe.

**`logs/resultados_treino.csv`** — Histórico de todos os treinamentos realizados. Colunas: `data_hora`, `modelo`, `acuracia_%`, `amostras_treino`, `augmentation`, `n_aumentos`. Atualizado automaticamente após cada treino (opção `[1]`).

**`logs/matriz_confusao_modelo_1.png`** — Figura da matriz de confusão do Random Forest. Salva automaticamente ao usar a opção `[6]`.

**`logs/matriz_confusao_modelo_2.png`** — Figura da matriz de confusão do LSTM.

**`logs/matriz_confusao_modelo_3.png`** — Figura da matriz de confusão do KNN.

---

### Arquivos auxiliares e documentação

**`docs/`** — Diretório contendo a documentação técnica complementar (`augmentation.md`, `CORREÇÕES_ACURÁCIA.md`, `RESUMO_CORRECOES.md`) e artigos de referência (`4574 (1).pdf`).

**`outputs/`** — Diretório contendo os relatórios e predições em formato `.txt` (`gabarito.txt`, `predicoes_modelo_X.txt`, `mediapipe_mundo_real.txt`).

**`hand_landmarker.task`** — Modelo pré-treinado do MediaPipe para detecção de landmarks de mãos. Deve estar na raiz do projeto. Necessário para qualquer etapa que use o MediaPipe (extração de features e inferência).

**`outputs/gabarito.txt`** — Rótulos reais dos vídeos de teste, usados como referência para a matriz de confusão.

**`requirements.txt`** — Lista de dependências Python. Instalar com `pip install -r requirements.txt`.

---

## Requisitos e instalação

O TensorFlow exige **Python 3.9, 3.10, 3.11 ou 3.12** (64-bit). Python 3.13+ não é suportado.

```bash
# Criar ambiente virtual
py -3.12 -m venv venv

# Ativar (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt
```

---

## Como o sistema funciona

```
Vídeos de exemplo
      ↓
Extração de frames  (data_preprocessing.py)
      ↓
Detecção de landmarks via MediaPipe  (feature_extraction.py)
      ↓
Treinamento do modelo  (model_training.py)
      ↓
Reconhecimento em novos vídeos  (sign_recognition.py)
```

**Landmarks:** MediaPipe detecta até 2 mãos por frame, retornando 21 pontos anatômicos por mão → **126 coordenadas por frame**. Mão não detectada → bloco de 63 zeros.

**Normalização:** X e Y divididos pela largura da palma (lm5→lm17, eixo XY), removendo variação de câmera. Z apenas centralizado no pulso, preservando profundidade relativa entre dedos.

**Modelos:**

| Modelo | Entrada | Melhor para |
|---|---|---|
| Random Forest | Frame a frame `(N, 126)` | Gestos estáticos (forma da mão) |
| LSTM | Sequências `(N, 20, 126)` | Gestos dinâmicos (movimento e trajetória) |
| KNN + K-Means | Sequências compactadas | Alternativa leve ao LSTM |

---

## Manual do usuário

```bash
python main.py
```

### `[1]` Treinar modelo
1. Escolha o modelo: `1` RF / `2` LSTM / `3` KNN
2. Origem dos dados: `1` Extrair agora (lento) ou `2` Carregar CSV **(recomendado)**
3. Data augmentation `s/n`: gera variações sintéticas aplicadas somente no treino
4. Modelo salvo em `models/`; acurácia registrada em `logs/resultados_treino.csv`

### `[2]` Testar via Webcam
Reconhecimento em tempo real. Escolha o modelo e realize o gesto. Tecla `Q` para sair.

### `[3]` Testar via vídeo (pasta de teste)
Processa todos os vídeos de uma pasta. Predições salvas em `predicoes_modelo_X.txt`.
- Escolha o modelo e informe a pasta (padrão: `videos/teste`)

### `[4]` Comparar pipeline direto vs CSV
Treina o mesmo modelo duas vezes (extração direta e CSV) e compara acurácias para validar consistência do dataset.

### `[5]` Extrair frames de vídeos em lote
Converte todos os vídeos de `videos/treino/` e `videos/teste/` em frames JPEG.
> Execute sempre que adicionar novos vídeos, antes de `[7]` e `[1]`.

### `[6]` Gerar Matriz de Confusão
Exibe e **salva** a matriz em `logs/matriz_confusao_modelo_X.png`.
- Pré-requisito: executar `[3]` antes
- Informe os rótulos reais separados por vírgula (ex: `abelha, abraco, acabar`)

### `[7]` Gerar Dataset (salvar CSV)
Extrai landmarks de `dataset/frames_treino/` e salva em CSV. **Obrigatório antes de `[1] → Carregar CSV`.**
- Escolha o modo: `1` para RF/KNN ou `2` para LSTM

### `[0]` Sair

---

## Fluxo de trabalho

**Primeira vez:**
```
[5] Extrair frames  →  [7] Gerar Dataset (RF e LSTM)  →  [1] Treinar  →  [3] Testar  →  [6] Matriz de Confusão
```

**Ao adicionar novos vídeos:**
```
[5] Extrair frames  →  [7] Gerar Dataset (sobrescrever: s)  →  [1] Treinar novamente
```

---

## Observações

- **Aviso `Failed to send to clearcut`**: telemetria interna do MediaPipe, inofensivo.
- **Modelos incompatíveis**: ao mudar o número de gestos ou regenerar o dataset, todos os modelos precisam ser retreinados.

