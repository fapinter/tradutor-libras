# Constantes para landmarks

NUM_LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
NUM_HANDS = 2
N_MAO = 63
# Features apenas das mãos (extração antiga, HandLandmarker): [mão direita] + [mão esquerda]
NUM_FEATURES_MAOS = 126

# Landmarks do Pose usados na extração com o Holistic (índice do MediaPipe -> nome da coluna no CSV):
# braços e tronco, sem os pontos do rosto (0-10) e sem os dedos estimados pelo Pose (17-22),
# já cobertos pelas mãos completas. Esquerda e direita são da pessoa.
POSE_LANDMARKS_USADOS = {
    11: "ombro_esquerdo", 12: "ombro_direito",
    13: "cotovelo_esquerdo", 14: "cotovelo_direito",
    15: "pulso_esquerdo", 16: "pulso_direito",
    23: "quadril_esquerdo", 24: "quadril_direito",
}
# Do Pose são usados apenas x e y: o z estimado pelo MediaPipe Pose é exagerado e instável
COORDINATES_PER_POSE_LANDMARK = 2
N_POSE = len(POSE_LANDMARKS_USADOS) * COORDINATES_PER_POSE_LANDMARK
# Features da extração com o Holistic: [mão direita (63)] + [mão esquerda (63)] + [pose (16)]
NUM_FEATURES = NUM_FEATURES_MAOS + N_POSE

# Índices do Pose (MediaPipe) usados na normalização relativa ao corpo
POSE_OMBRO_ESQUERDO = 11
POSE_OMBRO_DIREITO = 12
# Pares esquerda/direita do Pose, trocados no espelhamento do Data Augmentation
POSE_PARES_ESPELHO = ((11, 12), (13, 14), (15, 16), (23, 24))

# Vetor de mão ausente
ZEROS_MAO = [0.0] * N_MAO
# Vetores de features ausentes (frame sem detecção ou padding)
ZEROS_FRAME_MAOS = [0.0] * NUM_FEATURES_MAOS
ZEROS_FRAME = [0.0] * NUM_FEATURES


HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
                    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14),
                    (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]

# Constantes de extração de features

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
FRAME_RATE = 5
SEQUENCE_LENGTH = 15
DEFAULT_STEP = 5

LANDMARKER_PATH = "utils/hand_landmarker.task"
HOLISTIC_LANDMARKER_PATH = "utils/holistic_landmarker.task"

# Constantes do recorte das mãos a partir do MediaPipe Pose (test_crop.py)
POSE_LANDMARKER_PATH = "utils/pose_landmarker_full.task"
# Lado do recorte quadrado de cada mão, em larguras de ombro
CROP_ESCALA_OMBROS = 2.0
# Distância máxima entre o pulso detectado e o pulso do Pose, em larguras de ombro
DIST_MAX_PULSO_POSE = 0.6
# Distância média entre landmarks abaixo da qual duas detecções são a mesma mão, em larguras de ombro
DIST_MESMA_MAO_CROP = 0.1

# Parâmetros de Data Augmentation
AUG_NOISE_INTENSITY = 0.005

# escala aleatória aplicada ao pulso
AUG_SCALE_MIN = 0.85
AUG_SCALE_MAX = 1.15

# Rotação máxima
AUG_ROTATION_MAX_DEGREES = 15.0

# Probabilidade de cada mão ser removida em cada frame no dropout de mão por frame
# (simula as falhas de detecção das mãos pelo Holistic)
AUG_DROPOUT_FRAME_PROB = 0.15

# Número de amostras geradas pelo Data Augmentation
N_AUMENTOS = 10

# Configurações dos Treinamentos
SEED = 42
LSTM_EPOCHS = 100

# Paciencia Early Stopping (avaliar qual ficará)
LSTM_PATIENCE = 10
PIPELINE_PATIENCE = 8

K_FOLDS = 5

# Os vídeos de treino (MINDS) estão em blocos de 5 repetições por intérprete:
# 01-05 = intérprete 1, 06-10 = intérprete 2, ..., 36-40 = intérprete 8
VIDEOS_POR_INTERPRETE = 5
# Intérpretes separados para a validação (Early Stopping), sem augmentation.
# O 8 tem as duas mãos visíveis na maior parte dos frames e o 5 deixa a mão de apoio solta,
# como a maioria; o 7, o outro com as duas mãos visíveis, fica no treino.
INTERPRETES_VALIDACAO = (5, 8)
EPOCHS_POR_FOLD = 50
USAR_AUGMENTATION = True

# Hiperparametros Grid Search LSTM
PARAM_GRID_LSTM = {
    "model__lstm_units_1": [64, 128],
    "model__lstm_units_2": [32, 64],
    "model__dense_units": [32, 64],
    "model__dropout_rate": [0.2, 0.4],
    "model__learning_rate": [0.001, 0.0005],
    "batch_size": [16, 32],
    "epochs": [100]
}

# Hiperparametros Grid Search K-Means + Random Forest
PARAM_GRID_KMEANS = {
    "compactar__n_clusters": [30, 50, 70],
    "rf__n_estimators": [200, 300, 400],
    "rf__max_depth": [20, 30, None],
}

PARAM_GRID_RF = {
    "rf__n_estimators": [300, 400, 500, 600],
    "rf__max_depth": [20, 30, 50, None],
}


PARAM_GRID_TCN = {
    "model__filters": [32, 64],
    "model__dense_units": [32, 64],
    "model__dropout_rate": [0.2, 0.4],
    "model__learning_rate": [0.001, 0.0005],
    "batch_size": [16, 32],
    "epochs": [100],
}

PARAM_GRID_XGBOOST = {
    "xgb__n_estimators": [400, 600],#200, 400],
    "xgb__max_depth": [6, 10, 15],#3, 6],
    "xgb__learning_rate": [0.05, 0.1],#, 0.1],
    "xgb__subsample": [0.6, 0.8]#0.6, 0.8],
}

# Captura de Vídeo em Tempo Real

FONTE = 0
FRAME_SKIP = 20
MIN_CONFIANCA = 0.4
ALTA_CONFIANCA = 0.6

CONFIDENCE_THRESHOLD = 0.5

# Diretórios
VIDEOS_DIR = "dataset/videos"
FRAMES_DIR = "dataset/frames"

FRAMES_TREINO_DIR = "dataset/frames/treinamento"
FRAMES_TESTE_DIR = "dataset/frames/teste"


DATASET_TREINO_CSV = "dataset/treino.csv"
DATASET_TESTE_CSV = "dataset/teste.csv"

DATASET_TREINO_MINDS_CSV = "dataset/treino_minds_70.csv"
DATASET_TESTE_MINDS_CSV = "dataset/teste_minds_30.csv"

DATASET_TREINO_15_CSV = "dataset/treino_15.csv"
DATASET_TESTE_15_CSV = "dataset/teste_15.csv"

DATASET_TREINO_MINDS_CROP_CSV = "dataset/treino_minds_crop_mediapipe_separado.csv"
DATASET_TESTE_MINDS_CROP_CSV = "dataset/teste_minds_crop_mediapipe_separado.csv"

DATASET_TREINO_HOLISTIC_CSV = "dataset/treino_holistic.csv"
DATASET_TESTE_HOLISTIC_CSV = "dataset/teste_holistic.csv"

LOGS_DIR = "logs"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
BEST_DIR = "best"

RESULTADOS_TREINO_CSV = "logs/resultados_treino.csv"
GRID_SEARCH_RESULTS_CSV = "outputs/grid_search_results.csv"
RELATORIO_AVALIACAO_TESTE = "outputs/relatorio_avaliacao_teste.txt"

MATRIZ_PATH = "outputs/matriz_%s.png"
RESULTS_PATH = "outputs/results_%s.txt"
ACC_PATH = "outputs/acc_%s.png"
LOSS_PATH = "outputs/loss_%s.png"
# Predição de cada amostra do teste (colunas id, predicao, diagnostico)
PREDICOES_PATH = "outputs/predicoes_%s.csv"

# Métricas do melhor treino de cada modelo (f1-macro e acurácia, em %)
BEST_PATH = "best/%s.txt"

KERAS_PATH = "models/%s.keras"
PICKLE_PATH = "models/%s.pkl"
ENCODER_PATH = "models/label_encoder.pkl"