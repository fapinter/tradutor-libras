# Constantes para landmarks

NUM_LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
NUM_HANDS = 2
N_MAO = 63
NUM_FEATURES = 126

# Vetor de mão ausente
ZEROS_MAO = [0.0] * N_MAO
# Vetor de features ausente (nenhuma mão detectada)
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

# Constantes do recorte das mãos a partir do MediaPipe Pose (test_crop.py)
POSE_LANDMARKER_PATH = "utils/pose_landmarker_full.task"
# Lado do recorte quadrado de cada mão, em larguras de ombro
CROP_ESCALA_OMBROS = 2.0
# Distância máxima entre o pulso detectado e o pulso do Pose, em larguras de ombro
DIST_MAX_PULSO_POSE = 0.6
# Distância média entre landmarks abaixo da qual duas detecções são a mesma mão, em larguras de ombro
DIST_MESMA_MAO_CROP = 0.1

# Constantes do YOLO (detecção de landmarks das mãos)
# Modelo YOLO-Pose treinado no dataset hand-keypoints da Ultralytics
# (21 keypoints na mesma ordem do MediaPipe)
YOLO_HAND_MODEL_PATH = "utils/yolo_hand_pose.pt"
YOLO_DATASET_YAML = "utils/hand-keypoints.yaml"
# Pesos pré-treinados (COCO-Pose) usados como ponto de partida do treinamento
YOLO_MODELO_BASE = "yolo26n-pose.pt"
YOLO_CONFIANCA = 0.5
# Confiança mínima dos keypoints usados na normalização (pulso, base indicador, base mínimo)
YOLO_CONFIANCA_KPT = 0.5
# Distância média máxima (coordenadas normalizadas) para considerar duas detecções a mesma mão
DIST_MAX_MESMA_MAO = 0.15

# Constantes do YOLO (pose das pessoas, usada para localizar as mãos no test_crop.py)
# Os pesos pré-treinados já são treinados no COCO-Pose (17 keypoints do corpo), sem necessidade de treino
YOLO_PERSON_MODEL_PATH = YOLO_MODELO_BASE
# Índices dos keypoints do COCO-Pose (lado da pessoa, como no MediaPipe Pose)
YOLO_POSE_OMBROS = (6, 5)  # ombro direito, ombro esquerdo
YOLO_POSE_COTOVELOS = {"Right": 8, "Left": 7}
YOLO_POSE_PULSOS = {"Right": 10, "Left": 9}
# O COCO-Pose não tem pontos dos dedos: a ponta da mão é estimada prolongando o antebraço
# (cotovelo -> pulso) a partir do pulso, nesta fração do comprimento do antebraço
YOLO_POSE_COMPRIMENTO_MAO = 0.75

# Parâmetros de Data Augmentation
AUG_NOISE_INTENSITY = 0.005

# escala aleatória aplicada ao pulso
AUG_SCALE_MIN = 0.85
AUG_SCALE_MAX = 1.15

# Rotação máxima
AUG_ROTATION_MAX_DEGREES = 15.0

# Número de amostras geradas pelo Data Augmentation
N_AUMENTOS = 5

# Configurações dos Treinamentos
SEED = 42
LSTM_EPOCHS = 100

# Paciencia Early Stopping (avaliar qual ficará)
LSTM_PATIENCE = 10
PIPELINE_PATIENCE = 8

K_FOLDS = 5
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

# Métricas do melhor treino de cada modelo (f1-macro e acurácia, em %)
BEST_PATH = "best/%s.txt"

KERAS_PATH = "models/%s.keras"
PICKLE_PATH = "models/%s.pkl"
ENCODER_PATH = "models/label_encoder.pkl"