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

# Posição no vetor
PULSO_INDEX = 0
BASE_INDICADOR_INDEX = 5
BASE_MINIMO_INDEX = 17

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
                    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14),
                    (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]

# Constantes de extração de features

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
FRAME_RATE = 5
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
SEQUENCE_LENGTH = 20
DEFAULT_STEP = 5

LANDMARKER_PATH = "utils/hand_landmarker.task"

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
PARAM_GRID = {
    "units_1": [32, 64],
    "units_2": [64, 128],
    "dropout": [0.2, 0.3],
    "learning_rate": [0.001, 0.0005],
    "batch_size": [8, 16],
}

# Captura de Vídeo em Tempo Real

FONTE = 0
FRAME_SKIP = 20
MIN_CONFIANCA = 0.4
ALTA_CONFIANCA = 0.6

CONFIDENCE_THRESHOLD = 0.5

# Diretórios
VIDEOS_TREINO_DIR = "videos/treino"
VIDEOS_TESTE_DIR = "videos/teste"

FRAMES_TREINO_DIR = "dataset/frames_treino"
FRAMES_TESTE_DIR = "dataset/frames_teste"

DATASET_TREINO_CSV = "dataset/dataset_completo_lstm.csv"
DATASET_TESTE_CSV = "dataset/dataset_teste_lstm.csv"

LOGS_DIR = "logs"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

RESULTADOS_TREINO_CSV = "logs/resultados_treino.csv"
GRID_SEARCH_RESULTS_CSV = "outputs/grid_search_results.csv"
RELATORIO_AVALIACAO_TESTE = "outputs/relatorio_avaliacao_teste.txt"
PREDICOES_TESTE_PATH = "outputs/predicoes_modelo_2.txt"
MATRIZ_CONFUSAO_PNG = "logs/matriz_confusao_modelo_2.png"
LSTM_PATH = "models/lstm_sign_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"
