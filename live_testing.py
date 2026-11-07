"""
live_testing.py
----------------
Teste ao vivo do tradutor de Libras com a webcam.

Uso:
    python live_testing.py models/tcn_at_final.keras
    python live_testing.py models/rf_final.pkl

O único argumento é o modelo. LabelEncoder, câmera, Holistic e tamanho da janela vêm de utils/constants.py.

Os modelos não têm classe de repouso: dada qualquer janela de movimento eles sempre escolhem um dos
gestos. Por isso a predição só roda sobre um gesto gravado de forma explícita:

1. ESPAÇO inicia uma contagem regressiva de CONTAGEM_REGRESSIVA_S segundos.
2. Ao fim da contagem a tela pede o gesto. Um movimento inicia a gravação.
3. O gesto termina quando as mãos ficam paradas (ou fora da imagem) por GESTO_TEMPO_PARADO_S segundos.
4. As amostras gravadas viram janelas de SEQUENCE_LENGTH frames, como no treino, e o modelo prevê o gesto.

A extração é a mesma do treino (feature_extraction.py): 142 features do MediaPipe Holistic normalizadas em
relação ao corpo (normalizar_holistic), 1 frame a cada FRAME_RATE frames de um vídeo de 30 fps. Aqui um único
Holistic em modo VIDEO processa todos os frames da câmera (com tracking), em vez de um por frame.
"""

import argparse
import functools
import os
import sys
import time
import unicodedata

import cv2
import joblib
import mediapipe as mp
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # sem Pillow a interface usa a fonte do OpenCV (sem acentos)
    ImageFont = None

from utils.constants import (
    ALTA_CONFIANCA,
    AMOSTRAS_POR_SEGUNDO,
    CALIBRACAO_INICIO_S,
    CONFIDENCE_THRESHOLD,
    CONTAGEM_REGRESSIVA_S,
    DEFAULT_STEP,
    ENCODER_PATH,
    FONTE,
    GESTO_AMOSTRAS_INICIO,
    GESTO_DURACAO_MAX_S,
    GESTO_INICIO_TIMEOUT_S,
    GESTO_MARGEM_AMOSTRAS,
    GESTO_MIN_AMOSTRAS,
    GESTO_TEMPO_PARADO_S,
    HAND_CONNECTIONS,
    HOLISTIC_LANDMARKER_PATH,
    LARGURA_JANELA,
    LIMIAR_MOVIMENTO,
    LIMIAR_MOVIMENTO_MAX,
    LIMIAR_MOVIMENTO_BRACOS,
    MIN_CONFIANCA,
    N_MAO,
    NUM_FEATURES,
    NUM_FEATURES_MAOS,
    POSE_LANDMARKS_USADOS,
    RUIDO_FATOR,
    SEQUENCE_LENGTH,
)
from utils.constants_cv import (
    BaseOptions,
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    VisionRunningMode,
)
from utils.utils import normalizar_holistic

JANELA = "Testes Ao Vivo - Tradutor de Libras"

# Estados do fluxo de gravação
OCIOSO, CONTAGEM, GRAVANDO = "ocioso", "contagem", "gravando"

# Conexões do Pose entre os pontos usados (ombros, cotovelos, pulsos e quadris)
POSE_CONEXOES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                 (11, 23), (12, 24), (23, 24)]

# Cores (BGR)
COR_DIREITA = (255, 191, 0)
COR_ESQUERDA = (255, 0, 255)
COR_POSE = (0, 255, 0)
COR_FUNDO = (25, 25, 25)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def carregar_modelo(modelo_path, encoder_path):
    """
    Carrega o modelo (.keras ou .pkl, como salvos pelo pipeline.py) e o LabelEncoder.
    Retorna (modelo, label_encoder, tamanho da janela em frames).
    """
    import pickle

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    if modelo_path.endswith(".keras"):
        import tensorflow as tf

        import utils.redes  # noqa: F401  registra a camada AtencaoTemporal para o load_model
        modelo = tf.keras.models.load_model(modelo_path)
        _, janela, n_features = modelo.input_shape
        if n_features != NUM_FEATURES:
            raise ValueError(
                f"O modelo espera {n_features} features por frame, mas a extração gera {NUM_FEATURES} "
                "(Holistic: mãos + pose). Use um modelo treinado com o dataset atual."
            )
        return modelo, label_encoder, janela

    modelo = joblib.load(modelo_path)
    return modelo, label_encoder, SEQUENCE_LENGTH


def prever(modelo, sequencia, janela):
    """
    Prevê o gesto de uma sequência gravada (lista de vetores de NUM_FEATURES) e retorna as probabilidades.

    Mesmo formato do treino: janelas de `janela` frames com salto de DEFAULT_STEP; sequências curtas
    recebem frames zerados no final (padding). Em sequências longas, a média das probabilidades das janelas.
    """
    seq = np.asarray(sequencia, dtype=np.float32)
    if len(seq) < janela:
        padding = np.zeros((janela - len(seq), seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, padding])

    inicios = list(range(0, len(seq) - janela + 1, DEFAULT_STEP))
    if inicios[-1] != len(seq) - janela:
        inicios.append(len(seq) - janela)  # a última janela cobre o final do gesto
    janelas = np.stack([seq[i:i + janela] for i in inicios])

    if hasattr(modelo, "predict_proba"):
        probs = modelo.predict_proba(janelas)
    else:
        probs = modelo.predict(janelas, verbose=0)
    return np.mean(probs, axis=0)


# ---------------------------------------------------------------------------
# Gravação do gesto
# ---------------------------------------------------------------------------

def limiar_pelo_ruido(movimentos_parado):
    """Limiar de movimento a partir do movimento medido com o usuário parado (ver RUIDO_FATOR)."""
    if len(movimentos_parado) < 4:
        return 1.0
    return float(min(max(RUIDO_FATOR * np.percentile(movimentos_parado, 90), 1.0), LIMIAR_MOVIMENTO_MAX))


def calcular_movimento(atual, anterior):
    """
    Movimento entre duas amostras, relativo aos limiares (1.0 = no limiar, acima disso há movimento).
    Combina o deslocamento de cotovelos e pulsos do Pose (sempre disponível) com a variação dos landmarks
    das mãos detectadas nas duas amostras. As features do Pose são x e y de 8 pontos, depois das mãos,
    na ordem de POSE_LANDMARKS_USADOS (cotovelos e pulsos: pontos 2 a 5).
    """
    atual, anterior = np.asarray(atual), np.asarray(anterior)
    pose_atual = atual[NUM_FEATURES_MAOS:].reshape(-1, 2)[2:6]
    pose_anterior = anterior[NUM_FEATURES_MAOS:].reshape(-1, 2)[2:6]
    bracos = np.linalg.norm(pose_atual - pose_anterior, axis=1).max() / LIMIAR_MOVIMENTO_BRACOS

    maos = 0.0
    for ini in (0, N_MAO):
        a, b = atual[ini:ini + N_MAO], anterior[ini:ini + N_MAO]
        if a.any() and b.any():  # mão detectada nas duas amostras
            maos = max(maos, float(np.mean(np.abs(a - b))) / LIMIAR_MOVIMENTO)
    return float(max(bracos, maos))


class GravadorGesto:
    """
    Acumula as amostras (1 a cada 1/AMOSTRAS_POR_SEGUNDO s) durante a gravação e detecta o início e o fim
    do gesto pelo movimento entre amostras consecutivas.
    """

    def __init__(self, inicio_s, limiar=1.0):
        self.inicio_s = inicio_s
        self.limiar = limiar        # movimento (em unidades de calcular_movimento) acima do qual há gesto
        self.amostras = []          # features por amostra (None: pose não detectada)
        self.em_movimento = []      # se houve movimento ao chegar em cada amostra
        self.iniciado = False
        self.movimento = 0.0        # último movimento medido (1.0 = limiar)
        self.consecutivas = 0
        self.idx_inicio = 0
        self.idx_ultimo_mov = 0

    @property
    def segundos_parado(self):
        if not self.iniciado:
            return 0.0
        return (len(self.amostras) - 1 - self.idx_ultimo_mov) / AMOSTRAS_POR_SEGUNDO

    def adicionar(self, features, agora):
        """
        Registra uma amostra. Retorna None se a gravação continua, ou (motivo, sequência) ao terminar:
        motivo "ok", "timeout" (ninguém começou o gesto) ou "curto".
        """
        anterior = next((a for a in reversed(self.amostras) if a is not None), None)
        if features is None or anterior is None:
            movimento = 0.0
        else:
            movimento = calcular_movimento(features, anterior)
        self.movimento = movimento
        mov = movimento > self.limiar
        self.amostras.append(features)
        self.em_movimento.append(mov)
        idx = len(self.amostras) - 1

        if not self.iniciado:
            self.consecutivas = self.consecutivas + 1 if mov else 0
            if self.consecutivas >= GESTO_AMOSTRAS_INICIO:
                self.iniciado = True
                self.idx_inicio = idx - self.consecutivas + 1
                self.idx_ultimo_mov = idx
            elif agora - self.inicio_s > GESTO_INICIO_TIMEOUT_S:
                return "timeout", None
            return None

        if mov:
            self.idx_ultimo_mov = idx
        duracao = (idx - self.idx_inicio + 1) / AMOSTRAS_POR_SEGUNDO
        if self.segundos_parado >= GESTO_TEMPO_PARADO_S or duracao >= GESTO_DURACAO_MAX_S:
            return self._finalizar()
        return None

    def _finalizar(self):
        # Mantém um pouco de repouso nas pontas, como nos vídeos de treino
        ini = max(0, self.idx_inicio - GESTO_MARGEM_AMOSTRAS)
        fim = self.idx_ultimo_mov + 1 + GESTO_MARGEM_AMOSTRAS
        seq = self.amostras[ini:fim]
        if len(seq) < GESTO_MIN_AMOSTRAS:
            return "curto", None

        # Frames sem pose: forward-fill com o último frame válido e back-fill no início (como no treino)
        validos = [s for s in seq if s is not None]
        if not validos:
            return "curto", None
        ultimo = validos[0]
        for i, s in enumerate(seq):
            if s is None:
                seq[i] = ultimo
            else:
                ultimo = s
        return "ok", seq


# ---------------------------------------------------------------------------
# Desenho
# ---------------------------------------------------------------------------

def _pontos(landmarks, largura, altura):
    return [(int(lm.x * largura), int(lm.y * altura)) for lm in landmarks]


def desenhar_landmarks(frame, resultado):
    """Desenha as mãos (direita/esquerda da pessoa) e os pontos do Pose usados pelo modelo."""
    altura, largura = frame.shape[:2]

    for landmarks, cor_linha, cor_ponto in (
        (resultado.right_hand_landmarks, COR_DIREITA, (0, 255, 0)),
        (resultado.left_hand_landmarks, COR_ESQUERDA, (0, 165, 255)),
    ):
        if not landmarks:
            continue
        pontos = _pontos(landmarks, largura, altura)
        for inicio, fim in HAND_CONNECTIONS:
            cv2.line(frame, pontos[inicio], pontos[fim], cor_linha, 2, cv2.LINE_AA)
        for i, p in enumerate(pontos):
            raio = 6 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, p, raio, cor_ponto, -1)
            cv2.circle(frame, p, raio + 1, (255, 255, 255), 1, cv2.LINE_AA)

    if resultado.pose_landmarks:
        pontos = _pontos(resultado.pose_landmarks, largura, altura)
        for inicio, fim in POSE_CONEXOES:
            cv2.line(frame, pontos[inicio], pontos[fim], COR_POSE, 2, cv2.LINE_AA)
        for i in POSE_LANDMARKS_USADOS:
            cv2.circle(frame, pontos[i], 5, COR_POSE, -1)
            cv2.circle(frame, pontos[i], 6, (255, 255, 255), 1, cv2.LINE_AA)


FONTES = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
FONTES_NEGRITO = [
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# Cores da interface (BGR)
COR_TEXTO = (255, 255, 255)
COR_TEXTO_SUAVE = (200, 200, 200)
COR_AZUL = (255, 160, 60)
COR_AMARELO = (0, 200, 255)
COR_VERDE = (90, 220, 90)
COR_VERMELHO = (70, 70, 255)
COR_CINZA = (150, 150, 150)


@functools.lru_cache(maxsize=None)
def _fonte(tamanho, negrito):
    """Fonte TrueType (permite acentos). None se o sistema não tiver nenhuma das fontes conhecidas."""
    if ImageFont is None:
        return None
    for caminho in (FONTES_NEGRITO if negrito else []) + FONTES:
        if os.path.exists(caminho):
            return ImageFont.truetype(caminho, tamanho)
    return None


class Overlay:
    """
    Desenha formas (OpenCV) e textos (Pillow, com acentos) sobre o frame. Os textos são acumulados
    e aplicados de uma vez em `finalizar`, para converter o frame para Pillow só uma vez.
    """

    def __init__(self, frame):
        self.frame = frame
        self.u = frame.shape[0] / 720  # unidade: tudo escala com a altura da imagem
        self.textos = []

    def painel(self, x0, y0, x1, y1, alpha=0.65, cor=COR_FUNDO):
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1), self.frame.shape[1]), min(int(y1), self.frame.shape[0])
        roi = self.frame[y0:y1, x0:x1]
        roi[:] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, cor), alpha, 0)

    def texto(self, txt, x, y, tamanho, cor=COR_TEXTO, negrito=False, ancora="l"):
        """ancora: l (x é a borda esquerda), m (centro) ou r (borda direita); y é o centro vertical."""
        self.textos.append((txt, int(x), int(y), max(int(tamanho * self.u), 8), cor, negrito, ancora))

    def barra(self, x, y, largura, altura, fracao, cor):
        x, y, largura, altura = int(x), int(y), int(largura), int(altura)
        cv2.rectangle(self.frame, (x, y), (x + largura, y + altura), (90, 90, 90), -1)
        cv2.rectangle(self.frame, (x, y), (x + int(largura * min(max(fracao, 0), 1)), y + altura), cor, -1)

    def finalizar(self):
        if not self.textos:
            return
        if _fonte(10, False) is None:  # sem fonte TrueType: OpenCV, sem acentos
            for txt, x, y, tam, cor, negrito, ancora in self.textos:
                txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
                escala, esp = tam / 30, 2 if negrito else 1
                (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, esp)
                x -= {"l": 0, "m": w // 2, "r": w}[ancora]
                cv2.putText(self.frame, txt, (x, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, esp, cv2.LINE_AA)
            return
        imagem = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(imagem)
        for txt, x, y, tam, cor, negrito, ancora in self.textos:
            draw.text((x, y), txt, font=_fonte(tam, negrito), fill=cor[::-1], anchor=ancora + "m")
        self.frame[:] = cv2.cvtColor(np.asarray(imagem), cv2.COLOR_RGB2BGR)


def _cor_confianca(confianca):
    if confianca >= ALTA_CONFIANCA:
        return COR_VERDE
    return COR_AMARELO if confianca >= MIN_CONFIANCA else COR_CINZA


def _cabecalho(ov, estado, gravador, agora, ultimo, tem_pose):
    """Título e subtítulo do topo, e a cor do indicador de status."""
    if estado == CONTAGEM:
        return COR_AMARELO, "Prepare-se", "O gesto começa quando a contagem chegar a zero"
    if estado == GRAVANDO:
        if not gravador.iniciado:
            restante = max(GESTO_INICIO_TIMEOUT_S - (agora - gravador.inicio_s), 0)
            return COR_VERDE, "Faça o gesto agora!", f"Aguardando o início do movimento ({restante:.0f}s)  ·  movimento detectado: {gravador.movimento:.1f} (precisa passar de {gravador.limiar:.1f})"
        piscando = int(agora * 2) % 2 == 0
        return (COR_VERMELHO if piscando else COR_CINZA), "Gravando o gesto", f"Fique parado quando terminar  ·  movimento: {gravador.movimento:.1f} (parado abaixo de {gravador.limiar:.1f})"
    if isinstance(ultimo, str):
        return COR_AMARELO, "Não foi possível traduzir", ultimo
    if not tem_pose:
        return COR_AMARELO, "Posicione-se na câmera", "Deixe ombros, braços e mãos visíveis"
    return COR_AZUL, "Pronto", "Pressione ESPAÇO e faça o gesto quando a contagem terminar"


def desenhar_interface(frame, estado, gravador, inicio_contagem, agora, ultimo, fps, n_maos, tem_pose):
    """
    Overlay de status sobre o frame já espelhado e redimensionado. `ultimo` é o último resultado
    (gesto, confiança, top3) ou uma mensagem de erro (str).
    """
    ov = Overlay(frame)
    u = ov.u
    altura, largura = frame.shape[:2]
    cx = largura // 2

    # Moldura colorida durante a gravação
    if estado == GRAVANDO:
        cor_borda = COR_VERDE if not gravador.iniciado else COR_VERMELHO
        cv2.rectangle(frame, (0, 0), (largura - 1, altura - 1), cor_borda, max(int(8 * u), 2))

    # Topo: indicador, título, subtítulo e status da detecção
    ov.painel(0, 0, largura, 84 * u)
    cor, titulo, subtitulo = _cabecalho(ov, estado, gravador, agora, ultimo, tem_pose)
    cv2.circle(frame, (int(36 * u), int(42 * u)), int(13 * u), cor, -1, cv2.LINE_AA)
    ov.texto(titulo, 64 * u, 30 * u, 30, COR_TEXTO, True)
    ov.texto(subtitulo, 64 * u, 63 * u, 18, COR_TEXTO_SUAVE)
    for i, (rotulo, ok) in enumerate((("Corpo", tem_pose), (f"Mãos: {n_maos}", n_maos > 0))):
        x = largura - (230 - 105 * i) * u
        cv2.circle(frame, (int(x), int(42 * u)), int(7 * u), COR_VERDE if ok else COR_CINZA, -1, cv2.LINE_AA)
        ov.texto(rotulo, x + 14 * u, 42 * u, 18, COR_TEXTO_SUAVE)

    # Base: atalhos e FPS
    ov.painel(0, altura - 44 * u, largura, altura)
    ov.texto("ESPAÇO  iniciar gesto      R  cancelar      Q  sair", 20 * u, altura - 22 * u, 17, COR_TEXTO_SUAVE)
    ov.texto(f"{fps:.0f} FPS", largura - 20 * u, altura - 22 * u, 15, COR_CINZA, ancora="r")

    if estado == CONTAGEM:
        # Círculo com o número e o progresso da contagem
        decorrido = agora - inicio_contagem
        centro, raio = (cx, int(altura * 0.5)), int(95 * u)
        sobre = frame.copy()
        cv2.circle(sobre, centro, raio, COR_FUNDO, -1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(sobre, 0.75, frame, 0.25, 0)
        cv2.ellipse(frame, centro, (raio, raio), -90, 0, 360 * min(decorrido / CONTAGEM_REGRESSIVA_S, 1),
                    COR_AMARELO, max(int(8 * u), 2), cv2.LINE_AA)
        ov.texto(str(int(np.ceil(max(CONTAGEM_REGRESSIVA_S - decorrido, 0.01)))), centro[0], centro[1], 110, COR_AMARELO, True, "m")

    elif estado == GRAVANDO and gravador.iniciado:
        ov.painel(cx - 220 * u, altura - 112 * u, cx + 220 * u, altura - 56 * u)
        ov.texto("Finalizando o gesto quando as mãos pararem", cx, altura - 95 * u, 16, COR_TEXTO_SUAVE, ancora="m")
        ov.barra(cx - 190 * u, altura - 76 * u, 380 * u, 10 * u, gravador.segundos_parado / GESTO_TEMPO_PARADO_S, COR_AMARELO)

    elif estado == OCIOSO and isinstance(ultimo, tuple):
        gesto, confianca, top3 = ultimo
        cor_res = _cor_confianca(confianca)
        topo = altura - 44 * u - 175 * u
        ov.painel(cx - 300 * u, topo, cx + 300 * u, altura - 58 * u, 0.75)
        ov.texto("TRADUÇÃO", cx - 280 * u, topo + 22 * u, 15, COR_CINZA, True)
        ov.texto(gesto.capitalize(), cx - 280 * u, topo + 62 * u, 46, cor_res, True)
        ov.texto(f"{confianca * 100:.0f}%", cx + 280 * u, topo + 62 * u, 40, cor_res, True, "r")
        ov.barra(cx - 280 * u, topo + 98 * u, 560 * u, 8 * u, confianca, cor_res)
        if gesto == "incerto":
            ov.texto("Confiança baixa. Pressione ESPAÇO para tentar de novo", cx - 280 * u, topo + 130 * u, 17, COR_TEXTO_SUAVE)
        else:
            outras = "   ·   ".join(f"{g.capitalize()} {p * 100:.0f}%" for g, p in top3[1:])
            ov.texto("Outras possibilidades: " + outras, cx - 280 * u, topo + 130 * u, 17, COR_TEXTO_SUAVE)

    ov.finalizar()


# ---------------------------------------------------------------------------
# Câmera e laço principal
# ---------------------------------------------------------------------------

def abrir_camera(fonte):
    """Abre a câmera por índice (ex.: 0) ou caminho/URL de vídeo."""
    try:
        indice = int(fonte)
    except ValueError:
        return cv2.VideoCapture(fonte)
    if sys.platform == "win32":
        cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(indice)


def criar_holistic(holistic_path):
    """Um único Holistic em modo VIDEO, usado em todos os frames da câmera."""
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=holistic_path),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=CONFIDENCE_THRESHOLD,
        min_pose_landmarks_confidence=CONFIDENCE_THRESHOLD,
        min_hand_landmarks_confidence=CONFIDENCE_THRESHOLD,
    )
    return HolisticLandmarker.create_from_options(options)


def resultado_da_predicao(probs, label_encoder):
    """Gesto previsto, confiança e as 3 melhores classes. Abaixo de MIN_CONFIANCA o gesto é 'incerto'."""
    ordem = np.argsort(probs)[::-1][:3]
    top3 = [(str(label_encoder.inverse_transform([i])[0]), float(probs[i])) for i in ordem]
    gesto, confianca = top3[0]
    if confianca < MIN_CONFIANCA:
        gesto = "incerto"
    return gesto, confianca, top3


def main():
    parser = argparse.ArgumentParser(description="Teste ao vivo do tradutor de Libras.")
    parser.add_argument("modelo", help="caminho do modelo treinado (.keras ou .pkl), ex.: models/tcn_at_final.keras")
    args = parser.parse_args()

    for caminho, nome in ((args.modelo, "Modelo"), (ENCODER_PATH, "LabelEncoder"), (HOLISTIC_LANDMARKER_PATH, "Holistic")):
        if not os.path.exists(caminho):
            sys.exit(f"[ERRO] {nome} '{caminho}' nao foi encontrado.")

    try:
        modelo, label_encoder, janela = carregar_modelo(args.modelo, ENCODER_PATH)
    except Exception as e:
        sys.exit(f"[ERRO] Falha ao carregar o modelo: {e}")
    print(f"[OK] Modelo '{args.modelo}' carregado (janela de {janela} frames, {len(label_encoder.classes_)} gestos).")

    cap = abrir_camera(FONTE)
    if not cap.isOpened():
        sys.exit(f"[ERRO] Nao foi possivel abrir a camera (fonte: {FONTE}).")

    # Pede à câmera a resolução HD (se não suportar, ela mantém a própria); a janela é redimensionável.
    # O formato MJPG vem antes da resolução: no padrão (YUYV) muitas câmeras caem para ~10 fps em 720p
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow(JANELA, cv2.WINDOW_GUI_NORMAL)  # redimensionável, sem a barra de zoom/salvar do Qt

    estado = OCIOSO
    gravador = None
    inicio_contagem = 0.0
    proxima_amostra = 0.0
    calibracao = []
    calib_anterior = None
    ultimo = None  # último resultado (texto, confianca, top3) ou mensagem
    intervalo = 1.0 / AMOSTRAS_POR_SEGUNDO

    t0 = time.perf_counter()
    ultimo_ts = -1
    ultimo_frame = t0
    fps = 0.0
    janela_ajustada = False

    print("ESPACO: iniciar gesto | R: cancelar | Q/ESC: sair")
    with criar_holistic(HOLISTIC_LANDMARKER_PATH) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Falha na leitura do frame da camera.")
                break
            altura, largura = frame.shape[:2]
            agora = time.perf_counter()

            # Holistic no frame original (sem espelhar: direita/esquerda são as da pessoa, como no treino)
            ts = max(int((agora - t0) * 1000), ultimo_ts + 1)
            ultimo_ts = ts
            imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            resultado = holistic.detect_for_video(imagem, ts)
            tem_pose = bool(resultado.pose_landmarks)
            n_maos = bool(resultado.right_hand_landmarks) + bool(resultado.left_hand_landmarks)
            desenhar_landmarks(frame, resultado)

            if estado == CONTAGEM and agora - inicio_contagem >= CONTAGEM_REGRESSIVA_S:
                estado = GRAVANDO
                limiar = limiar_pelo_ruido(calibracao)
                print(f"[INFO] Ruido parado: {len(calibracao)} amostras | limiar de movimento: {limiar:.2f}")
                gravador = GravadorGesto(agora, limiar)
                proxima_amostra = agora

            elif estado == CONTAGEM and agora >= proxima_amostra:
                # Calibração: mede o movimento (ruído do Pose) com o usuário parado, a cada amostra
                proxima_amostra = max(proxima_amostra + intervalo, agora)
                features = normalizar_holistic(resultado, largura, altura)
                if features is not None and calib_anterior is not None and agora - inicio_contagem >= CALIBRACAO_INICIO_S:
                    calibracao.append(calcular_movimento(features, calib_anterior))
                calib_anterior = features

            elif estado == GRAVANDO and agora >= proxima_amostra:
                # Sem acumular atraso se um frame demorar mais que o intervalo
                proxima_amostra = max(proxima_amostra + intervalo, agora)
                fim = gravador.adicionar(normalizar_holistic(resultado, largura, altura), agora)
                if fim is not None:
                    motivo, sequencia = fim
                    estado = OCIOSO
                    if motivo == "ok":
                        probs = prever(modelo, sequencia, janela)
                        ultimo = resultado_da_predicao(probs, label_encoder)
                        print(f"[GESTO] {ultimo[0]} ({ultimo[1] * 100:.1f}%) | {len(sequencia)} amostras | top3: {ultimo[2]}")
                    elif motivo == "timeout":
                        ultimo = "Nenhum gesto detectado. ESPACO para tentar de novo"
                    else:
                        ultimo = "Gesto muito curto. ESPACO para tentar de novo"

            # FPS suavizado (média móvel exponencial) para o número não oscilar na tela
            instantaneo = 1.0 / max(agora - ultimo_frame, 1e-6)
            fps = instantaneo if fps == 0.0 else 0.95 * fps + 0.05 * instantaneo
            ultimo_frame = agora

            # Espelha e amplia só para exibição, depois de desenhar os landmarks (os textos não ficam invertidos)
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (LARGURA_JANELA, int(altura * LARGURA_JANELA / largura)), interpolation=cv2.INTER_LINEAR)
            if not janela_ajustada:
                cv2.resizeWindow(JANELA, frame.shape[1], frame.shape[0])
                janela_ajustada = True
            desenhar_interface(frame, estado, gravador, inicio_contagem, agora, ultimo, fps, n_maos, tem_pose)
            cv2.imshow(JANELA, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" ") and estado == OCIOSO:
                estado = CONTAGEM
                inicio_contagem = agora
                proxima_amostra = agora
                calibracao, calib_anterior = [], None
                ultimo = None
            elif key in (ord("r"), ord("R")):
                estado = OCIOSO
                ultimo = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
