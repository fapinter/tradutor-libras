"""
live_testing.py
----------------
Script para teste ao vivo do tradutor de Libras em tempo real via câmera do computador.

Fluxo:
1. Captura o vídeo da câmera (webcam).
2. Tenta carregar o modelo LSTM treinado e o Label Encoder da pasta 'models/'.
3. Se o modelo não for encontrado, continua a captura exibindo 'LSTM nao encontrado'.
4. Utiliza o MediaPipe Hand Landmarker para detectar até 2 mãos por frame.
5. Desenha visualmente os landmarks (articulações) e conexões da mão no vídeo.
6. Se o modelo estiver disponível, passa os landmarks para o LSTM prever o gesto.
7. Exibe a tradução e os landmarks na tela em tempo real.
"""

import os
import pickle
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from utils.constants import (
    ALTA_CONFIANCA,
    CONFIDENCE_THRESHOLD,
    ENCODER_PATH,
    FONTE,
    FRAME_SKIP,
    HAND_CONNECTIONS,
    LANDMARKER_PATH,
    LSTM_PATH,
    MIN_CONFIANCA,
    NUM_FEATURES,
    NUM_HANDS,
    SEQUENCE_LENGTH,
)
from utils.constants_cv import (
    BaseOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    VisionRunningMode,
)
from utils.utils import extrair_ambas_maos


def desenhar_landmarks(frame,
                       hand_landmarks_list,
                       handedness_list=None):
    """
    Desenha as articulações (landmarks) e conexões da mão no frame do vídeo.
    Aplica cores distintas para mão direita (Ciano/Verde) e mão esquerda (Magenta/Laranja).
    """
    if not hand_landmarks_list:
        return

    altura, largura = frame.shape[:2]

    for idx, hand_landmarks in enumerate(
            hand_landmarks_list):
        label = "Hand"
        if handedness_list and idx < len(handedness_list):
            label = handedness_list[idx][
                0].category_name  # "Right" ou "Left"

        if label == "Right":
            cor_linha = (255, 191, 0)  # Ciano (BGR)
            cor_ponto = (0, 255, 0)  # Verde (BGR)
            tag_texto = "Mao Direita"
        else:
            cor_linha = (255, 0, 255)  # Magenta (BGR)
            cor_ponto = (0, 165, 255)  # Laranja (BGR)
            tag_texto = "Mao Esquerda"

        pontos = []
        for lm in hand_landmarks:
            cx, cy = int(lm.x * largura), int(lm.y * altura)
            pontos.append((cx, cy))

        # Desenhar conexões entre articulações (esqueleto)
        for inicio, fim in HAND_CONNECTIONS:
            if inicio < len(pontos) and fim < len(pontos):
                cv2.line(frame, pontos[inicio], pontos[fim],
                         cor_linha, 2, cv2.LINE_AA)

        # Desenhar nós de articulação (pontos)
        for i, (cx, cy) in enumerate(pontos):
            raio = 6 if i in (
                4, 8, 12, 16,
                20) else 3  # Destaque na ponta dos dedos
            cv2.circle(frame, (cx, cy), raio, cor_ponto, -1)
            cv2.circle(frame, (cx, cy), raio + 1,
                       (255, 255, 255), 1, cv2.LINE_AA)

        # Rótulo da mão posicionado próximo ao pulso (landmark 0)
        if pontos:
            px, py = pontos[0]
            cv2.putText(
                frame,
                tag_texto,
                (px - 30, py + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def carregar_modelo_lstm(LSTM_PATH=LSTM_PATH,
                         encoder_path=ENCODER_PATH):
    """
    Tenta carregar o modelo LSTM treinado e o Label Encoder salvos no diretório 'models/'.
    Se não for encontrado, retorna (None, None).
    """
    caminhos_modelo = [
        LSTM_PATH,
        "models/lstm_sign_model.keras",
        "models/lstm_direto.h5",
        "models/lstm_csv.h5",
    ]
    modelo_path_encontrado = None
    for path in caminhos_modelo:
        if os.path.exists(path):
            modelo_path_encontrado = path
            break

    caminhos_encoder = [
        encoder_path,
        "models/encoder_direto.pkl",
        "models/encoder_csv.pkl",
    ]
    encoder_path_encontrado = None
    for path in caminhos_encoder:
        if os.path.exists(path):
            encoder_path_encontrado = path
            break

    if not modelo_path_encontrado or not encoder_path_encontrado:
        print(
            "[AVISO] Modelo LSTM ou Label Encoder nao encontrado na pasta 'models/'."
        )
        return None, None

    try:
        model = tf.keras.models.load_model(
            modelo_path_encontrado)
        with open(encoder_path_encontrado, "rb") as f:
            label_encoder = pickle.load(f)
        print(
            f"[OK] Modelo LSTM ({modelo_path_encontrado}) e Label Encoder carregados com sucesso!"
        )
        return model, label_encoder
    except Exception as e:
        print(
            f"[ERRO] Falha ao carregar modelo LSTM ou Label Encoder: {e}"
        )
        return None, None


def abrir_camera(fonte=FONTE):
    """
    Abre a câmera do computador.
    """
    try:
        fonte_int = int(fonte)
        cap = cv2.VideoCapture(fonte_int, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(fonte_int)
        return cap
    except ValueError:
        return cv2.VideoCapture(fonte)


if __name__ == "__main__":
    if not os.path.exists(LANDMARKER_PATH):
        print(
            f"[ERRO] Arquivo de tarefa MediaPipe '{LANDMARKER_PATH}' nao foi encontrado na raiz do projeto."
        )
        sys.exit(1)

    model, label_encoder = carregar_modelo_lstm(
        LSTM_PATH, ENCODER_PATH)
    tem_modelo = model is not None and label_encoder is not None

    cap = abrir_camera(FONTE)
    if not cap.isOpened():
        print(
            f"[ERRO] Nao foi possivel abrir a camera (fonte: {FONTE}). Verifique se esta conectada."
        )
        sys.exit(1)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            LANDMARKER_PATH=LANDMARKER_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=CONFIDENCE_THRESHOLD,
        min_hand_presence_confidence=CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD,
    )

    sequence_buffer = []
    last_known_landmarks = [0.0] * NUM_FEATURES
    gesto_atual = "Aguardando sinal..." if tem_modelo else "LSTM nao encontrado"
    confianca_atual = 0.0
    frame_idx = 0

    inicio_tempo = time.perf_counter()
    ultimo_tempo_fps = inicio_tempo
    frames_fps_count = 0
    fps_exibido = 0.0

    print("=" * 60)
    print(
        "   TESTE AO VIVO - RECONHECIMENTO DE LIBRAS COM MEDIAPIPE   "
    )
    print("=" * 60)
    print(" -> Mostre as maos para a camera.")
    print(
        f" -> Modelo LSTM: {'Carregado' if tem_modelo else 'LSTM nao encontrado'}"
    )
    print(" -> Pressione 'Q' ou 'ESC' para sair.")
    print(
        " -> Pressione 'R' para resetar o buffer de sequencias."
    )
    print("=" * 60 + "\n")

    with HandLandmarker.create_from_options(
            options) as hand_landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(
                    "[AVISO] Falha na leitura do frame da camera."
                )
                break

            frame_idx += 1
            frames_fps_count += 1

            # Inverte o frame horizontalmente para modo espelho (mais intuitivo)
            frame = cv2.flip(frame, 1)
            altura, largura = frame.shape[:2]

            # Converte BGR para RGB para o MediaPipe
            frame_rgb = cv2.cvtColor(frame,
                                     cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb)

            # Timestamp em ms para o modo VIDEO do MediaPipe
            timestamp_ms = int(
                (time.perf_counter() - inicio_tempo) * 1000)
            resultado = hand_landmarker.detect_for_video(
                mp_image, timestamp_ms)

            maos_detectadas = 0

            if resultado.hand_landmarks:
                maos_detectadas = len(
                    resultado.hand_landmarks)
                landmarks = extrair_ambas_maos(
                    resultado.hand_landmarks,
                    resultado.handedness)

                # Se last_known_landmarks era zeros (primeira detecção), faz back-fill nos frames anteriores
                if not any(last_known_landmarks):
                    for idx_buf in range(
                            len(sequence_buffer)):
                        if not any(
                                sequence_buffer[idx_buf]):
                            sequence_buffer[
                                idx_buf] = landmarks

                last_known_landmarks = landmarks

                # DESENHA OS LANDMARKS VISUALMENTE NO FRAME
                desenhar_landmarks(frame,
                                   resultado.hand_landmarks,
                                   resultado.handedness)

            if tem_modelo:
                # Adiciona ao buffer apenas a cada 'FRAME_SKIP' frames
                if frame_idx % FRAME_SKIP == 0:
                    if resultado.hand_landmarks:
                        sequence_buffer.append(landmarks)
                    else:
                        sequence_buffer.append(
                            last_known_landmarks)

                # Quando o buffer atinge o tamanho da sequência (20 frames), faz a predição no modelo LSTM
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(
                        sequence_buffer,
                        axis=0)  # Formato: (1, 20, 126)
                    predicoes = model.predict(input_data,
                                              verbose=0)[0]
                    idx_predito = np.argmax(predicoes)
                    confianca_atual = float(
                        predicoes[idx_predito])

                    if confianca_atual >= MIN_CONFIANCA:
                        gesto_atual = label_encoder.inverse_transform(
                            [idx_predito])[0]
                    else:
                        gesto_atual = "Incerto"

                    sequence_buffer.pop(0)
            else:
                gesto_atual = "LSTM nao encontrado"

            # Cálculo de FPS
            agora = time.perf_counter()
            if agora - ultimo_tempo_fps >= 1.0:
                fps_exibido = frames_fps_count / (
                    agora - ultimo_tempo_fps)
                frames_fps_count = 0
                ultimo_tempo_fps = agora

            # --- INTERFACE GRÁFICA / OVERLAY NA TELA ---
            # Painel superior (Header)
            cv2.rectangle(frame, (0, 0), (largura, 85),
                          (20, 20, 20), -1)

            # Texto do sinal traduzido / Status do modelo
            if tem_modelo:
                cor_gesto = ((0, 255, 0) if confianca_atual
                             >= ALTA_CONFIANCA else
                             ((0, 215,
                               255) if confianca_atual
                              >= MIN_CONFIANCA else
                              (180, 180, 180)))
                texto_principal = f"Traducao: {gesto_atual.upper()}"
                texto_secundario = f"Confianca: {confianca_atual * 100:.1f}% | Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
            else:
                cor_gesto = (0, 0, 255)  # Vermelho
                texto_principal = f"Status: {gesto_atual}"
                texto_secundario = (
                    "Modelo LSTM nao carregado (exibindo apenas deteccao de landmarks)"
                )

            cv2.putText(
                frame,
                texto_principal,
                (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                cor_gesto,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                texto_secundario,
                (15, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # Painel inferior (Status)
            cv2.rectangle(frame, (0, altura - 35),
                          (largura, altura), (20, 20, 20),
                          -1)
            cv2.putText(
                frame,
                f"FPS: {fps_exibido:.1f} | Maos: {maos_detectadas} | [Q] Sair  [R] Resetar Buffer",
                (15, altura - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Exibe a janela com a câmera e os landmarks
            cv2.imshow(
                "Testes Ao Vivo - Tradutor de Libras",
                frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"),
                       27):  # 'q', 'Q' ou ESC
                print(
                    "[INFO] Encerrando o teste ao vivo...")
                break
            elif key in (ord("r"), ord("R")):
                sequence_buffer.clear()
                last_known_landmarks = [0.0] * NUM_FEATURES
                if tem_modelo:
                    gesto_atual = "Aguardando sinal..."
                    confianca_atual = 0.0
                print(
                    "[INFO] Buffer de sequencias resetado.")

    cap.release()
    cv2.destroyAllWindows()
