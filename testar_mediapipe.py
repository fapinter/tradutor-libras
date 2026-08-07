import argparse
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def abrir_fonte_video(fonte):
    try:
        fonte = int(fonte)
        cap = cv2.VideoCapture(fonte, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(fonte)

        return cap

    except ValueError:
        return cv2.VideoCapture(fonte)


def desenhar_landmarks(frame, hand_landmarks):
    altura, largura = frame.shape[:2]
    pontos = []

    for landmark in hand_landmarks:
        x = int(landmark.x * largura)
        y = int(landmark.y * altura)

        pontos.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    for inicio, fim in HAND_CONNECTIONS:
        if inicio < len(pontos) and fim < len(pontos):
            cv2.line(frame, pontos[inicio], pontos[fim], (255, 255, 255), 2)


def salvar_relatorio(
    total_frames,
    frames_com_mao,
    fps_medio,
    fonte,
    caminho_relatorio="outputs/mediapipe_mundo_real.txt"
):
    os.makedirs(os.path.dirname(caminho_relatorio), exist_ok=True)

    taxa_deteccao = (frames_com_mao / total_frames * 100) if total_frames > 0 else 0

    if taxa_deteccao >= 60:
        status = "OK"
    else:
        status = "ATENCAO"

    conteudo = [
        "RELATORIO - TESTE DO MEDIAPIPE NO MUNDO REAL",
        f"Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        f"Fonte testada: {fonte}",
        f"Total de frames analisados: {total_frames}",
        f"Frames com mao detectada: {frames_com_mao}",
        f"Taxa de deteccao: {taxa_deteccao:.2f}%",
        f"FPS medio aproximado: {fps_medio:.2f}",
        f"Status: {status}",
        "",
        "Criterio:",
        "- OK: mao detectada em pelo menos 60% dos frames.",
        "- ATENCAO: deteccao baixa.",
        "",
        "Se o status for ATENCAO, verificar:",
        "- iluminacao;",
        "- distancia da camera;",
        "- enquadramento da mao;",
        "- fundo muito poluido;",
        "- camera com baixa qualidade.",
    ]

    with open(caminho_relatorio, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(conteudo))

    return taxa_deteccao, status, caminho_relatorio


def testar_mediapipe(fonte="0", duracao=None, modelo="hand_landmarker.task"):
    if not os.path.exists(modelo):
        print(f"Erro: modelo do MediaPipe nao encontrado: {modelo}")
        print("Verifique se o arquivo hand_landmarker.task esta na raiz do projeto.")
        return

    cap = abrir_fonte_video(fonte)

    if not cap.isOpened():
        print(f"Erro: nao foi possivel abrir a fonte de video: {fonte}")
        return

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=modelo),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total_frames = 0
    frames_com_mao = 0

    inicio = time.perf_counter()
    ultimo_tempo_fps = inicio
    frames_fps = 0
    fps_atual = 0

    print("\n--- TESTE DO MEDIAPIPE NO MUNDO REAL ---")
    print("Mostre a mao para a camera.")
    print("Pressione Q para sair.\n")

    with HandLandmarker.create_from_options(options) as hand_landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Fim do video ou falha na leitura da camera.")
                break

            total_frames += 1
            frames_fps += 1

            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            timestamp_ms = int((time.perf_counter() - inicio) * 1000)

            resultado = hand_landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            maos_detectadas = 0

            if resultado.hand_landmarks:
                maos_detectadas = len(resultado.hand_landmarks)
                frames_com_mao += 1

                for hand_landmarks in resultado.hand_landmarks:
                    desenhar_landmarks(frame, hand_landmarks)

            agora = time.perf_counter()

            if agora - ultimo_tempo_fps >= 1:
                fps_atual = frames_fps / (agora - ultimo_tempo_fps)
                frames_fps = 0
                ultimo_tempo_fps = agora

            taxa_deteccao = (frames_com_mao / total_frames * 100) if total_frames > 0 else 0

            if maos_detectadas > 0:
                status_tela = "MediaPipe OK"
                cor_status = (0, 255, 0)
            else:
                status_tela = "Mao nao detectada"
                cor_status = (0, 0, 255)

            cv2.putText(
                frame,
                status_tela,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                cor_status,
                2
            )

            cv2.putText(
                frame,
                f"Maos detectadas: {maos_detectadas}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Taxa de deteccao: {taxa_deteccao:.1f}%",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps_atual:.1f}",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Q = sair",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("Teste MediaPipe - Mundo Real", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if duracao is not None and (time.perf_counter() - inicio) >= duracao:
                break

    cap.release()
    cv2.destroyAllWindows()

    tempo_total = time.perf_counter() - inicio
    fps_medio = total_frames / tempo_total if tempo_total > 0 else 0

    taxa_deteccao, status, caminho_relatorio = salvar_relatorio(
        total_frames=total_frames,
        frames_com_mao=frames_com_mao,
        fps_medio=fps_medio,
        fonte=fonte
    )

    print("\n--- RESULTADO DO TESTE ---")
    print(f"Frames analisados: {total_frames}")
    print(f"Frames com mao detectada: {frames_com_mao}")
    print(f"Taxa de deteccao: {taxa_deteccao:.2f}%")
    print(f"FPS medio: {fps_medio:.2f}")
    print(f"Status final: {status}")
    print(f"Relatorio salvo em: {caminho_relatorio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Teste do MediaPipe no mundo real usando webcam ou video."
    )

    parser.add_argument(
        "--fonte",
        default="0",
        help="Indice da webcam ou caminho do video. Exemplo: 0, 1 ou videos/teste.mp4"
    )

    parser.add_argument(
        "--duracao",
        type=int,
        default=None,
        help="Duracao do teste em segundos. Se nao informar, encerra com Q."
    )

    parser.add_argument(
        "--modelo",
        default="hand_landmarker.task",
        help="Caminho do modelo hand_landmarker.task"
    )

    args = parser.parse_args()

    testar_mediapipe(
        fonte=args.fonte,
        duracao=args.duracao,
        modelo=args.modelo
    )