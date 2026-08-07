import cv2
import os


def extrair_dataset_completo(pasta_videos, pasta_destino_frames):
    """
    Varre a estrutura de pastas e extrai os vídeos automaticamente.
    Ex: extrair_dataset_completo("videos/treino", "dataset/frames_treino")
    """
    if not os.path.exists(pasta_videos):
        print(f"Aviso: A pasta '{pasta_videos}' nao foi encontrada.")
        return

    for gesto_label in os.listdir(pasta_videos):
        caminho_gesto = os.path.join(pasta_videos, gesto_label)

        if not os.path.isdir(caminho_gesto):
            continue

        print(f"\nProcessando vídeos do gesto: {gesto_label}...")

        for nome_video in os.listdir(caminho_gesto):
            if nome_video.lower().endswith(('.mp4', '.avi', '.mov')):
                caminho_video = os.path.join(caminho_gesto, nome_video)
                print(f" -> Extraindo: {nome_video}")
                extract_frames(
                    video_path=caminho_video,
                    output_root_dir=pasta_destino_frames,
                    gesture_label=gesto_label
                )


def extract_frames(video_path, output_root_dir, gesture_label, frame_rate=5):
    """
    Extrai frames de um vídeo e os salva em uma subpasta exclusiva desse vídeo.

    Estrutura gerada:
        output_root_dir/gesture_label/v0000/frame_0.jpg
        output_root_dir/gesture_label/v0001/frame_0.jpg  ← próximo vídeo do mesmo gesto

    Cada chamada cria uma nova subpasta v{N:04d}, nunca sobrescreve vídeos anteriores.
    O parâmetro frame_rate define quantos frames pular entre capturas (não é fps):
        frame_rate=5 captura 1 frame a cada 5 → ~6 fps para vídeos a 30fps.
    """
    gesture_dir = os.path.join(output_root_dir, gesture_label)
    os.makedirs(gesture_dir, exist_ok=True)

    # Conta subpastas de vídeo já existentes para gerar nome único
    existing_video_dirs = [
        d for d in os.listdir(gesture_dir)
        if os.path.isdir(os.path.join(gesture_dir, d)) and d.startswith('v')
    ]
    video_idx = len(existing_video_dirs)
    output_dir = os.path.join(gesture_dir, f"v{video_idx:04d}")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        count += 1

    cap.release()
    print(f"Frames da classe '{gesture_label}' salvos em {output_dir} ({frame_count} frames)")
