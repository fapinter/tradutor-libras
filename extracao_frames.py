import os

import cv2

from utils.constants import FRAME_RATE, FRAMES_DIR, VIDEO_EXTENSIONS, VIDEOS_DIR


# Estrutura de diretorio
# dataset/
#   - frames/
#       - {treinamento|teste}/
#           - {gesto}/
#               - {id_video}/
#                   - {id_frame}.jpg
#   - videos/
#       - {gesto}/
#           - {minds|malta}/ (treinamento e teste respectivamente)
#               - minds_{id_video}.mp4
def extrair_dataset_completo(pasta_videos=VIDEOS_DIR,
                             pasta_destino_frames=FRAMES_DIR):
    if not os.path.exists(pasta_videos):
        print(f"Aviso: A pasta '{pasta_videos}' nao foi encontrada.")
        return

    gestos = sorted(os.listdir(pasta_videos))
    if not gestos:
        print(f"Aviso: A pasta '{pasta_videos}' está vazia.")
        return

    for gesto in gestos:
        path_gesto = os.path.join(pasta_videos, gesto)
        datasets = os.listdir(path_gesto)
        for dt in datasets:
            path_dt = os.path.join(path_gesto, dt)

            # Processamento dos videos
            print(f'\tExtraindo {gesto} de {path_dt}')            
            for video in sorted(os.listdir(path_dt)):
                if video.endswith(VIDEO_EXTENSIONS):
                    video_path = os.path.join(path_dt, video)
                    output_dir = pasta_destino_frames
                    if dt == 'minds':
                        output_dir = os.path.join(pasta_destino_frames, 'treinamento')
                    else:
                        output_dir = os.path.join(pasta_destino_frames, 'teste')
                    extract_frames(video_path=video_path, output_dir=output_dir, gesture_label=gesto)

# Coleta frames de um video especifico
def extract_frames(video_path,
                   output_dir,
                   gesture_label,
                   frame_rate=FRAME_RATE):
    gesture_dir = os.path.join(output_dir, gesture_label)
    os.makedirs(gesture_dir, exist_ok=True)

    # Get video index by its path
    video_idx = (video_path
        .split('/')[-1]
        .split('_')[1]
        .removesuffix('.mp4')
    )

    output_dir = os.path.join(gesture_dir, video_idx)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame_path = os.path.join(output_dir, f"{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        count += 1

    cap.release()


if __name__ == '__main__':
    extrair_dataset_completo(pasta_videos=VIDEOS_DIR,pasta_destino_frames=FRAMES_DIR)