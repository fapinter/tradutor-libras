import cv2
import os
import time
from utils.constants import (
    FRAMES_TREINO_DIR,
    FRAMES_TESTE_DIR
)

pixels_x_treino = []
pixels_y_treino = []
num_images_treino = 0

pixels_x_teste = []
pixels_y_teste = []
num_images_teste = 0

res_dict_teste = {}
res_dict_treino = {}

pixels_x = []
pixels_y = []
num_images = 0

for gesto in os.listdir(FRAMES_TREINO_DIR):
    gesto_dir = os.path.join(FRAMES_TREINO_DIR, gesto)

    for video in os.listdir(gesto_dir):
        video_dir = os.path.join(gesto_dir, video)

        for frame_name in os.listdir(video_dir):
            path = os.path.join(video_dir,frame_name)
            image = cv2.imread(path)
            pixels_x.append(image.shape[1])
            pixels_y.append(image.shape[0])
            num_images += 1

            pixels_x_treino.append(image.shape[1])
            pixels_y_treino.append(image.shape[0])
            num_images_treino += 1

            key = f'{image.shape[1]}_{image.shape[0]}'
            if not key in res_dict_treino.keys():
                res_dict_treino[key] = 1
            else:
                res_dict_treino[key] += 1

for gesto in os.listdir(FRAMES_TESTE_DIR):
    gesto_dir = os.path.join(FRAMES_TESTE_DIR, gesto)

    for video in os.listdir(gesto_dir):
        video_dir = os.path.join(gesto_dir, video)

        for frame_name in os.listdir(video_dir):
            path = os.path.join(video_dir,frame_name)
            image = cv2.imread(path)
            pixels_x.append(image.shape[1])
            pixels_y.append(image.shape[0])
            num_images += 1

            pixels_x_teste.append(image.shape[1])
            pixels_y_teste.append(image.shape[0])
            num_images_teste += 1


            key = f'{image.shape[1]}_{image.shape[0]}'
            if not key in res_dict_teste.keys():
                res_dict_teste[key] = 1
            else:
                res_dict_teste[key] += 1


def test

print('Numero imagens treino:', num_images_treino)
print('Numero imagens teste:', num_images_teste)
print('Numero imagens geral:', num_images)
print('Média pixels X treino: ', sum(pixels_x_treino)/num_images_treino)
print('Média pixels Y treino: ', sum(pixels_y_treino)/num_images_treino)
print('Média pixels X teste: ', sum(pixels_x_teste)/num_images_teste)
print('Média pixels Y teste: ', sum(pixels_y_teste)/num_images_teste)
print('Média pixels X geral: ', sum(pixels_x)/num_images)
print('Média pixels Y geral: ', sum(pixels_y)/num_images)

print('Resoluções videos treino ', res_dict_treino.items())
print('Resoluções videos teste ', res_dict_teste.items())
