# Teste ao vivo (`live_testing.py`)

Reconhecimento de um gesto por vez pela webcam, com um modelo treinado pelo `pipeline.py`.

```
python live_testing.py <modelo>
```

O único argumento é o `modelo`: um arquivo `.keras` (lstm, bilstm, tcn, com ou sem atenção) ou `.pkl` (rf, xgboost,
kmeans) salvo pelo `pipeline.py`, por exemplo `models/tcn_at_final.keras`. O resto é fixo, definido em
`utils/constants.py`:

| Constante | Valor | Descrição |
|---|---|---|
| `ENCODER_PATH` | `models/label_encoder.pkl` | LabelEncoder salvo pelo `pipeline.py` |
| `FONTE` | `0` | Índice da câmera |
| `HOLISTIC_LANDMARKER_PATH` | `utils/holistic_landmarker.task` | Modelo do MediaPipe Holistic |
| `LARGURA_JANELA` | `1280` | Largura da janela em pixels (a janela também é redimensionável) |

Teclas: **ESPAÇO** inicia um gesto, **R** cancela, **Q** ou **ESC** saem.

O script recusa modelos cujo número de features por frame não seja `NUM_FEATURES` (142).

## Por que o gesto é gravado explicitamente

Os modelos não têm classe de repouso: o softmax é sempre distribuído entre os gestos, então qualquer janela de
movimento, ou de repouso, recebe algum gesto como resposta. Em vez de prever continuamente, o sistema só prevê sobre
um gesto que o usuário iniciou e que o próprio sistema delimitou.

## Fluxo

```
OCIOSO ──ESPAÇO──▶ CONTAGEM (3 s) ──▶ GRAVANDO ──gesto finalizado──▶ OCIOSO (mostra o resultado)
                   calibra o ruído     espera o movimento
                                       grava até as mãos pararem
```

1. **Contagem** (`CONTAGEM_REGRESSIVA_S`): o usuário se posiciona e fica parado. Nas amostras a partir de
   `CALIBRACAO_INICIO_S` mede-se o movimento do Pose parado, que define o limiar de movimento (abaixo).
2. **Gravação**: a tela pede o gesto. A gravação espera o movimento começar (`GESTO_AMOSTRAS_INICIO` amostras
   seguidas acima do limiar). Sem movimento em `GESTO_INICIO_TIMEOUT_S` a tentativa é cancelada.
3. **Fim do gesto**: quando não há movimento por `GESTO_TEMPO_PARADO_S`, ou a duração chega a `GESTO_DURACAO_MAX_S`.
   Essa espera evita cortar gestos que seguram a posição por pouco tempo. Mãos fora da imagem também contam como parado.
4. **Predição** sobre as amostras entre o início e o fim do movimento, mais `GESTO_MARGEM_AMOSTRAS` de repouso em cada
   ponta (os vídeos de treino têm repouso nas pontas). Gravações com menos de `GESTO_MIN_AMOSTRAS` amostras são
   descartadas como "gesto muito curto".

## Detecção de movimento

A cada amostra, `calcular_movimento` compara o frame com o anterior e retorna um valor relativo aos limiares
(1,0 = no limiar):

- **Braços**: maior deslocamento entre cotovelos e pulsos do Pose, em larguras de ombro, dividido por
  `LIMIAR_MOVIMENTO_BRACOS`. O Pose continua rastreando o corpo quando a detecção das mãos falha, o que acontece
  mais com pouca luz ou longe da câmera.
- **Mãos**: diferença média por feature dos landmarks de cada mão detectada nas duas amostras, dividida por
  `LIMIAR_MOVIMENTO`. Uma mão que aparece ou some não conta como movimento da mão.

O resultado é o maior dos dois. Uma média sobre as 142 features não serve: com a mão não detectada, o movimento do
braço seria diluído pelas features zeradas e nunca passaria do limiar.

### Calibração do limiar pelo ruído

O ruído do Pose parado aumenta com pouca luz e com a distância, e um limiar fixo faria o sistema ver movimento o tempo
todo (o gesto só terminaria em `GESTO_DURACAO_MAX_S`, com a janela cheia de repouso e a confiança baixa). O limiar de
cada gravação é `RUIDO_FATOR` vezes o percentil 90 do movimento medido na contagem, limitado entre 1,0 e
`LIMIAR_MOVIMENTO_MAX`. Se o usuário se mexer no fim da contagem, o limiar sobe. Limiar no teto
(`[INFO] ... limiar de movimento: 4.00` no terminal) indica ambiente ruidoso demais.

## Extração igual à do treino

- **Features**: `normalizar_holistic` (`utils/utils.py`), as mesmas 142 features do `feature_extraction.py`: mão direita
  (63) + mão esquerda (63) + pose (16), relativas ao centro e à largura dos ombros.
- **Taxa**: 1 amostra a cada `FRAME_RATE` frames de um vídeo de `VIDEO_FPS_TREINO` fps, ou seja,
  `AMOSTRAS_POR_SEGUNDO`, medidas pelo relógio. O FPS da câmera não altera a taxa de amostragem.
- **Frames sem pose**: repetem o último frame válido; os do início recebem o primeiro válido, como na extração.
- **Janelas**: `SEQUENCE_LENGTH` frames com salto de `DEFAULT_STEP`. Gestos curtos recebem padding de zeros no final;
  nos longos, a última janela cobre o final e a probabilidade é a média das janelas. Redes Keras usam o tamanho de
  janela do próprio modelo.
- **Lateralidade**: o Holistic roda no frame original, sem espelhar, para direita e esquerda serem as da pessoa. O
  espelhamento é só na exibição.

Diferenças em relação ao treino: um único `HolisticLandmarker` em modo VIDEO (com tracking) processa todos os frames,
enquanto a extração cria um novo por frame; e o pré-processamento do `feature_extraction.py` (letterbox, filtro
bilateral e sharpening) não é aplicado.

## Resultado

A confiança é a probabilidade máxima. Abaixo de `MIN_CONFIANCA` o resultado é "Incerto"; a cor do resultado muda em
`ALTA_CONFIANCA`. A tela mostra o gesto, a confiança e as duas classes seguintes; o terminal registra o gesto, o
número de amostras e as três melhores classes.

## Interface

- Janela redimensionável (`WINDOW_GUI_NORMAL`, sem a barra de zoom e salvar do Qt) e espelhada, com mãos e pontos do
  Pose desenhados.
- Topo: estado atual (Pronto, Prepare-se, Faça o gesto agora, Gravando), instrução e indicadores de detecção do corpo e
  das mãos. Durante a gravação mostra o movimento medido e o limiar.
- Contagem com círculo de progresso, moldura colorida na gravação, barra de "finalizando" enquanto o usuário está
  parado e cartão com o resultado.
- Textos com Pillow (acentos), com fallback para a fonte do OpenCV, sem acentos, se não houver Pillow ou fonte
  TrueType. O layout escala com a altura da imagem.
- FPS exibido como média móvel exponencial.

## Câmera

- O formato MJPG é pedido antes da resolução (1280x720): no formato padrão muitas câmeras caem para ~10 fps em 720p.
- Com a exposição automática, a câmera aumenta o tempo de exposição quando a imagem escurece e o FPS cai (até ~10 fps).
  Iluminar o ambiente é a solução; exposição manual mantém o FPS mas escurece demais a imagem para o Holistic.
- O Holistic leva 25 a 37 ms por frame em CPU, perto do período de uma câmera de 30 fps, e recebe a imagem original da
  câmera (reduzi-la ganharia poucos ms, sem avaliação do efeito na precisão dos dedos).

## Parâmetros

Em `utils/constants.py`, seção "Captura de Vídeo em Tempo Real": `FONTE`, `LARGURA_JANELA`, `MIN_CONFIANCA`, `ALTA_CONFIANCA`,
`VIDEO_FPS_TREINO`, `AMOSTRAS_POR_SEGUNDO`, `CONTAGEM_REGRESSIVA_S`, `GESTO_INICIO_TIMEOUT_S`, `GESTO_TEMPO_PARADO_S`,
`GESTO_DURACAO_MAX_S`, `LIMIAR_MOVIMENTO`, `LIMIAR_MOVIMENTO_BRACOS`, `RUIDO_FATOR`, `LIMIAR_MOVIMENTO_MAX`,
`CALIBRACAO_INICIO_S`, `GESTO_AMOSTRAS_INICIO`, `GESTO_MARGEM_AMOSTRAS`, `GESTO_MIN_AMOSTRAS`. O fps de `VIDEO_FPS_TREINO`
(30) é uma suposição: não há vídeos no repositório para conferir.
