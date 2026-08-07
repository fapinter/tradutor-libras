# 📊 Sumário Completo e Análise do Projeto: Tradutor de LIBRAS

Este documento apresenta uma análise detalhada, exaustiva e estruturada de todos os arquivos, diretórios, funções e componentes do projeto **Tradutor de LIBRAS**. 

---

## 1. 🎯 Visão Geral do Arquitetura do Projeto

O projeto é um sistema completo de visão computacional e aprendizado de máquina para reconhecimento e tradução automática da Língua Brasileira de Sinais (LIBRAS). Ele abrange desde a captura de vídeos brutos até a inferência em tempo real via webcam ou arquivos de vídeo.

### Fluxo de Dados e Processamento Pipeline
```
[Vídeos de Entrada (MP4/AVI)] 
        ↓ 
[data_preprocessing.py] → Extração de Frames organizados por vídeo (v0000, v0001...)
        ↓ 
[feature_extraction.py] → MediaPipe Hand Landmarker (126 Coordenadas Normalizadas por Frame)
        ↓ 
[landmark_augmentation.py] → Data Augmentation Geométrico (Treino Apenas: Ruído, Escala, Rotação 2D, Espelhamento)
        ↓ 
[model_training.py] → Treinamento dos Classificadores:
                        1. Random Forest (Estático / Frame-a-frame)
                        2. LSTM (Dinâmico / Sequências Temporais)
                        3. KNN + K-Means Temporal (Compactação Temporal Baseada no SBrT 2023)
        ↓ 
[sign_recognition.py] → Inferência em Tempo Real / Vídeo com Voto Majoritário
```

---

## 2. 📁 Análise Detalhada por Arquivo e Diretório

### 2.1. Arquivos Executáveis e Módulos Python (`.py`)

#### 📄 `main.py`
- **O que implementa:** Ponto de entrada (CLI) principal da aplicação. Gerencia um menu interativo (opções 0 a 7) para orquestrar todo o ciclo de vida do sistema (extração, geração de CSV, treinamento, testes em webcam/vídeo, comparação de pipelines e matriz de confusão).
- **Funções presentes:**
  - `_salvar_log_treino(tipo_modelo, acuracia, n_amostras, augmentar, n_aumentos)`: Registra os resultados e metadados de cada treinamento em `logs/resultados_treino.csv`.
  - `extrair_dataset_completo(videos, pasta_destino_frames)`: Varre diretórios de vídeo e extrai frames.
  - `executar_extracao_de_frames()`: Executa em lote a extração de frames para as pastas de treino e teste.
  - `_perguntar_augmentation()`: Interage com o usuário para obter preferências de data augmentation.
  - `gerar_dataset_csv()`: Extrai as features originais (sem augmentation) e salva em arquivo CSV (`dataset_completo_rf.csv` ou `dataset_completo_lstm.csv`).
  - `executar_treinamento()`: Conduz o usuário na escolha do modelo, fonte de dados (RAM vs CSV) e inicia o treino.
  - `comparar_pipelines()`: Compara a acurácia de treinamento direto da RAM versus carregamento via CSV para garantir paridade nos dados.
  - Bloco `__main__`: Loop principal exibindo o menu numérico interativo.
- **Papel no projeto:** Centralizar a usabilidade do projeto, permitindo executar todos os fluxos sem necessidade de passar argumentos por linha de comando.

---

#### 📄 `feature_extraction.py`
- **O que implementa:** Módulo central de extração e normalização de pontos anatômicos (landmarks) de mãos utilizando a API Tasks (`HandLandmarker`) do MediaPipe.
- **Funções presentes:**
  - `_normalizar_mao(hand_landmarks)`: Centraliza as 63 coordenadas de uma mão no pulso (lm 0) e escala X e Y pela largura da palma (distância 2D entre lm 5 e lm 17). Mantém a profundidade Z relativa.
  - `_extrair_ambas_maos(hand_landmarks_list, handedness_list)`: Organiza os vetores das duas mãos no formato fixo de 126 features `[Mão Direita (63) + Mão Esquerda (63)]`. Preenche mãos não detectadas com blocos de zeros.
  - `_listar_frames(directory)`: Ordena numericamente os arquivos de imagem de um diretório de vídeo.
  - `_video_dirs_de_classe(class_dir)`: Identifica subpastas por vídeo (`v0000`, `v0001`), evitando que janelas deslizantes LSTM misturem frames de vídeos diferentes.
  - `extract_features_from_directory(...)`: Varre o dataset de frames e gera matrizes de dados 2D `(amostras, 126)` para modelos estáticos ou 3D `(amostras, frames, 126)` para LSTM, realizando forward-fill em falhas de detecção.
  - `_exportar_csv(features, labels, mode)`: Exporta o conjunto de dados extraído para arquivos CSV.
- **Papel no projeto:** Converter imagens brutas de mãos em vetores numéricos estruturados e espacialmente invariantes a distância da câmera e posição na tela.

---

#### 📄 `landmark_augmentation.py`
- **O que implementa:** Módulo de aumento sintético de dados (Data Augmentation) que opera diretamente nos vetores de landmarks 2D e 3D sem precisar reprocessar imagens.
- **Funções presentes:**
  - `_reshape(frame_mao)` / `_flatten(pts)`: Converte entre lista plana (63,) e matriz 2D (21, 3).
  - `_eh_ausente(pts)`: Verifica se o bloco de landmarks representa uma mão ausente (todos zeros).
  - `_ruido(pts, intensidade)`: Adiciona ruído gaussiano (σ=0.005) para simular imprecisão do sensor/tremores.
  - `_escala(pts, fator_min, fator_max)`: Escala os landmarks em relação ao pulso (0.85 a 1.15).
  - `_rotacao_2d(pts, angulo_max_graus)`: Aplica matriz de rotação 2D no plano XY (±15°).
  - `_espelhamento(pts)`: Inverte o eixo X.
  - `_aplicar(...)`: Aplica transformações com parâmetros fixos ou sorteados na hora.
  - `_processar_frame(...)`: Lida com 1 ou 2 mãos, realizando espelhamento e troca de lateralidade (direita ↔ esquerda).
  - `augmentar_frame(frame, transformacoes)`: Interface pública para um frame isolado (63 ou 126 features).
  - `augmentar_sequencia(sequencia, transformacoes)`: Interface pública para sequências LSTM. Garante coerência temporal aplicando os **mesmos parâmetros geométricos em todos os 20 frames da sequência**.
  - `gerar_amostras_aumentadas(...)`: Gera variações sintéticas para o dataset inteiro (1 espelhamento obrigatório + N-1 combinações aleatórias).
- **Papel no projeto:** Multiplicar o volume do dataset de treino (ex: 6x mais dados), evitando *overfitting* em conjuntos de dados com poucos vídeos e aumentando a capacidade de generalização dos modelos.

---

#### 📄 `model_training.py`
- **O que implementa:** Módulo de construção, treinamento e avaliação dos três classificadores do projeto (Random Forest, KNN e LSTM).
- **Funções presentes:**
  - `train_random_forest(...)`: Realiza split 80/20 estratificado, aplica augmentation opcional no conjunto de treino (pós-split), treina um `RandomForestClassifier` (200 árvores, peso balanceado) e salva o modelo serializado.
  - `_aplicar_kmeans_temporal(sequences, n_clusters)`: Implementação da técnica do artigo Caiafa et al. (SBrT 2023). Agrupa frames de cada sequência com K-Means, ordena os centróides cronologicamente pela mediana dos índices de frames e achata o resultado em um vetor 1D fixo.
  - `train_knn(...)`: Treina um `KNeighborsClassifier` com padronização via `StandardScaler`. Caso a entrada seja 3D (sequências), aplica o K-Means temporal antes da classificação.
  - `train_lstm(...)`: Constrói e treina uma rede neural recorrente Keras de duas camadas LSTM (64 e 128 unidades) com *recurrent dropout*, *early stopping* e ponderação de classes (*class_weight*).
- **Papel no projeto:** Fornecer os motores de aprendizado supervisionado para classificar gestos estáticos e dinâmicos de LIBRAS.

---

#### 📄 `model_training_leaveoneout.py`
- **O que implementa:** Versão alternativa do treinamento que utiliza validação cruzada *Leave-One-Out* (LOOCV), indicada para validação rigorosa em datasets muito pequenos. Contém barras de progresso com `tqdm`.
- **Funções presentes:**
  - `train_random_forest(...)`: Avalia RF via LOOCV amostra a amostra e retreina o modelo final com 100% dos dados.
  - `train_knn(...)`: Avalia KNN via LOOCV com `StandardScaler` por dobra e retreina modelo final.
  - `build_lstm_model(input_shape, num_classes)`: Função auxiliar para instanciar a arquitetura Keras LSTM limpa em cada iteração do LOOCV.
  - `train_lstm(...)`: Executa LOOCV na rede LSTM limpando a sessão Keras a cada iteração (`K.clear_session()`).
- **Papel no projeto:** Servir como script de validação acadêmica alternativa para cálculo da taxa de erro assintótica via LOOCV.

---

#### 📄 `sign_recognition.py`
- **O que implementa:** Motor de inferência em tempo real e processamento de arquivos de vídeo de teste.
- **Funções presentes:**
  - `_kmeans_temporal_single(seq_array, n_clusters)`: Versão individual da compactação K-Means para um único buffer de inferência.
  - `_normalizar_mao(hand_landmarks)`: Reimplementação da normalização de landmarks.
  - `extrair_landmarks(hand_landmarks_list, handedness_list)`: Reimplementação da extração de 126 features.
  - `recognize_sign(video_path, tipo_modelo)`: Carrega o modelo escolhido (1: RF, 2: LSTM, 3: KNN), abre o vídeo/webcam com OpenCV, amostra os frames no mesmo intervalo do treino (`FRAME_RATE_TREINO = 5`), mantém buffer deslizante, renderiza a tradução na tela e retorna o gesto vencedor por voto majoritário.
- **Papel no projeto:** Executar o reconhecimento prático de sinais em novos vídeos ou webcam.

---

#### 📄 `data_preprocessing.py`
- **O que implementa:** Utilitário de pré-processamento de vídeos brutos para extração de frames em disco.
- **Funções presentes:**
  - `extrair_dataset_completo(pasta_videos, pasta_destino_frames)`: Percorre subpastas de classes e aciona a extração para cada arquivo de vídeo.
  - `extract_frames(video_path, output_root_dir, gesture_label, frame_rate)`: Decodifica o vídeo com OpenCV, cria uma subpasta exclusiva por vídeo (`v0000`, `v0001`...) e salva 1 frame a cada `frame_rate` (padrão 5, ~6 fps em vídeos de 30 fps).
- **Papel no projeto:** Estruturar o dataset de vídeos em pastas de frames estáticos para consumo do MediaPipe.

---

#### 📄 `import_from_csv.py`
- **O que implementa:** Módulo para carregamento e reconstrução dos conjuntos de dados salvos em disco (CSV).
- **Funções presentes:**
  - `import_from_csv(filepath, mode)`: Lê o CSV usando pandas. No modo `"rf"`, retorna matriz 2D. No modo `"lstm"`, reconstrói a estrutura de tensor 3D `(amostras, frames, features)` com base nas colunas `sample_idx` e `frame_idx`.
- **Papel no projeto:** Garantir a persistência e reprodutibilidade do treinamento sem a necessidade de reprocessar todas as imagens via MediaPipe.

---

#### 📄 `testar_extracao.py`
- **O que implementa:** Script de teste rápido para validar o funcionamento do módulo `data_preprocessing.py` em uma pasta temporária de teste (`dataset/frames_treino_teste`).
- **Funções presentes:**
  - `extrair_dataset_completo(...)`: Cópia local da função de extração.
- **Papel no projeto:** Script de verificação pontual/desenvolvimento.

---

#### 📄 `testar_mediapipe.py`
- **O que implementa:** Ferramenta de diagnóstico em tempo real do MediaPipe e benchmark de captura de mãos.
- **Funções presentes:**
  - `abrir_fonte_video(fonte)`: Inicializa captura OpenCV (com suporte a `cv2.CAP_DSHOW` no Windows).
  - `desenhar_landmarks(frame, hand_landmarks)`: Desenha visualmente as articulações e conexões da mão no frame.
  - `salvar_relatorio(...)`: Escreve relatório com taxa de detecção e status (`OK` ou `ATENCAO`) em `outputs/mediapipe_mundo_real.txt`.
  - `testar_mediapipe(...)`: Executa o teste contínuo em vídeo/webcam e exibe telemetria de FPS e detecção.
- **Papel no projeto:** Testar a qualidade da iluminação, enquadramento e câmera antes de realizar treinos e testes formais.

---

### 2.2. Documentação e Artigos Acadêmicos

1. **`README.md`**: Guia principal do projeto contendo instruções de instalação, arquitetura, manual de uso do menu do `main.py` e descrição dos artefatos.
2. **`Summary_Dir.md`**: Estrutura detalhada de arquivos, módulos e responsabilidades do projeto.
3. **`docs/`**: Diretório centralizador da documentação e referências técnicas:
   - `CORREÇÕES_ACURÁCIA.md`: Relatório técnico detalhado sobre a resolução da queda de acurácia no LSTM.
   - `RESUMO_CORRECOES.md`: Sumário executivo das correções de bugs, checklist de validação e testes de paridade.
   - `augmentation.md`: Documentação teórica sobre data augmentation em landmarks.
   - `4574 (1).pdf`: Artigo científico do SBrT 2023 ("Interpretação de gestos de Libras usando K-Means e Random Forest" por Caiafa et al.).

---

### 2.3. Diretórios de Dados, Logs e Modelos Binários

1. **`hand_landmarker.task`**: Arquivo binário pré-treinado da API MediaPipe Tasks contendo a rede neural de detecção de landmarks das mãos.
2. **`models/`**: Diretório que armazena os modelos treinados:
   - `sign_model.pkl`: Random Forest serializado.
   - `lstm_sign_model.h5`: Rede Neural LSTM salva no formato Keras HDF5.
   - `knn_sign_model.pkl`: Pacote Pickle contendo o modelo KNN, `StandardScaler` e metadados.
   - `label_encoder.pkl`: Mapeamento de texto para inteiro do scikit-learn.
3. **`logs/`**: Registros de execução e métricas:
   - `resultados_treino.csv`: Histórico dos treinos (data, modelo, acurácia %, nº amostras, augmentation).
   - `matriz_confusao_modelo_1.png`, `matriz_confusao_modelo_2.png`, `matriz_confusao_modelo_3.png`: Gráficos gerados da matriz de confusão.
4. **`outputs/`**: Diretório centralizador de saídas em `.txt`:
   - `gabarito.txt`: Rótulos reais dos vídeos de teste.
   - `mediapipe_mundo_real.txt`: Relatório diagnóstico gerado por `testar_mediapipe.py`.
   - `predicoes_modelo_1.txt`, `predicoes_modelo_2.txt`, `predicoes_modelo_3.txt`: Predições linha a linha para os vídeos de teste.
5. **`dataset/`**:
   - `frames_treino/` e `frames_teste/`: Subpastas com as imagens `.jpg` extraídas dos vídeos.
   - `dataset_completo_rf.csv` e `dataset_completo_lstm.csv`: Datasets exportados em formato tabular.
6. **`videos/`**:
   - `videos/treino/`: Vídeos originais por classe para treinamento.
   - `videos/teste/`: Vídeos para validação final.
7. **`requirements.txt`**: Lista de bibliotecas Python necessárias (`tensorflow`, `scikit-learn`, `numpy`, `mediapipe`, `opencv-python`, `pandas`, `matplotlib`, `tqdm`).

---

## 3. 🛠️ Oportunidades de Melhoria Detectadas

Após auditoria completa do código-fonte, foram identificados os seguintes pontos que podem ser aprimorados em termos de arquitetura, qualidade de código, desempenho e usabilidade:

### 3.1. Eliminação de Duplicação de Código (Refatoração)
- **Normalização e Extração de Landmarks:**
  - As funções `_normalizar_mao` e `_extrair_ambas_maos` estão idênticas em `feature_extraction.py` e redefinidas em `sign_recognition.py` (`extrair_landmarks`).
  - *Como melhorar:* Centralizar a normalização de landmarks em um único módulo utilitário (ex: `utils/geometry.py` ou dentro de `feature_extraction.py`) e importá-la em `sign_recognition.py`.
- **K-Means Temporal:**
  - A função `_aplicar_kmeans_temporal` em `model_training.py` é praticamente idêntica a `_kmeans_temporal_single` em `sign_recognition.py`.
  - *Como melhorar:* Mover `_aplicar_kmeans_temporal` para um utilitário compartilhado ou reutilizá-la diretamente.
- **Extração de Vídeos em Lote:**
  - A função `extrair_dataset_completo` está definida em três lugares: `data_preprocessing.py`, `main.py` e `testar_extracao.py`.
  - *Como melhorar:* Manter a definição exclusivamente em `data_preprocessing.py` e importá-la nos outros scripts.

### 3.2. Robustez na Ordenação Numérica de Frames
- Em `feature_extraction.py`, a função `_listar_frames` tenta ordenar frames usando `int(x.split('_')[1].split('.')[0])`.
- *Problema:* Se o nome do arquivo for diferente do padrão estrito `frame_12.jpg` (ex: `frame_12_crop.jpg` ou `img_12.jpg`), o script lança exceção e cai no `sort()` alfabético padrão, onde `frame_10.jpg` vem antes de `frame_2.jpg`.
- *Como melhorar:* Usar expressões regulares (`re.findall(r'\d+', filename)`) para extrair com segurança o número ordinal do frame.

### 3.3. Otimização no Carregamento do CSV (`import_from_csv.py`)
- No modo LSTM, o código realiza uma busca `df[df['sample_idx'] == sample_id]` em laço `for` para cada amostra do dataset.
- *Problema:* Operação $O(N^2)$ em relação ao número de linhas do DataFrame, tornando o carregamento lento em datasets grandes.
- *Como melhorar:* Utilizar o método `groupby('sample_idx')` do pandas ou reformatar diretamente utilizando reshaping de arrays NumPy 3D (`df[feature_cols].values.reshape(num_samples, num_frames, num_features)`).

### 3.4. Melhoria na Usabilidade da Matriz de Confusão (`main.py`)
- Na opção `[6]` do `main.py`, o sistema exige que o usuário digite manualmente todos os rótulos reais separados por vírgula no terminal.
- *Problema:* Processo propenso a erros de digitação e cansativo.
- *Como melhorar:* Carregar automaticamente o gabarito a partir do arquivo `gabarito.txt` ou inferir os rótulos reais diretamente dos nomes das pastas em `videos/teste/`.

### 3.5. Tratamento de Falhas no Primeiro Frame da Inferência
- Em `feature_extraction.py` e `sign_recognition.py`, se o primeiro frame de um vídeo não detectar nenhuma mão, a variável `ultimo_valido` inicializa como um vetor contendo apenas zeros (`[0.0]*126`).
- *Como melhorar:* Implementar um mecanismo de *back-fill* ou descartar frames iniciais sem detecção até encontrar a primeira mão válida.

---

## 4. 🧹 Peças e Arquivos que Podem Ser Removidos

A auditoria identificou partes do código e arquivos que são redundantes, obsoletos ou desnecessários para a execução principal do projeto:

| Arquivo / Função a Remover | Localização | Razão da Remoção |
| :--- | :--- | :--- |
| **`testar_extracao.py`** | Raiz do projeto | **Script de teste temporário / descartável.** O teste de extração já pode ser realizado pelo menu principal do `main.py` (Opção 5). O arquivo gera apenas sujeira e duplica funções. |
| **`model_training_leaveoneout.py`** | Raiz do projeto | **Código duplicado e orfão.** O `main.py` importa apenas `model_training.py`. Toda a lógica do `model_training_leaveoneout.py` é uma duplicação com LOOCV que nunca é chamada no fluxo do sistema. |
| **Função `extrair_dataset_completo` duplicada em `main.py`** | `main.py` (linhas 43–61) | **Duplicação de código.** `main.py` já importa `extract_frames` de `data_preprocessing.py`. A função inline `extrair_dataset_completo` deve ser removida de `main.py` e importada de `data_preprocessing.py`. |
| **`explicacao-do-knn.txt`** | Raiz do projeto | **Documentação informal redundante.** As informações contidas neste arquivo já foram consolidadas no `README.md` e na presente documentação. |

---

## 5. 💡 Resumo das Recomendações de Ação

1. **Executar a Limpeza:** Excluir os arquivos obsoletos (`testar_extracao.py`, `model_training_leaveoneout.py`, `explicacao-do-knn.txt`).
2. **Refatorar Módulos Compartilhados:** Criar um arquivo utilitário `utils.py` contendo as funções de normalização de landmarks e K-Means temporal para eliminar as duplicações entre `feature_extraction.py`, `model_training.py` e `sign_recognition.py`.
3. **Automatizar Matriz de Confusão:** Atualizar a opção `[6]` do `main.py` para ler automaticamente de `gabarito.txt`.
4. **Otimizar `import_from_csv.py`:** Substituir o loop em `sample_idx` por `reshape` vetorizado do NumPy.

---
*Relatório gerado automaticamente após análise minuciosa do repositório.*
