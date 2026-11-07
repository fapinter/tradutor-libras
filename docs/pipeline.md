# Pipeline de treinamento e avaliação (`pipeline.py`)

O `pipeline.py` treina, otimiza e avalia os modelos da lista `models` (no final do arquivo). Para cada modelo ele
repete várias execuções; em cada execução escolhe os hiperparâmetros com GridSearch (com Early Stopping nas redes),
retreina o modelo com todo o treino e avalia no conjunto de teste. A parte de cada etapa está nas seções abaixo; o
protocolo de validação e os experimentos que justificam as escolhas estão em `docs/protocolo_validacao.md`, e o
augmentation em `docs/augmentation.md`.

```
CSV de treino e de teste
      ↓
Augmentation (uma vez) + separação da validação
      ↓
Para cada modelo da lista, n_execucoes vezes:
      GridSearch (Leave-One-Group-Out; redes com Early Stopping)  →  melhores hiperparâmetros
      ↓
      Treino final no treino completo (todos os intérpretes)
      ↓
      Avaliação no teste → log, e se for o melhor treino: outputs/, models/ e best/
```

## Configuração da lista de modelos

Cada item de `models` tem quatro campos: `(nome, grade de hiperparâmetros, usar augmentation, número de execuções)`.

```python
('tcn_at:minds_aug', PARAM_GRID_TCN, True, 4)
```

- **Nome**: a parte antes de `:` escolhe a arquitetura (`lstm`, `lstm_at`, `bilstm`, `bilstm_at`, `tcn`, `tcn_at`,
  `rf`, `kmeans` ou `xgboost`; `_at` é a versão com atenção temporal). A parte depois de `:` só entra no nome dos
  arquivos salvos (`tcn_at_minds_aug`).
- **Grade**: os hiperparâmetros testados no GridSearch (`PARAM_GRID_*` em `utils/constants.py`).
- **Usar augmentation**: escolhe entre os conjuntos de treino com e sem augmentation.
- **Execuções**: quantas vezes o processo completo (GridSearch, treino final e avaliação) é repetido.

Redes e modelos clássicos não podem estar na mesma execução do script (o pipeline recusa): o GridSearch das redes usa
7 processos e o dos demais usa todos os núcleos, então eles são treinados em execuções separadas.

## Dados

1. O treino e o teste são lidos dos CSVs (`dataset/treino.csv` e `dataset/teste.csv`), com sequências de
   `SEQUENCE_LENGTH` frames e 142 features.
2. O grupo de cada amostra passa a ser o **intérprete** (e não o vídeo), calculado a partir do vídeo
   (`interprete_do_video`). Validação e folds do GridSearch separam sempre por intérprete, para o modelo ser avaliado
   em pessoas que não viu no treino.
3. O `LabelEncoder` é ajustado nos rótulos de treino e salvo em `models/label_encoder.pkl` ao final.

### Data augmentation

O augmentation é gerado **uma única vez**, por `gerar_amostras_aumentadas`, na cópia do treino completo: cada amostra
original gera `N_AUMENTOS` (10) variações, o espelhamento e 9 combinações aleatórias das demais transformações
(ruído, escala, rotação e os dois dropouts de mão), resultando em 11× mais dados. A geração usa uma seed fixa, então o
conjunto é o mesmo em todas as execuções. A função também devolve o índice da amostra original de cada amostra
gerada, que serve para montar o treino reduzido (abaixo). As transformações estão em `docs/augmentation.md`.

### Os quatro conjuntos de treino

A validação (`INTERPRETES_VALIDACAO`, atualmente os intérpretes 1 e 6) é separada dos dados **originais**, então nunca
tem augmentation. Os quatro conjuntos são sempre criados, e o campo "usar augmentation" de cada modelo escolhe a
versão:

| Conjunto | Amostras | Intérpretes | Usado em |
|---|---|---|---|
| Completo sem augmentation | 2397 | 1–8 | treino final |
| Completo com augmentation | 26367 | 1–8 | treino final |
| Reduzido sem augmentation | 1918 | 2, 3, 4, 5, 7, 8 | GridSearch das redes |
| Reduzido com augmentation | 21098 | 2, 3, 4, 5, 7, 8 | GridSearch das redes |

O reduzido com augmentation é filtrado do completo com augmentation pelo índice de origem das amostras (só as geradas
a partir de intérpretes do treino reduzido), sem gerar um segundo augmentation. Os modelos clássicos (K-Means, Random
Forest e XGBoost) não usam validação, então o GridSearch e o treino final usam o treino completo.

## Múltiplas execuções de um mesmo modelo

Cada modelo roda `n_execucoes` vezes (4 na lista atual) e cada execução é independente: cria o modelo do zero, refaz o
GridSearch, o treino final e a avaliação. O objetivo é reduzir o efeito do acaso de uma única execução, e ao final
fica salva a melhor delas. O que varia entre as execuções depende do modelo:

- **Redes (LSTM, BiLSTM e TCN)**: nenhuma seed é fixada, então os pesos iniciais e a ordem dos lotes mudam a cada
  execução.
- **Random Forest e XGBoost**: a seed é `42 + número da execução` (`random_state`).
- **K-Means + Random Forest**: as seeds são fixas (42), então as execuções repetem o mesmo resultado.

Todas as execuções são registradas em `logs/resultados_treino.csv`. A comparação com as anteriores é descrita em
"Avaliação".

## GridSearch com Early Stopping

`applyGridSearch` roda um `GridSearchCV` com `LeaveOneGroupOut`: cada fold deixa de fora um intérprete (o grupo), treina
nos demais e mede nele. Com `refit=False`, ele só escolhe os hiperparâmetros; o modelo final é treinado depois. A
métrica de escolha é o **F1-Macro** (todas as classes com o mesmo peso). O melhor conjunto de hiperparâmetros é o de
maior F1-Macro médio entre os folds.

| | Redes | K-Means, RF e XGBoost |
|---|---|---|
| Dados do GridSearch | treino reduzido (6 intérpretes) | treino completo (8 intérpretes) |
| Folds | 6 (um por intérprete do treino reduzido) | 8 |
| Early Stopping | sim, na validação | não |
| Processos paralelos | 7 | todos os núcleos |

A grade de cada modelo gera: LSTM e BiLSTM 32 combinações, TCN 8, K-Means 27, Random Forest 16 e XGBoost 18. O total de
treinos do GridSearch é o número de combinações vezes o número de folds.

### Early Stopping nas redes

Nas redes, cada treino do GridSearch recebe como validação os intérpretes de `INTERPRETES_VALIDACAO` (sem
augmentation), que ficam de fora do treino reduzido. A cada época é calculado o F1-Macro na validação
(`val_f1_macro`), e o `EarlyStopping` monitora essa métrica, com paciência `EARLY_STOPPING_PATIENCE` (15 épocas),
restaurando os pesos da melhor época. O Keras não calcula F1, e a `val_loss` não serve: ela atinge o mínimo cedo e sobe
enquanto o F1 ainda melhora.

O fold avalia o modelo (com os pesos restaurados) no intérprete deixado de fora, e a época restaurada de cada fold é
registrada pelo scorer `melhor_epoca_fold`. Essas épocas servem apenas como informação: o treino final não usa Early
Stopping.

## Treino final com o dataset completo

Escolhidos os hiperparâmetros, `treinar_modelo_final` clona o modelo, aplica os melhores parâmetros e o treina no
**treino completo** (os 8 intérpretes, com ou sem augmentation conforme o modelo), incluindo os intérpretes de
validação, que até então só serviam para o Early Stopping.

- **Redes**: treino por `EPOCAS_TREINO_FINAL` épocas fixas (100), sem validação nem Early Stopping. A escolha de 100
  épocas e o motivo de não usar a época do Early Stopping estão em `docs/protocolo_validacao.md`.
- **Modelos clássicos**: um `fit` simples no treino completo.

## Avaliação

`avaliar_modelo_teste` avalia o modelo final no conjunto de teste:

1. Prevê o teste (probabilidades e argmax nas redes) e converte os índices em gestos com o `LabelEncoder`.
2. Calcula a **acurácia** e o **F1-Macro**, e o relatório de classificação por classe.
3. Registra a execução em `logs/resultados_treino.csv` (data, modelo, métricas, amostras de treino, augmentation,
   época final, tempo e hiperparâmetros). Isso acontece sempre, para todas as execuções.
4. Compara com o melhor treino já salvo em `best/<modelo>.txt` (`eh_melhor_treino`): é melhor quem tiver maior
   F1-Macro, com a acurácia como desempate. Sem arquivo em `best/`, a execução é considerada a melhor.

Só quando a execução supera o melhor treino, ela grava (e sobrescreve):

| Arquivo | Conteúdo |
|---|---|
| `outputs/results_<modelo>.txt` | Métricas, F1 médio do GridSearch, melhores épocas dos folds, hiperparâmetros e relatório por classe |
| `outputs/matriz_<modelo>.png` | Matriz de confusão no teste |
| `outputs/acc_<modelo>.png`, `outputs/loss_<modelo>.png` | Curvas do treino final por época (apenas redes) |
| `models/<modelo>.keras` ou `.pkl` | O modelo treinado |
| `best/<modelo>.txt` | F1-Macro e acurácia (em %) do novo melhor treino |

As métricas em `best/` são gravadas depois do modelo, para o arquivo nunca apontar para um modelo que não foi salvo.
Assim, depois das `n_execucoes` execuções, `models/` e `outputs/` ficam com a melhor delas, e uma execução pior não
apaga um resultado melhor que veio de outra execução, nem de uma rodada anterior do script.

O teste (`dataset/teste.csv`) não participa da escolha de hiperparâmetros do pipeline: o GridSearch e o Early Stopping
usam só os dados de treino. O número de épocas do treino final, porém, foi definido olhando a curva do teste (ver os
cuidados em `docs/protocolo_validacao.md`).
