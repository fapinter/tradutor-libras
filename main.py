import csv
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.constants import (
    DATASET_TESTE_CSV,
    DATASET_TREINO_CSV,
    ENCODER_PATH,
    FRAMES_TESTE_DIR,
    FRAMES_TREINO_DIR,
    LOGS_DIR,
    LSTM_PATH,
    N_AUMENTOS,
    RESULTADOS_TREINO_CSV,
    VIDEOS_TESTE_DIR,
    VIDEOS_TREINO_DIR,
)

os.makedirs(LOGS_DIR, exist_ok=True)

from data_preprocessing import extrair_dataset_completo
from feature_extraction import extract_features_from_directory, import_from_csv
from model_training import train_lstm


def _salvar_log_treino(acuracia, n_amostras, augmentar,
                       n_aumentos):
    """Persiste a acurácia de cada treino em logs/resultados_treino.csv."""
    log_path = RESULTADOS_TREINO_CSV
    cabecalho = [
        'data_hora', 'modelo', 'acuracia_%',
        'amostras_treino', 'augmentation', 'n_aumentos'
    ]
    novo_arquivo = not os.path.exists(log_path)

    with open(log_path, 'a', newline='',
              encoding='utf-8') as f:
        writer = csv.writer(f)
        if novo_arquivo:
            writer.writerow(cabecalho)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'LSTM',
            f'{acuracia * 100:.2f}',
            n_amostras,
            'sim' if augmentar else 'nao',
            n_aumentos if augmentar else 0,
        ])
    print(f"[LOG] Resultado salvo em '{log_path}'")


def executar_extracao_de_frames():
    print("\n--- EXTRACAO AUTOMATICA DE VIDEOS ---")
    print("\n[Etapa 1/2] Extraindo pasta de TREINAMENTO...")
    extrair_dataset_completo(VIDEOS_TREINO_DIR,
                             FRAMES_TREINO_DIR)
    print(
        "\n[Etapa 2/2] Extraindo pasta de TESTE/VALIDACAO..."
    )
    extrair_dataset_completo(VIDEOS_TESTE_DIR,
                             FRAMES_TESTE_DIR)
    print("\nProcesso de extracao em lote finalizado!")


def _perguntar_augmentation():
    """Pergunta ao usuario se quer augmentation e quantas variacoes."""
    usar = input(
        "\nUsar data augmentation nos landmarks? (s/n) [recomendado com poucos videos]: "
    ).strip().lower()
    if usar == 's':
        try:
            n = int(
                input(
                    f"Quantas variacoes por amostra? [padrao: {N_AUMENTOS}, recomendado: 5-10]: "
                ).strip() or str(N_AUMENTOS))
        except ValueError:
            n = N_AUMENTOS
        print(
            f"Augmentation ativada: {n} variacoes por amostra (~{n + 1}x mais dados)"
        )
        print(
            "IMPORTANTE: a augmentation sera aplicada APENAS nos dados de treino (apos o split)."
        )
        return True, n
    return False, 0


def gerar_dataset_csv():
    """Gera e salva o dataset SEM augmentation (dados originais puros) para o LSTM."""
    print(
        "\n--- GERAR DATASET LSTM (FEATURES ORIGINAIS) ---")
    print(
        f"[1] Treino ({FRAMES_TREINO_DIR} -> {DATASET_TREINO_CSV})"
    )
    print(
        f"[2] Teste ({FRAMES_TESTE_DIR} -> {DATASET_TESTE_CSV})"
    )
    print("[3] Ambos")
    opcao = input(
        "Escolha uma opção (1, 2 ou 3) [padrão: 1]: "
    ).strip() or "1"

    alvos = []
    if opcao == '1':
        alvos.append((FRAMES_TREINO_DIR, DATASET_TREINO_CSV,
                      "Treino"))
    elif opcao == '2':
        alvos.append(
            (FRAMES_TESTE_DIR, DATASET_TESTE_CSV, "Teste"))
    elif opcao == '3':
        alvos.append((FRAMES_TREINO_DIR, DATASET_TREINO_CSV,
                      "Treino"))
        alvos.append(
            (FRAMES_TESTE_DIR, DATASET_TESTE_CSV, "Teste"))
    else:
        print("Opção inválida.")
        return

    for dataset_root, csv_path, rotulo in alvos:
        print(
            f"\n--- Processando conjunto de {rotulo.upper()} ---"
        )
        if not os.path.exists(dataset_root):
            print(
                f"⚠️  Diretório '{dataset_root}' não encontrado."
            )
            print(
                f"Dica: Execute a opção [2] do menu principal para extrair os frames primeiro."
            )
            continue

        if os.path.exists(csv_path):
            print(
                f"\n⚠️  AVISO: O arquivo {csv_path} já existe!"
            )
            confirmacao = input(
                f"Deseja SOBRESCREVER '{csv_path}'? (s/n): "
            ).strip().lower()
            if confirmacao != 's':
                print(f"Extração de {rotulo} cancelada.")
                continue

        print(
            f"\nExtraindo features de '{dataset_root}'...")
        extract_features_from_directory(
            dataset_root,
            mode="lstm",
            export_dataframe=True,
            output_csv_path=csv_path,
        )
        print(
            f"[OK] Dataset de {rotulo} salvo em '{csv_path}'!"
        )


def executar_treinamento():
    """Treinamento exclusivo do modelo LSTM."""
    dataset_root = FRAMES_TREINO_DIR

    print("\n--- INICIANDO TREINAMENTO DO MODELO LSTM ---")
    print("Origem dos dados de treinamento:")
    print(
        "[1] Extrair agora (Processar as imagens na hora, sem salvar)"
    )
    print(
        "[2] Carregar Dataset pré-gerado (Ler o arquivo CSV salvo)"
    )
    origem = input("Escolha (1 ou 2): ").strip()

    if origem == '1':
        print("\nExtraindo features para o modelo LSTM...")
        features, labels = extract_features_from_directory(
            dataset_root,
            mode="lstm",
        )
    elif origem == '2':
        csv_path = DATASET_TREINO_CSV
        if not os.path.exists(csv_path):
            print(
                f"\nERRO: O arquivo {csv_path} não foi encontrado."
            )
            print(
                "Você precisa rodar a opção 'Gerar Dataset' primeiro!"
            )
            return
        print(
            f"\nCarregando dados do arquivo {csv_path}...")
        features, labels = import_from_csv(csv_path,
                                           mode="lstm")
    else:
        print("Opção inválida.")
        return

    augmentar, n_aumentos = _perguntar_augmentation()

    print("\nIniciando Treinamento do Modelo LSTM...")
    if len(features) < 2:
        print(
            "Erro: Dados insuficientes para treino do LSTM."
        )
        return

    acc = train_lstm(features,
                     labels,
                     augmentar=augmentar,
                     n_aumentos=n_aumentos,
                     return_accuracy=True)

    if acc is not None:
        _salvar_log_treino(acc, len(features), augmentar,
                           n_aumentos)


def executar_pipeline_completo():
    """Executa o pipeline completo de Grid Search + K-Fold CV + Treino Final + Avaliação no Teste."""
    from pipeline import (
        avaliar_modelo_teste,
        carregar_dados,
        executar_grid_search_cv,
        treinar_modelo_final,
    )

    print(
        "\n--- EXECUTANDO PIPELINE COMPLETO (GRID SEARCH + K-FOLD + TESTE) ---"
    )
    X_train, y_train, groups = carregar_dados(
        return_groups=True)
    melhor_config, label_encoder = executar_grid_search_cv(
        X_train, y_train, groups=groups)
    model = treinar_modelo_final(X_train, y_train,
                                 melhor_config,
                                 label_encoder)
    avaliar_modelo_teste(model, label_encoder)


def executar_avaliacao_teste():
    """Avalia o modelo LSTM salvo atualmente contra o conjunto de teste independente."""
    import pickle

    import tensorflow as tf

    from pipeline import avaliar_modelo_teste

    print(
        "\n--- AVALIAR MODELO NO CONJUNTO DE TESTE INDEPENDENTE ---"
    )
    LSTM_PATH = LSTM_PATH
    encoder_path = ENCODER_PATH

    if not os.path.exists(LSTM_PATH) or not os.path.exists(
            encoder_path):
        print(
            f"⚠️  Modelo '{LSTM_PATH}' ou encoder '{encoder_path}' não encontrado."
        )
        print(
            "Treine o modelo antes de executar a avaliação de teste."
        )
        return

    print(f"Carregando modelo de '{LSTM_PATH}'...")
    model = tf.keras.models.load_model(LSTM_PATH)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    avaliar_modelo_teste(model, label_encoder)


if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print(
            "   SISTEMA DE RECONHECIMENTO DE LIBRAS (LSTM)")
        print("=" * 50)
        print(
            "[1] Treinar modelo LSTM rápido (Holdout split simples)"
        )
        print(
            "[2] Pipeline Completo (Grid Search + K-Fold CV + Avaliação Teste)"
        )
        print(
            "[3] Avaliar modelo atual no conjunto de Teste")
        print("[4] Extrair frames de vídeos em lote")
        print("[5] Gerar Dataset (Salvar CSV)")
        print("[0] Sair do programa")
        print("=" * 50)
        print(
            "\n DICA: Para testes ao vivo em tempo real com a câmera,"
        )
        print(
            "       execute diretamente o script 'live_testing.py'."
        )
        print("=" * 50)

        escolha = input(
            "\nEscolha uma opção (0 a 5): ").strip()

        if escolha == '1':
            executar_treinamento()

        elif escolha == '2':
            executar_pipeline_completo()

        elif escolha == '3':
            executar_avaliacao_teste()

        elif escolha == '4':
            executar_extracao_de_frames()

        elif escolha == '5':
            gerar_dataset_csv()

        elif escolha == '0':
            print("\nEncerrando. Até logo!")
            break

        else:
            print("\nOpção inválida.")
