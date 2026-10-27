from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    Conv1D,
    Add,
    Activation,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def create_lstm_model(
    input_shape, num_classes, lstm_units_1=128, 
    lstm_units_2=64, dense_units=64, dropout_rate=0.3, learning_rate=0.001
):
    """
        Constrói e compila o modelo LSTM com os hiperparâmetros fornecidos.
    """
    tf.keras.backend.clear_session()

    model = Sequential([
        # Camada de entrada do Modelo
        Input(shape=input_shape),
        # Primeira camada LSTM, aprende padrões temporais
        # ao longo dos frames
        LSTM(units=lstm_units_1, return_sequences=True),
        # Reduz overfitting, desliga parte
        # das ativações aleatoriamente
        Dropout(dropout_rate),
        # Segunda camada LSTM, recebe padrões temporais
        # da primeira LSTM, gera representação final da
        # sequência completa
        LSTM(units=lstm_units_2, return_sequences=False),
        # Reduz dependência excessiva de características
        # específicas do treino
        Dropout(dropout_rate),
        # Combina características temporais, ReLU
        # permite a representação de relações não linerares
        # entre características extraídas da sequência
        Dense(units=dense_units, activation="relu"),
        # Auxiliar na generalização do modelo
        Dropout(dropout_rate),
        # Camada de Saída
        # Um neurônio por classe
        # Softmax permite a classificação multiclasse
        Dense(units=num_classes, activation="softmax")
    ])
    # Adequado para o treinamento de redes recorrentes
    optimizer = Adam(learning_rate=learning_rate)

    # sparse_categorical_crossentropy: classes como inteiros (0, 1, 2, ...)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def tcn_residual_block(
    x,
    filters,
    kernel_size,
    dilation_rate,
    dropout_rate,
):
    """
    Bloco residual da TCN.

    As convoluções dilatadas permitem analisar padrões temporais
    em diferentes distâncias da sequência sem utilizar recorrência.
    """

    residual = x

    # Primeira convolução temporal dilatada.
    # padding="causal" preserva a ordem temporal:
    # um instante não utiliza informações de frames futuros.
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)

    # Regularização para reduzir overfitting.
    x = Dropout(dropout_rate)(x)

    # Segunda convolução temporal do bloco.
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)

    x = Dropout(dropout_rate)(x)

    # Ajusta a quantidade de canais do residual caso necessário.
    # A entrada possui 126 (mãos) ou 142 (Holistic) features, enquanto a TCN pode utilizar
    # 32 ou 64 filtros.
    if residual.shape[-1] != filters:
        residual = Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
        )(residual)

    # Conexão residual ajuda o treinamento de redes mais profundas.
    x = Add()([x, residual])

    return Activation("relu")(x)


def create_tcn_model(
    input_shape,
    num_classes,
    filters=64,
    dense_units=64,
    dropout_rate=0.3,
    learning_rate=0.001,
):
    """
    Cria a Temporal Convolutional Network para classificação
    das sequências de landmarks do MediaPipe.

    Entrada:
        (frames, features): 126 features (apenas mãos) ou 142 (Holistic)

    Exemplo:
        (15, 126)
        ou
        (15, 142)
    """

    tf.keras.backend.clear_session()

    inputs = Input(shape=input_shape)

    x = inputs

    # Dilatação 1: padrões entre frames próximos.
    x = tcn_residual_block(
        x,
        filters=filters,
        kernel_size=3,
        dilation_rate=1,
        dropout_rate=dropout_rate,
    )

    # Dilatação 2: aumenta o campo temporal observado.
    x = tcn_residual_block(
        x,
        filters=filters,
        kernel_size=3,
        dilation_rate=2,
        dropout_rate=dropout_rate,
    )

    # Dilatação 4: captura padrões de movimento mais longos.
    x = tcn_residual_block(
        x,
        filters=filters,
        kernel_size=3,
        dilation_rate=4,
        dropout_rate=dropout_rate,
    )

    # Resume toda a dimensão temporal em um vetor.
    x = GlobalAveragePooling1D()(x)

    # Combina as características temporais aprendidas.
    x = Dense(
        units=dense_units,
        activation="relu",
    )(x)

    x = Dropout(dropout_rate)(x)

    # Uma saída para cada gesto.
    outputs = Dense(
        units=num_classes,
        activation="softmax",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="tcn_libras",
    )

    optimizer = Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model