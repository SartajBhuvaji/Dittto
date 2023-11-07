#for testing purposes
from tensorflow import keras
import pandas as pd
from tensorflow import keras


#Scenario 1: Needs model
def generate_model(input_shape:int, **kwargs):

    # Default parameter values
    input_shape = int(input_shape)
    if input_shape < 0:
        raise ValueError("Input shape must be greater than 0.")
    try:
        encoder_dense_layers = kwargs.get('encoder_dense_layers', [18,20])
        bottle_neck = kwargs.get('bottle_neck', input_shape // 2)
        decoder_dense_layers = kwargs.get('decoder_dense_layers', [20,18])
        decoder_activation = kwargs.get('decoder_activation', 'sigmoid')    
        summary = kwargs.get('summary', False)

    except Exception as e:
        raise Exception(e)    
    
        # Encoder Model
    encoder_input = keras.Input(shape=(input_shape,), name="encoder")
    x = keras.layers.Flatten()(encoder_input)

    # Encoder Dense Layers
    for units in encoder_dense_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    encoder_output = keras.layers.Dense(bottle_neck, activation="relu")(x)
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")

    # Decoder Model
    decoder_input = keras.Input(shape=(bottle_neck,), name="decoder")
    x = decoder_input

    # Decoder Dense Layers
    for units in decoder_dense_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    decoder_output = keras.layers.Dense(input_shape, activation=decoder_activation)(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # Autoencoder Model
    autoencoder_input = keras.Input(shape=(input_shape,), name="input")
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

    if summary:
        print("Encoder Summary")
        encoder.summary()
        print("Decoder Summary")
        decoder.summary()
        print("Autoencoder Summary")
        autoencoder.summary()

    return autoencoder, encoder, decoder


#Scenario 2: Needs synthetic data

def generate_synthetic_data(model_name: str, original_df: pd.DataFrame, minority_class_column: str = 'class', 
                            minority_class_label: str = '0', decoder_activation: str = 'sigmoid',
                            epochs:int = 100) -> pd.DataFrame:
    
    if original_df.empty:
        raise ValueError("Empty dataframe.")
    
    if epochs < 1:
        raise ValueError("Invalid number of epochs.")
    
    original_df[minority_class_column] = original_df[minority_class_column].astype(str)  
    minority_df = original_df[original_df[minority_class_column] == minority_class_label]

    # Check if minority class label is present in the dataset
    if minority_df.empty:
        raise ValueError("Minority class label not found in the dataset.")
    
    majority_df = original_df[original_df[minority_class_column] != minority_class_label]

    minority_df = minority_df.drop(columns=[minority_class_column])
    input_shape = minority_df.shape[1]

    bottle_neck = 16
    if model_name == 'single_encoder' :
        encoder_dense_layers = [20]
        decoder_dense_layers = [18, 20]

    elif  model_name == 'balanced': 
        encoder_dense_layers = [22, 20]
        decoder_dense_layers = [20, 22]

    elif model_name == 'heavy_decoder':
        encoder_dense_layers = [22,20]
        decoder_dense_layers = [18, 20, 22, 24]

    else:
        raise ValueError("Invalid model name.") 
    pass
   
    try:
        autoencoder, encoder, decoder = generate_model(input_shape, encoder_dense_layers=encoder_dense_layers,
                                                            bottle_neck=bottle_neck, 
                                                            decoder_dense_layers=decoder_dense_layers,
                                                            decoder_activation=decoder_activation)
    except ValueError:
        raise ValueError("Invalid model parameters.")
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss='mse')

    batch_size = 16
    validation_split = 0.25
    verbose = 0

    try:
        autoencoder.fit(minority_df, minority_df, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
        synthetic_df = pd.DataFrame()

        while len(synthetic_df) < len(minority_df):
            synthetic_minority_df = autoencoder.predict(minority_df, verbose=verbose)
            reshaped_data = synthetic_minority_df.reshape(len(minority_df), -1)
            df_generated = pd.DataFrame(reshaped_data, columns = minority_df.columns)

            if minority_class_label.isnumeric():
                df_generated[minority_class_column] = int(minority_class_label)
            else:
                df_generated[minority_class_column] = minority_class_label

            synthetic_df = pd.concat([minority_df, df_generated, majority_df], ignore_index=True)
            synthetic_df = synthetic_df.sample(frac=1).reset_index(drop=True)
    
    except Exception as e:
        raise Exception(e)

    return synthetic_df