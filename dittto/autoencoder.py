from tensorflow import keras
import pandas as pd

def generate_model(input_shape:int, **kwargs):
    """
    Generates an autoencoder model using the given input shape and optional parameters.

    Args:
        input_shape (int): The shape of the input data.
        **kwargs: Optional keyword arguments for configuring the model. Possible arguments are:
            encoder_dense_layers (list): A list of integers representing the number of units in each dense layer of the encoder. Default is [18, 20].
            bottle_neck (int): The number of units in the bottleneck layer. Default is half of the input shape.
            decoder_dense_layers (list): A list of integers representing the number of units in each dense layer of the decoder. Default is [20, 18].
            decoder_activation (str): The activation function to use in the decoder output layer. Default is 'sigmoid'.
            summary (bool): Whether to print the summary of the models. Default is False.

    Returns:
        tuple: A tuple containing the autoencoder, encoder, and decoder models.

    Use Case:
        Use this function to generate an autoencoder model with custom parameters. Function returns models, that can be custon trained.

    Possible Next Steps:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        autoencoder.compile(opt, loss=loss)
        history = autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
        synthetic_minority_df = autoencoder.predict(minority_df, verbose=verbose)

    Example:
        a. autoencoder, encoder, decoder = generate_model(input_shape=10, encoder_dense_layers=[20, 18], bottle_neck=16, decoder_dense_layers=[18, 20], decoder_activation='sigmoid',summary=True)
        b. autoencoder, encoder, decoder = generate_model(input_shape=128)    
    """

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
    
    encoder_input = keras.Input(shape=(input_shape,), name="encoder")
    x = keras.layers.Flatten()(encoder_input)

    for units in encoder_dense_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    encoder_output = keras.layers.Dense(bottle_neck, activation="relu")(x)
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")

    decoder_input = keras.Input(shape=(bottle_neck,), name="decoder")
    x = decoder_input

    for units in decoder_dense_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    decoder_output = keras.layers.Dense(input_shape, activation=decoder_activation)(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

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


def generate_synthetic_data(model_name: str, original_df: pd.DataFrame, minority_class_column: str = 'class', 
                            minority_class_label: str = '0', decoder_activation: str = 'sigmoid',
                            epochs:int = 100):
    """
    Generates synthetic data using an autoencoder model.

    Args:
        model_name (str): Name of the autoencoder model to use. Valid options are 'single_encoder', 'balanced', and 'heavy_decoder'.
        original_df (pd.DataFrame): Original dataset to generate synthetic data from.
        minority_class_column (str, optional): Name of the column containing the minority class label. Defaults to 'class'.
        minority_class_label (str, optional): Label of the minority class. Defaults to '0'.
        decoder_activation (str, optional): Activation function for the decoder layers. Defaults to 'sigmoid'.
        epochs (int, optional): Number of epochs to train the autoencoder model. Defaults to 100.

    Returns:
        synthetic_df (pd.DataFrame): Balanced dataset with synthetic data.
        generated_data (pd.DataFrame): Synthetic data generated by the autoencoder model.
        minority_df (pd.DataFrame): Minority class data from the original dataset.
        majority_df (pd.DataFrame): Majority class data from the original dataset.

    Use Case:
        Use this function to generate synthetic data using an autoencoder model. Function returns a balanced dataset with synthetic data, synthetic data generated by the autoencoder model, minority class data from the original dataset, and majority class data from the original dataset.

    Possible Next Steps:
        synthetic_df.to_csv('synthetic_data.csv', index=False)

    Example:
        a. synthetic_df, generated_data, minority_df, majority_df = generate_synthetic_data(model_name='single_encoder', original_df, minority_class_column='class', minority_class_label='0', decoder_activation='sigmoid', epochs=100)
        b. synthetic_df, generated_data, minority_df, majority_df = generate_synthetic_data(model_name='balanced', original_df, minority_class_column='class', minority_class_label='disease', decoder_activation='sigmoid', epochs=100)
        c. synthetic_df, generated_data, minority_df, majority_df = generate_synthetic_data(model_name='heavy_decoder', original_df, minority_class_column='class', minority_class_label='0', decoder_activation='softmax', epochs=100)
                    
    """
  
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

    # Select model parameters based on model name
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
        autoencoder, _, _ = generate_model(input_shape, encoder_dense_layers=encoder_dense_layers,
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
        generated_data = pd.DataFrame()

        while len(generated_data) <= len(majority_df):
            synthetic_minority_df = autoencoder.predict(minority_df, verbose=verbose)
            reshaped_data = synthetic_minority_df.reshape(len(minority_df), -1)
            df_generated = pd.DataFrame(reshaped_data, columns = minority_df.columns)

            if minority_class_label.isnumeric():
                df_generated[minority_class_column] = int(minority_class_label)
            else:
                df_generated[minority_class_column] = minority_class_label

            generated_data = pd.concat([generated_data, df_generated], ignore_index=True)

        synthetic_df = pd.concat([minority_df, generated_data, majority_df], ignore_index=True)
        synthetic_df = synthetic_df.sample(frac=1).reset_index(drop=True)
    
    except Exception as e:
        raise Exception(e)

    return synthetic_df, generated_data[:len(majority_df)], minority_df, majority_df