#python setup.py sdist
#pip install dist/dittto-0.1.0.tar.gz
from lib import autoencoder
from tensorflow import keras
import pandas as pd

df1 = pd.read_csv('Original.csv')

autoencoder, encoder, decoder =  autoencoder.generate_autoencoder(df1.shape[1], encoder_dense_layers=[5, 3],
                                                                   bottle_neck=2, decoder_dense_layers=[3, 5], decoder_activation='relu')

opt = keras.optimizers.Adam(learning_rate=0.001)
autoencoder.compile(opt, loss="mse")
history = autoencoder.fit(df1, df1, epochs=2, batch_size=16)
