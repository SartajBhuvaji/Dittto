import unittest
from autoencoder import generate_model, generate_synthetic_data 
from tensorflow import keras
import pandas as pd

class TestAutoencoder(unittest.TestCase):

    def test_autoencoder_model_type(self):
        input_shape = 10
        autoencoder, encoder, decoder = generate_model(input_shape)
        self.assertEqual(type(autoencoder).__name__, 'Functional')
        self.assertEqual(type(encoder).__name__, 'Functional')
        self.assertEqual(type(decoder).__name__, 'Functional')

    def test_default_autoencoder(self):
        input_shape = 10
        autoencoder, encoder, decoder = generate_model(input_shape)
        self.assertEqual(autoencoder.input_shape, (None, input_shape))
        self.assertEqual(autoencoder.output_shape, (None, input_shape))
        self.assertEqual(encoder.input_shape, (None, input_shape))
        self.assertEqual(encoder.output_shape, (None, 5))
        self.assertEqual(decoder.input_shape, (None, 5))
        self.assertEqual(decoder.output_shape, (None, input_shape))

    def test_gegative_input_shape(self):
        input_shape = -10
        self.assertRaises(ValueError, generate_model, input_shape)

    def test_autoencoder(self):        
        input_shape = 256
        encoder_dense_layers = [128, 64]
        bottle_neck = 32
        decoder_dense_layers = [64, 128]
        decoder_activation = 'sigmoid'
        autoencoder, encoder, decoder = generate_model(input_shape, encoder_dense_layers=encoder_dense_layers, 
                                                            bottle_neck=bottle_neck, decoder_dense_layers=decoder_dense_layers,
                                                            decoder_activation=decoder_activation)
        
        self.assertEqual(autoencoder.input_shape, (None, input_shape))
        self.assertEqual(autoencoder.output_shape, (None, input_shape))
        self.assertEqual(encoder.input_shape, (None, input_shape))
        self.assertEqual(encoder.output_shape, (None, bottle_neck))
        self.assertEqual(decoder.input_shape, (None, bottle_neck))
        self.assertEqual(decoder.output_shape, (None, input_shape))

    def test_custom_activation(self):
        input_shape = 100
        decoder_activation = 'tanh'
        _ , _ , decoder = generate_model(input_shape, decoder_activation=decoder_activation)
        
        self.assertEqual(decoder_activation, decoder.layers[-1].activation.__name__)

    def test_large_input_shape(self):
        input_shape = 1000
        bottle_neck = 256
        _ , encoder, decoder = generate_model(input_shape, bottle_neck=bottle_neck)
        
        self.assertEqual(encoder.output_shape, (None, bottle_neck))
        self.assertEqual(decoder.input_shape, (None, bottle_neck))

    def test_multiple_dense_layers(self):
        input_shape = 100
        encoder_dense_layers = [64, 32, 16]
        decoder_dense_layers = [16, 32, 64]
        _ , encoder, decoder = generate_model(input_shape, encoder_dense_layers=encoder_dense_layers, 
                                                            decoder_dense_layers=decoder_dense_layers)

        self.assertEqual(len(encoder_dense_layers), len(encoder.layers) - 3)  
        self.assertEqual(len(decoder_dense_layers), len(decoder.layers) - 2)

    def test_heavy_encoder_layer(self):
        input_shape = 100
        encoder_dense_layers = [64, 32, 16, 8]
        decoder_dense_layers = [16]
        _ , encoder, decoder = generate_model(input_shape, encoder_dense_layers=encoder_dense_layers, 
                                                            decoder_dense_layers=decoder_dense_layers)

        self.assertEqual(len(encoder_dense_layers), len(encoder.layers) - 3)  
        self.assertEqual(len(decoder_dense_layers), len(decoder.layers) - 2) 
   
    def test_heavy_decoder_layer(self):
        input_shape = 256
        encoder_dense_layers = [64]
        decoder_dense_layers = [16,32,64,128]
        _ , encoder, decoder = generate_model(input_shape, encoder_dense_layers=encoder_dense_layers, 
                                                            decoder_dense_layers=decoder_dense_layers)

        self.assertEqual(len(encoder_dense_layers), len(encoder.layers) - 3)  
        self.assertEqual(len(decoder_dense_layers), len(decoder.layers) - 2)    

    def test_autoencoder_compile(self):
        input_shape = 128
        encoder_dense_layers = [64, 32, 16, 8]
        decoder_dense_layers = [16, 32, 64, 128]
        bottle_neck = 32
        autoencoder, _ , _ = generate_model(input_shape, encoder_dense_layers=encoder_dense_layers, 
                                                            bottle_neck=bottle_neck, decoder_dense_layers=decoder_dense_layers)
        
        opt = keras.optimizers.Adam(learning_rate=0.001)
        autoencoder.compile(opt, loss="mse")

    def test_autoencoder_synthetic_data_genedator(self):
        test_df = pd.DataFrame({'a': [1,2,3,4,5,6,7,8,9,10], 'b': [1,2,3,4,5,6,7,8,9,10], 'class': [0,1,0,1,1,0,1,1,1,0]})
        # test_df = pd.read_csv('test_dataset.csv')
        
        synthetic_df, generated_data, minority_df, majority_df = generate_synthetic_data('single_encoder', test_df, 
                                minority_class_column='class', minority_class_label='0',
                                decoder_activation='sigmoid')
        
        self.assertEqual(len(generated_data), len(majority_df))
        
       
    def test_autoencoder_synthetic_data_generator_no_minority(self):
        test_df = pd.DataFrame({'a': [1,2,3,4,5,6,7,8,9,10], 'b': [1,2,3,4,5,6,7,8,9,10], 'class': [1,1,1,1,1,1,1,1,1,1]})
        with self.assertRaises(ValueError):
            generate_synthetic_data('single_encoder', test_df,
                                    minority_class_column='class', minority_class_label='0',
                                    decoder_activation='sigmoid')
    
    def test_autoencoder_synthetic_data_generator_empty_df(self):
        test_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            generate_synthetic_data('single_encoder', test_df,
                                    minority_class_column='class', minority_class_label='0',
                                    decoder_activation='sigmoid')
            
    def test_autoencoder_synthetic_data_generator_negative_epochs(self):
        test_df = pd.DataFrame({'a': [1,2,3,4,5,6,7,8,9,10], 'b': [1,2,3,4,5,6,7,8,9,10], 'class': [0,1,0,1,1,0,1,1,1,0]})
        with self.assertRaises(ValueError):
            generate_synthetic_data('single_encoder', test_df,
                                    minority_class_column='class', minority_class_label='0',
                                    decoder_activation='sigmoid', epochs=-1)
            
    def test_autoencoder_synthetic_data_generator_incorrect_model_name(self):
        test_df = pd.DataFrame({'a': [1,2,3,4,5,6,7,8,9,10], 'b': [1,2,3,4,5,6,7,8,9,10], 'class': [0,1,0,1,1,0,1,1,1,0]})
        with self.assertRaises(ValueError):
            generate_synthetic_data('single_encoder_', test_df,
                                    minority_class_column='class', minority_class_label='0',
                                    decoder_activation='sigmoid')        


if __name__ == '__main__':
    unittest.main()