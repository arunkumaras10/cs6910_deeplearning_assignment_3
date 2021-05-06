import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend


class RNN:
    def __init__(self, embedding_size, n_encoder_tokens, n_decoder_tokens, n_encoder_layers,
                 n_decoder_layers, latent_dimension, cell_type,
                 dropout=0.0, beam_size=1):
        self.embedding_size = embedding_size
        self.n_encoder_tokens = n_encoder_tokens
        self.n_decoder_tokens = n_decoder_tokens
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.latent_dimension = latent_dimension
        self.cell_type = cell_type
        self.dropout = dropout
        self.beam_size = beam_size
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.initRNN()

    def initRNN(self):
        backend.clear_session()
        # create training model

        # input layer
        encoder_inputs = keras.Input(shape=(None,), name='encoder_input')
        # word embedding layer
        encoder = None
        encoder_outputs = None
        state_h = None
        state_c = None
        embeden = tf.keras.layers.Embedding(input_dim=self.n_encoder_tokens, output_dim=self.embedding_size, name='encoder_embedding')(encoder_inputs)
        if self.cell_type is not None and self.cell_type.lower() == 'rnn':
            encoder = keras.layers.SimpleRNN(self.latent_dimension, return_state=True, return_sequences=True, name='encoder_hidden_1', dropout=self.dropout)
            encoder_outputs, state_h = encoder(embeden)
        elif self.cell_type is not None and self.cell_type.lower() == 'gru':
            encoder = keras.layers.GRU(self.latent_dimension, return_state=True, return_sequences=True, name='encoder_hidden_1', dropout=self.dropout)
            encoder_outputs, state_h = encoder(embeden)
        else:
            encoder = keras.layers.LSTM(self.latent_dimension, return_state=True, return_sequences=True, name='encoder_hidden_1', dropout=self.dropout)
            encoder_outputs, state_h, state_c = encoder(embeden)

        # 1st layer
        # number of encoder layers
        e_layer = self.n_encoder_layers
        for i in range(2, e_layer + 1):
            # give the output sequences as input to the next layer also the last state is set as initial state of
            # next layer
            layer_name = ('encoder_hidden_%d') % i
            if self.cell_type is not None and self.cell_type.lower() == 'rnn':
                encoder = keras.layers.SimpleRNN(self.latent_dimension, return_state=True, return_sequences=True, name=layer_name, dropout=self.dropout)
                encoder_outputs, state_h = encoder(encoder_outputs, initial_state=[state_h])
            elif self.cell_type is not None and self.cell_type.lower() == 'gru':
                encoder = keras.layers.GRU(self.latent_dimension, return_state=True, return_sequences=True, name=layer_name, dropout=self.dropout)
                encoder_outputs, state_h = encoder(encoder_outputs, initial_state=[state_h])
            else:
                encoder = keras.layers.LSTM(self.latent_dimension, return_state=True, return_sequences=True, name=layer_name, dropout=self.dropout)
                encoder_outputs, state_h, state_c = encoder(encoder_outputs, initial_state=[state_h, state_c])

        encoder_states = None
        # save the last state
        if self.cell_type is not None and (self.cell_type.lower() == 'rnn' or self.cell_type.lower() == 'gru'):
            encoder_states = [state_h]
        else:
            encoder_states = [state_h, state_c]
        decoder_inputs = keras.Input(shape=(None,), name='decoder_input')
        embedde = tf.keras.layers.Embedding(self.n_decoder_tokens, self.embedding_size, name='decoder_embedding')(decoder_inputs)
        # number of decoder layers
        d_layer = self.n_decoder_layers
        decoder = None
        # first layer
        if self.cell_type is not None and self.cell_type.lower() == 'rnn':
            decoder = keras.layers.SimpleRNN(self.latent_dimension, return_sequences=True, return_state=True, name='decoder_hidden_1', dropout=self.dropout)
            # all decoders the initial state is encoder last state of last layer
            decoder_outputs, _ = decoder(embedde, initial_state=encoder_states)
        elif self.cell_type is not None and self.cell_type.lower() == 'gru':
            decoder = keras.layers.GRU(self.latent_dimension, return_sequences=True, return_state=True, name='decoder_hidden_1', dropout=self.dropout)
            # all decoders the initial state is encoder last state of last layer
            decoder_outputs, _ = decoder(embedde, initial_state=encoder_states)
        else:
            decoder = keras.layers.LSTM(self.latent_dimension, return_sequences=True, return_state=True, name='decoder_hidden_1', dropout=self.dropout)
            # all decoders the initial state is encoder last state of last layer
            decoder_outputs, _, _ = decoder(embedde, initial_state=encoder_states)

        for i in range(2, d_layer + 1):
            layer_name = 'decoder_hidden_%d' % i
            if self.cell_type is not None and self.cell_type.lower() == 'rnn':
                decoder = keras.layers.SimpleRNN(self.latent_dimension, return_sequences=True, return_state=True, name=layer_name, dropout=self.dropout)
                decoder_outputs, _ = decoder(decoder_outputs, initial_state=encoder_states)
            elif self.cell_type is not None and self.cell_type.lower() == 'gru':
                decoder = keras.layers.GRU(self.latent_dimension, return_sequences=True, return_state=True, name=layer_name, dropout=self.dropout)
                decoder_outputs, _ = decoder(decoder_outputs, initial_state=encoder_states)
            else:
                decoder = keras.layers.LSTM(self.latent_dimension, return_sequences=True, return_state=True, name=layer_name, dropout=self.dropout)
                decoder_outputs, _, _ = decoder(decoder_outputs, initial_state=encoder_states)
        # add a dense layer
        decoder_dense = keras.layers.Dense(self.n_decoder_tokens, activation="softmax", name='decoder_output')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def fit(self, encoder_input_data, decoder_input_data, decoder_target_data,
            val_encoder_input_data, val_decoder_input_data, val_decoder_target_data,
            batch_size, epochs):
        self.model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([val_encoder_input_data, val_decoder_input_data],
                             val_decoder_target_data)
        )

        # create inference model
        encoder_inputs = self.model.input[0]  # input_1
        if self.cell_type is not None and (self.cell_type.lower() == 'rnn' or self.cell_type.lower() == 'gru'):
            encoder_outputs, state_h_enc = self.model.get_layer(
                'encoder_hidden_' + str(self.n_encoder_layers)).output
            encoder_states = [state_h_enc]
            self.encoder_model = keras.Model(encoder_inputs, encoder_states)

            decoder_inputs = self.model.input[1]  # input_2

            embedde = self.model.get_layer('decoder_embedding')(decoder_inputs)

            # all decoders the initial state is encoder last state of last lay

            decoder_state_input_h = keras.Input(shape=(self.latent_dimension,))
            decoder_states_inputs = [decoder_state_input_h]
            decoder = self.model.get_layer('decoder_hidden_' + str(self.n_decoder_layers))
            decoder_outputs, state_h_dec = decoder(
                embedde, initial_state=decoder_states_inputs
            )
            decoder_states = [state_h_dec]
        else:
            encoder_outputs, state_h_enc, state_c_enc = self.model.get_layer(
                'encoder_hidden_' + str(self.n_encoder_layers)).output
            encoder_states = [state_h_enc, state_c_enc]
            self.encoder_model = keras.Model(encoder_inputs, encoder_states)

            decoder_inputs = self.model.input[1]  # input_2

            embedde = self.model.get_layer('decoder_embedding')(decoder_inputs)

            # all decoders the initial state is encoder last state of last lay

            decoder_state_input_h = keras.Input(shape=(self.latent_dimension,))
            decoder_state_input_c = keras.Input(shape=(self.latent_dimension,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder = self.model.get_layer('decoder_hidden_' + str(self.n_decoder_layers))
            decoder_outputs, state_h_dec, state_c_dec = decoder(
                embedde, initial_state=decoder_states_inputs
            )
            decoder_states = [state_h_dec, state_c_dec]

        decoder_dense = self.model.get_layer('decoder_output')
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

    def decode_sequence(self, input_seq, empty_seq, max_decoder_seq_length, reverse_target_char_index):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = empty_seq

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            if self.cell_type is not None and (self.cell_type.lower() == 'rnn' or self.cell_type.lower() == 'gru'):
                output_tokens, h = self.decoder_model.predict([target_seq] + [states_value])
                # Update states
                states_value = h
            else:
                output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
                # Update states
                states_value = [h, c]

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

        return decoded_sentence
    
    def summary(self):
        self.model.summary()
