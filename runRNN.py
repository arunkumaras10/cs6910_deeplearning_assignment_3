import random
import numpy as np
import RNN

batch_size = 64  # Batch size for training.
epochs = 25  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space. #hidden states hyperparameter
# Path to the data txt file on disk.
train_data = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv"
val_data = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv"
# open and save the files to lists
with open(train_data, "r", encoding="utf-8") as f:
    train_lines = f.read().split("\n")
with open(val_data, "r", encoding="utf-8") as f:
    val_lines = f.read().split("\n")
# popping the last element of all the lists since it is empty character
train_lines.pop()
val_lines.pop()
random.shuffle(train_lines)
print(train_lines[0:2])

# embedding pre processing
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
# go through the train lines and split them into 3 and save input and target
for line in train_lines[: (len(train_lines) - 1)]:
    # because we want english to devanagiri conversion
    target_text, input_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    # append it to the main input texts list
    input_texts.append(input_text)
    # append it to the main target texts list
    target_texts.append(target_text)
    # to find the number of unique characters in both
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
# add the space character to both
input_characters.add(" ")
target_characters.add(" ")
# sort it
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
# find the number
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
# find the maximum length of input word and target word
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)
# create an index
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
print((input_token_index))
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
print((target_token_index))
# create an 0 array for encoder input size of (input_texts,max_seqlen,tokens)
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length), dtype="float32"
)
# create decoder input
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length), dtype="float32"
)
# create decoder target
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
# for each sample convert it into character encoding i.e. if
# at that position a character is present then encode the index of that character there
# this is done for both encoder and decoder input data for further word embedding
# but target data is one hot encoded.
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t] = input_token_index[char]
    # remaining positions set as empty space
    encoder_input_data[i, t + 1:] = input_token_index[" "]
    # similarly do for decoder data
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[char]
        # check if t >0 since decoder targer data is ahead
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    # append both the remaining positions of both the datas with empty space
    decoder_input_data[i, t + 1:] = target_token_index[" "]
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

# embedding validation
# for validation data, almost same
val_input_texts = []
val_target_texts = []
for line in val_lines[: (len(val_lines) - 1)]:
    target_text, input_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    val_input_texts.append(input_text)
    val_target_texts.append(target_text)
val_max_encoder_seq_length = max([len(txt) for txt in val_input_texts])
val_max_decoder_seq_length = max([len(txt) for txt in val_target_texts])
val_encoder_input_data = np.zeros(
    (len(val_input_texts), val_max_encoder_seq_length), dtype="float32"
)
val_decoder_input_data = np.zeros(
    (len(val_input_texts), val_max_decoder_seq_length), dtype="float32"
)
val_decoder_target_data = np.zeros(
    (len(val_input_texts), val_max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
for i, (input_text, target_text) in enumerate(zip(val_input_texts, val_target_texts)):
    for t, char in enumerate(input_text):
        val_encoder_input_data[i, t] = input_token_index[char]
    val_encoder_input_data[i, t + 1:] = input_token_index[" "]
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        val_decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            val_decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    val_decoder_input_data[i, t + 1:] = target_token_index[" "]
    val_decoder_target_data[i, t:, target_token_index[" "]] = 1.0

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# create RNN model
model = RNN.RNN(embedding_size=256, n_encoder_tokens=num_encoder_tokens, n_decoder_tokens=num_decoder_tokens,
                n_encoder_layers=2, n_decoder_layers=3, latent_dimension=latent_dim,
                cell_type='lstm', target_token_index=target_token_index, max_decoder_seq_length=max_decoder_seq_length,
                reverse_target_char_index=reverse_target_char_index, dropout=0.2)
model.fit(encoder_input_data, decoder_input_data, decoder_target_data,
          batch_size, epochs=epochs
          )
subset = 100
val_accuracy = model.accuracy(val_encoder_input_data[0:subset], val_target_texts[0:subset]) if subset>0 \
    else model.accuracy(val_encoder_input_data, val_target_texts)
print('Validation accuracy: ', val_accuracy)