import random
import numpy as np
import RNN

def accuracy(val_encoder_input_data, target_token_index, max_decoder_seq_length, reverse_target_char_index):
    n_correct = 0
    n_total = 0
    for seq_index in range(len(val_encoder_input_data[0:200])):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = val_encoder_input_data[seq_index: seq_index + 1]
        # Generate empty target sequence of length 1.
        empty_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        empty_seq[0, 0] = target_token_index["\t"]
        decoded_sentence = model.decode_sequence(input_seq, empty_seq, max_decoder_seq_length,
                                                 reverse_target_char_index)

        if decoded_sentence.strip() == val_target_texts[seq_index].strip():
            n_correct += 1

        n_total += 1

    return n_correct * 100.0 / n_total
    
#%%

batch_size = 64  # Batch size for training.
epochs = 15  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space. #hidden states hyperparameter
#num_samples = 25000  # Number of samples to train on.
# Path to the data txt file on disk.
train_hindi = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
val_hindi="dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
train_marathi="dakshina_dataset_v1.0/mr/lexicons/mr.translit.sampled.train.tsv"
val_marathi="dakshina_dataset_v1.0/mr/lexicons/mr.translit.sampled.dev.tsv"
#open and save the files to lists
#using hindi and marathi since it is asked devanagiri and both are devanagiri
with open(train_hindi, "r", encoding="utf-8") as f:
    train_hindi_lines = f.read().split("\n")
with open(val_hindi, "r", encoding="utf-8") as f:
    val_hindi_lines = f.read().split("\n")
with open(train_marathi, "r", encoding="utf-8") as f:
    train_marathi_lines = f.read().split("\n")
with open(val_marathi, "r", encoding="utf-8") as f:
    val_marathi_lines = f.read().split("\n")
#popping the last element of all the lists since it is empty character
train_hindi_lines.pop()
train_marathi_lines.pop()
val_hindi_lines.pop()
val_marathi_lines.pop()
#combine the train of hindi and marathi
#uncomment to include marathi also
train_lines=train_hindi_lines#+train_marathi_lines
#combine the validation of hindi and marathi
val_lines=val_hindi_lines#+val_marathi_lines
random.shuffle(train_lines)
print(train_lines[1:2])

#embedding pre processing
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
#go through the train lines and split them into 3 and save input and target
for line in train_lines[: (len(train_lines) - 1)]:
    #because we want english to devanagiri conversion
    target_text, input_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    #append it to the main input texts list
    input_texts.append(input_text)
    #append it to the main target texts list
    target_texts.append(target_text)
    # to find the number of unique characters in both
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
#add the space character to both
input_characters.add(" ")
target_characters.add(" ")
#sort it
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
#find the number
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
#create an 0 array for encoder input size of (input_texts,max_seqlen,tokens)
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length), dtype="float32"
)
#create decoder input
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length), dtype="float32"
)
#create decoder target
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
#for each sample convert it into character encoding i.e. if
#at that position a character is present then encode the index of that character there
#this is done for both encoder and decoder input data for further word embedding
#but target data is one hot encoded.
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t]=input_token_index[char]
    #remaining positions set as empty space
    encoder_input_data[i, t + 1 :]= input_token_index[" "]
    #similarly do for decoder data
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t]= target_token_index[char]
        #check if t >0 since decoder targer data is ahead
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1,target_token_index[char]]=1.0
    #append both the remaining positions of both the datas with empty space
    decoder_input_data[i, t + 1 :]= target_token_index[" "]
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

#embedding validation
#for validation data, almost same
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
        val_encoder_input_data[i, t]= input_token_index[char]
    val_encoder_input_data[i, t + 1 :]= input_token_index[" "]
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        val_decoder_input_data[i, t]= target_token_index[char]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            val_decoder_target_data[i, t - 1,target_token_index[char]]=1.0
    val_decoder_input_data[i, t + 1 :]= target_token_index[" "]
    val_decoder_target_data[i, t:,target_token_index[" "]]=1.0

# create RNN model
model = RNN.RNN(embedding_size=32, n_encoder_tokens=num_encoder_tokens, n_decoder_tokens=num_decoder_tokens,
                n_encoder_layers=1, n_decoder_layers=1, latent_dimension=latent_dim,
                cell_type='LSTM')
model.fit(encoder_input_data, decoder_input_data, decoder_target_data,
          val_encoder_input_data, val_decoder_input_data, val_decoder_target_data,
          batch_size, epochs
          )
model.model.summary()

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

#%%
val_acc = accuracy(val_encoder_input_data[0:100], target_token_index, max_decoder_seq_length, reverse_target_char_index)
print('Validation accuracy: ', val_acc)
