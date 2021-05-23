# RNN.py
This python script defines a class RNN which supports lstm, gru and rnn cells. The constructor takes in the required parameters like embedding size, number of encoder tokens, number of decoder tokens, latent dimension, cell type, and dropout. This class has the fit method which can be used to train the model. The decode_sequence method can be used to transliterate a word in the source language into the target language. The accuracy method can be used to compute the accuracy of the given data. The beam search transliterates a word in the source language into the target language using beam search with given beam size.

# runRNN.py
This python script instantiates an object of the class RNN, trains it and computes the validation accuracy.

# rnn_sweep.ipynb
This python notebook contains code to configure and run wandb sweeps.

# rnn_best_model.ipynb
This python notebook configures the best model from the sweep, trains the model, make predictions on test data and computes the test accuracy.

# plotPredictions.py
This python script generates a html page with a table of test set predictions. On running the script, it generates a file called prediction_grid.html.

# attentionModel.ipynb
This python notebook contains code to configure and run wandb sweeps for attention models.

# connectivity_visualization.html
This HTML file has the code to visualize connectivity. This has jquery dependency(jquery.min.js) which is also provided in the repo.

# predictions_vanilla
All the test data predictions made by the best vanilla model for various beam sizes can be found in this folder. For example, predictions_1.csv has the test data predictions made by beam search with beam size 1.

# predictions_attention
All the test data predictions made by the best attention model can be found in this folder.
