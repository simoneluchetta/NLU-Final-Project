# NLU-Final-Project
Final project for the NLU course


# Quick guide:
The code is written on a separate Google Colab notebook using PyTorch's framework. It contains both the GRU and LSTM classes. Only the LSTM Layers are used in the model, because they seemed to give better performances over the GRU ones. Nevertheless, slight changes could be applied in order to make GRU Layers work.

Be sure to specify the correct input and output folders in the '''main''' section of the code, in order for the model to firstly train and then output the model_test.pt file.
In my case, I put the input data in a drive folder; the path is clearly visible in the '''main''' section.

## Imports:
Contain the basic imports to make the project work

## Corpus:
Takes in the various files and do some basic computations (create a dictionary, insert elements if not present...).

## GRU Cell:
Implementation of the basic GRU Cell with PyTorch.

## LSTM Cell:
Implementation of the LSTM Cell with PyTorch.

## Model:
Depicts the network's architecture. Deeply inspired as the one taught by Zaremba et al.

## Main:
Where training and evaluation take place. The corpus is loaded, the model is instantiated, all the sentences get batchified through the get_data function; the loss is calculated and then used to obtain the perplexity value.


# List of useful links and references:
What is a RNN, how do we use them and why:

[Illustrated Guide to Recurrent Neural Networks: Understanding the Intuition](https://www.youtube.com/watch?v=LHXXI4-IEns)

[Guide on PyTorch](https://www.tutorialspoint.com/pytorch/pytorch_quick_guide.htm)

[State of the art papers with code](https://paperswithcode.com/sota/language-modelling-on-penn-treebank-word)

[LSTM Cell wiki](https://en.wikipedia.org/wiki/Long_short-term_memory#:~:text=Long%20short-term%20memory%20%28%20LSTM%29%20is%20an%20artificial,sequences%20of%20data%20%28such%20as%20speech%20or%20video%29.)

[How an LSTM works](https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/)

[GRU Cell wiki](https://en.wikipedia.org/wiki/Gated_recurrent_unit)

[How a GRU works](https://d2l.ai/chapter_recurrent-modern/gru.html)

[How LSTM works](http://dprogrammer.org/rnn-lstm-gru)

[GRU math](https://www.data-blogger.com/2017/08/27/gru-implementation-tensorflow/)


How to implement a small portion of the code, and why do we pick an optimizer over another, or a loss metric over another one and so on...:

[PyTorch stuff](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)

[PyTorch stuff](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)

[PyTorch basic op](https://towardsdatascience.com/perplexity-in-language-models-87a196019a94)

[PyTorch basic op](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

[PyTorch basic op](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)

[Pytorch stuff](https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence)

[PyTorch stuff](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html#torch.nn.utils.rnn.pad_sequence)

[PyTorch stuff](https://discuss.pytorch.org/t/packedsequence-for-seq2seq-model/3907)

[Handling stuff in PyTorch](https://github.com/pytorch/pytorch/issues/25310)

[Handling stuff in PyTorch](https://medium.com/@florijan.stamenkovic_99541/rnn-language-modelling-with-pytorch-packed-batching-and-tied-weights-9d8952db35a9)

[Handling stuff in PyTorch](https://discuss.pytorch.org/t/how-to-use-pack-padded-sequence-correctly-how-to-compute-the-loss/38284)

[Handling stuff in PyTorch](https://discuss.pytorch.org/t/calculating-loss-on-sequences-with-variable-lengths/9891/6)

[Data types in PyTorch](https://pytorch.org/docs/stable/tensors.html)

[Optimizer in PyTorch](https://pytorch.org/docs/stable/optim.html)

[Negative Log Likelihood Loss in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)

[Negative Log Likelihood variant in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss)

[Operation on sequences in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html)

[Basic building blocks in PyTorch](https://pytorch.org/docs/stable/nn.html)

[Initialize weights in PyTorch](https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch)

[Pad-pack-other useful things PyTorch](https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec)


How language modelling works:

[Language modelling in general](https://towardsdatascience.com/language-modelling-with-penn-treebank-64786f641f6)

[Language modelling in general](https://www.techopedia.com/definition/20810/modeling-language#:~:text=What%20does%20Modeling%20Language%20mean%3F%20Modeling%20language%20is,is%20part%20of%20and%20similar%20to%20artificial%20language.)

[Language modelling on wiki](https://en.wikipedia.org/wiki/Language_model#Neural_network)

[What a treebank is](https://en.wikipedia.org/wiki/Treebank)

[Perplexity in language modelling](https://towardsdatascience.com/perplexity-in-language-models-87a196019a94)

[Perplexity in language modelling](https://www.quora.com/In-NLP-why-do-we-use-perplexity-instead-of-the-loss)

[Why did I put dropout on embeddings](https://www.reddit.com/r/MachineLearning/comments/60fczp/d_using_dropout_on_embeddings/)

[How to put minibatches inside my model](https://www.kdnuggets.com/2018/06/taming-lstms-variable-sized-mini-batches-pytorch.html)

[Dealing with overfitting](https://stats.stackexchange.com/questions/351741/dealing-with-lstm-overfitting)


GitHub - Programming the network, the cell, and so on:

[How to create an LSTM Cell](https://github.com/piEsposito/pytorch-lstm-by-hand/blob/master/LSTM.ipynb)

[Official PyTorch repo](https://github.com/pytorch/pytorch/blob/3d70ab08ae58b1cf12c09e12729c61d3e850f739/torch/nn/modules/rnn.py)

[Advanced LSTM/GRU](https://github.com/soyoung97/awd-lstm-gru)

[Official Zaremba repo](https://github.com/ahmetumutdurmus/zaremba)

[Another useful repo](https://github.com/salesforce/awd-lstm-lm)

[Official mogrifier repo](https://github.com/RMichaelSwan/MogrifierLSTM/blob/master/MogrifierLSTM.ipynb)

[Mogrifier implementation repo](https://github.com/fawazsammani/mogrifier-lstm-pytorch)

[Zaremba-like repo](https://github.com/hjc18/language_modeling_lstm)

[LSTM on PTB repo](https://github.com/tmatha/lstm)

[LSTM implementation](https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb)

[LSTM / GRU SOTA](https://github.com/sebastianruder/NLP-progress/blob/master/english/language_modeling.md)


Papers:

[Mogrifier paper](https://arxiv.org/pdf/1909.01792.pdf)

[Regularization on RNN](https://arxiv.org/pdf/1409.2329.pdf)


How to use Latex math commands for writing the report file:

[Latex commands](http://www.mathacademy.ws/i-comandi-latex/)
