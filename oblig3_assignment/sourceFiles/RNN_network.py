import tensorflow as tf
import numpy as np


########################################################################################################################
def getInputPlaceholders(VggFc7Size, truncated_backprop_length):
    """
    The inputs to the image captioning network are the input tokens and the feature vector from the CNN. This function
    should return two placeholders.

    Args:
        VggFc7Size:                Integer with value equal to the size of the VGG16 fc7 layer
        truncated_backprop_length: Integer representing the length of the rnn sequence

    Return:
        xVggFc7: A placeholder "xVggFc7" with shape [batch size, VggFc7Size] and datatype float32.
        xTokens: A placeholder "xTokens" with shape [batch size, truncated_backprop_length] and datatype int32.

    Both placeholders should handle dynamic batch sizes.
    """

    # TODO:
    xVggFc7     = None
    xTokens     = None
    return xVggFc7, xTokens

########################################################################################################################
def getInitialState(x_VggFc7, VggFc7Size, hidden_state_sizes):
    """
    This function shall map the output from the convolutional neural network to the size of "hidden_state_sizes".
    The mapping shall be done using a fully connected layer with tanh activation function. You are not allowed to use
    high level functions e.g. from tf.layers / tf.contrib

     Args:
        x_VggFc7:           A matrix holding the features from the VGG16 network, has shape [batch size, VggFc7Size].
        VggFc7Size:         Integer with value equal to the size of the VGG16 fc7 layer
        hidden_state_sizes: Integer defining the size of the hidden stats within the rnn cells

    Intermediate:
        W_vggFc7: A tf.Variable with shape [VggFc7Size, hidden_state_sizes]. Initialized using variance scaling with
                  zero mean. Name within the tensorflow graph "W_vggFc7"
        b_vggFc7: A tf.Variable with shape [1, hidden_state_sizes]. Initialized to zero. Name within the tensorflow graph
                  "b_vggFc7"

    Returns:
        initial_state: A matrix with shape [batch size, hidden_state_sizes].

    Tips:
        Variance scaling:  Var[W] = 1/n
    """

    # TODO:
    W_vggFc7 =  None
    b_vggFc7 =  None
    initial_state =  None
    return initial_state


########################################################################################################################
def getWordEmbeddingMatrix(vocabulary_size, embedding_size):
    """
    Args:
        vocabulary_size: Integer indicating the number of different words in the vocabulary
        embedding_size:  Integer indicating the size of the embedding (features) of the words.

    Returns:
        wordEmbeddingMatrix: a tf.Variable with shape [vocabulary_size, embedding_size], initialized with zero mean
        and unit standard deviation.
    """

    # TODO:
    wordEmbeddingMatrix =  None
    return wordEmbeddingMatrix


########################################################################################################################
def getInputs(wordEmbeddingMatrix, xTokens):
    """
    Args:
        wordEmbeddingMatrix: Tensor with shape [vocabulary_size, embedding_size].
        xTokens: Tensor with shape [batch_size, truncated_backprop_length] holding the input tokens.

    Returns:
        inputs: List with length truncated_backprop_length. Each element is a tensor with shape [batch_size, embedding_size]

    Tips:
        tf.nn.embedding_lookup()
    """
    # TODO:

    inputs =  None
    return inputs



########################################################################################################################
def getRNNOutputWeights(hidden_state_sizes, vocabulary_size):
    """
    Args:
        vocabulary_size: Integer indicating the number of different words in the vocabulary
        hidden_state_sizes: Integer defining the size of the hidden stats within the rnn cells

    Returns:
        W_hy: A tf.Variable with shape [hidden_state_sizes, vocabulary_size]. Initialized using variance scaling with
              zero mean.
        b_hy: A tf.Variable with shape [1, vocabulary_size]. Initialized to zero.

    Tips:
        Variance scaling:  Var[W] = 1/n
    """
    # TODO:
    W_hy =  None
    b_hy =  None
    return W_hy, b_hy


########################################################################################################################
class RNNcell():
    def __init__(self, hidden_state_sizes, inputSize, ind):
        """
        Args:
            hidden_state_sizes: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn
            ind: Integer indicating the rnn position in a stacked/multilayer rnn.

        Returns:
            self.W: A tf.Variable with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                    variance scaling with zero mean. Name in tensorflow graph: "layer#/W", were #=ind

            self.b: A tf.Variable with shape [1, hidden_state_sizes]. Initialized to zero. Name in tensorflow graph:
                    "layer#/b", were #=ind

        Tips:
            Variance scaling:  Var[W] = 1/n

        Note:
            You are NOT allowed to use high level modules as "tf.contrib.rnn"
        """
        self.hidden_state_sizes = hidden_state_sizes

        # TODO:
        self.W =  None
        self.b =  None

    def forward(self, input, state_old):
        """
        Args:
            input: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        Note:
            You are NOT allowed to use high level modules as "tf.contrib.rnn"
        """

        # TODO:
        state_new =  None
        return state_new


########################################################################################################################
class GRUcell():
    def __init__(self, hidden_state_sizes, inputSize, ind):
        """
        Args:
            hidden_state_sizes: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn
            ind: Integer indicating the rnn position in a stacked/multilayer rnn.

        Returns:
            self.W_u: A tf.Variable with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                    variance scaling with zero mean. Name in tensorflow graph: "layer#/update/W", were #=ind

            self.W_r: A tf.Variable with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                    variance scaling with zero mean. Name in tensorflow graph: "layer#/reset/W", were #=ind

            self.W: A tf.Variable with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                    variance scaling with zero mean. Name in tensorflow graph: "layer#/candidate/W", were #=ind

            self.b_u: A tf.Variable with shape [1, hidden_state_sizes]. Initialized to zero. Name in tensorflow graph:
                    "layer#/update/b", were #=ind

            self.b_r: A tf.Variable with shape [1, hidden_state_sizes]. Initialized to zero. Name in tensorflow graph:
                    "layer#/reset/b", were #=ind

            self.b: A tf.Variable with shape [1, hidden_state_sizes]. Initialized to zero. Name in tensorflow graph:
                    "layer#/candidate/b", were #=ind

        Tips:
            Variance scaling:  Var[W] = 1/n
        Note:
            You are NOT allowed to use high level modules as "tf.contrib.rnn"
        """
        self.hidden_state_sizes = hidden_state_sizes

        # TODO:
        self.W_u =  None
        self.b_u =  None
        self.W_r =  None
        self.b_r =  None
        self.b =  None
        self.W =  None
        return

    def forward(self, input, state_old):
        """
        Args:
            input: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        Note:
            You are NOT allowed to use high level modules as "tf.contrib.rnn"
        """

        # TODO:
        state_new =  None
        return state_new


########################################################################################################################
def buildRNN(networkConfig, inputs, initial_states, wordEmbeddingMatrix, W_hy, b_hy, is_training):
    """
    Args:
        networkConfig:       Dictionary with information of the network structure
        inputs:              A list holding the inputs to the RNN for each time step. The entities have shape
                             [batch_size, embedding_size]
        initial_states:      A list holding the initial state for each layer in the RNN. The entities have shape
                             [batch_size, hidden_state_sizes]
        wordEmbeddingMatrix: A tensor with shape [vocabulary_size, embedding_size].
        W_hy:                The RNN's output weight matrix
        b_hy:                The RNN's output weight bias
        is_training:         A flag indicating test or training mode.

    Returns:
        logits_series:      A list with the logist from the output layer for each time step. The entities have shape
                            [batch_size, vocabulary_size]
        predictions_series: A list with the probabilities for all words for each time step. The entities have shape
                            [batch_size, vocabulary_size]
        current_state:      A list with the values for all the hidden state at the last time step. The list shall start
                            with the hidden state for layer 0 at index 0.
        predicted_tokens:   A list with the predicted tokens for each time step. The entities are an array with length
                            [batch_size,]

    Note:
        You are NOT allowed to use high level modules as "tf.contrib.rnn"
    """

    truncated_backprop_length = networkConfig['truncated_backprop_length']
    hidden_state_sizes        = networkConfig['hidden_state_sizes']
    num_layers                = networkConfig['num_layers']
    cellType                  = networkConfig['cellType']
    embedding_size            = networkConfig['embedding_size']

    #Initialize the rnn cells
    cells              = []
    for ii in range(num_layers):
        if ii==0:
            if cellType=='RNN':
                cell = RNNcell(hidden_state_sizes, embedding_size, ii)
            else:
                cell = GRUcell(hidden_state_sizes, embedding_size, ii)
        else:
            if cellType == 'RNN':
                cell = RNNcell(hidden_state_sizes, hidden_state_sizes, ii)
            else:
                cell = GRUcell(hidden_state_sizes, hidden_state_sizes, ii)
        cells.append(cell)

    # TODO:
    #Build the RNN loop based on looping through the "truncated_backprop_length" and the "num_layers"
    current_state      = initial_states
    logits_series      = []
    predictions_series = []
    predicted_tokens   = []

    return logits_series, predictions_series, current_state, predicted_tokens

########################################################################################################################
def loss(yTokens, yWeights, logits_series):
    yTokens_series = tf.unstack(yTokens, axis=1)
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series, yTokens_series)]

    losses = tf.stack(losses, axis=1)
    mean_loss = tf.reduce_mean(losses * yWeights)
    sum_loss  = tf.reduce_sum(losses * yWeights)
    return mean_loss, sum_loss
