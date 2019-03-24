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
    xVggFc7     = tf.placeholder(tf.float32, [None, VggFc7Size], 'xVggFc7')
    xTokens     = tf.placeholder(tf.int32, [None, truncated_backprop_length], name='xTokens')
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
    W_vggFc7 = tf.Variable(tf.random_normal([VggFc7Size, hidden_state_sizes]) / tf.sqrt(tf.cast(VggFc7Size, dtype=tf.float32)), name='W_vggFc7')
    b_vggFc7 = tf.Variable(tf.zeros([1, hidden_state_sizes]), name='b_vggFc7')
    initial_state = tf.tanh(tf.matmul(x_VggFc7, W_vggFc7)+b_vggFc7, name='initial_state')
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
    wordEmbeddingMatrix = tf.Variable(tf.random_normal([vocabulary_size, embedding_size], mean=0.0, stddev=1.0), name='wordEmbeddingMatrix')
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
    xEmbed = tf.nn.embedding_lookup(wordEmbeddingMatrix, xTokens)
    inputs = tf.unstack(xEmbed, axis=1, name='inputList')
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
    W_hy = tf.Variable(tf.random_normal([hidden_state_sizes, vocabulary_size]) / tf.sqrt(tf.cast(hidden_state_sizes, dtype=tf.float32)), name='W_hy')
    b_hy = tf.Variable(tf.zeros([1, vocabulary_size]), name='b_hy')
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
        with tf.variable_scope('layer%d'%ind, reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable(name='W', initializer=tf.random_normal(shape=(self.hidden_state_sizes + inputSize,
                        self.hidden_state_sizes), mean=0, stddev=1.0) / np.sqrt(inputSize + self.hidden_state_sizes))
            self.b = tf.get_variable(name='b', initializer=tf.zeros(shape=(1, self.hidden_state_sizes)))

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
        h = tf.concat([input, state_old], axis=1, name='stateConcat')
        state_new = tf.tanh(tf.matmul(h, self.W) + self.b, name='tanhForward')
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
        with tf.variable_scope('layer%d'% (ind), reuse=tf.AUTO_REUSE):
            with tf.variable_scope('update', reuse=tf.AUTO_REUSE):
                self.W_u = tf.get_variable(name='W', initializer=tf.random_normal(shape=(self.hidden_state_sizes + inputSize, self.hidden_state_sizes), mean=0, stddev=1.0) / np.sqrt(inputSize + self.hidden_state_sizes))
                self.b_u = tf.get_variable(name='b', initializer=tf.zeros(shape=(1, self.hidden_state_sizes)))

            with tf.variable_scope('reset', reuse=tf.AUTO_REUSE):
                self.W_r = tf.get_variable(name='W', initializer=tf.random_normal(shape=(self.hidden_state_sizes + inputSize, self.hidden_state_sizes), mean=0, stddev=1.0) / np.sqrt(inputSize + self.hidden_state_sizes))
                self.b_r = tf.get_variable(name='b', initializer=tf.zeros(shape=(1, self.hidden_state_sizes)))

            with tf.variable_scope('candidate', reuse=tf.AUTO_REUSE):
                self.b = tf.get_variable(name='b', initializer=tf.zeros(shape=(1, self.hidden_state_sizes)))
                self.W = tf.get_variable(name='W', initializer=tf.random_normal(shape=(self.hidden_state_sizes + inputSize, self.hidden_state_sizes), mean=0, stddev=1.0) / np.sqrt(self.hidden_state_sizes + inputSize))

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
        h1 = tf.concat([input, state_old], axis=1, name='stateConcat')
        Gu = tf.sigmoid(tf.matmul(h1, self.W_u) + self.b_u, name='UpdateGateOps')
        Gr = tf.sigmoid(tf.matmul(h1, self.W_r) + self.b_r, name='ResetGateOps')

        h2 = tf.concat([input, Gr*state_old], axis=1, name='stateConcat2')
        h_tilde = tf.tanh(tf.matmul(h2, self.W) + self.b, name='h_tilde')
        state_new = Gu*state_old + (1.0-Gu)*h_tilde
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

    cells              = []
    current_state      = initial_states
    logits_series      = []
    predictions_series = []
    predicted_tokens   = []

    #Initialize the rnn cells
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
    x = inputs[0]
    for kk in range(truncated_backprop_length):
        for ii in range(num_layers):
            if ii == 0:
                current_state[ii] = cells[ii].forward(input=x, state_old=current_state[ii])
            else:
                current_state[ii] = cells[ii].forward(input=current_state[ii - 1], state_old=current_state[ii])

            if ii==num_layers-1:
                logits = tf.matmul(current_state[ii], W_hy) + b_hy
                tokens = tf.argmax(logits, axis=1)
                pred   = tf.nn.softmax(logits)
                est_x  = tf.nn.embedding_lookup(wordEmbeddingMatrix, tokens)

                logits_series.append(logits)
                predictions_series.append(pred)
                predicted_tokens.append(tokens)
                if kk < truncated_backprop_length - 1:
                    if is_training == True:
                        x = inputs[kk + 1]
                    elif is_training == False:
                        x = est_x
                #x = tf.where(condition=is_training, x=inputs[kk + 1], y=est_x)

    return logits_series, predictions_series, current_state, predicted_tokens

########################################################################################################################
def loss(yTokens, yWeights, logits_series):
    yTokens_series = tf.unstack(yTokens, axis=1)
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series, yTokens_series)]

    losses = tf.stack(losses, axis=1)
    mean_loss = tf.reduce_mean(losses * yWeights)
    sum_loss  = tf.reduce_sum(losses * yWeights)
    return mean_loss, sum_loss
