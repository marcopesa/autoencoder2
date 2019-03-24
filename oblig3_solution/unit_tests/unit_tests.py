from sourceFiles import RNN_network
import tensorflow as tf
import numpy as np


########################################################################################################################
def checkGetInputPlaceholders(xVggFc7, xTokens):
    errorFlag = 0
    if not np.array_equal(xVggFc7.get_shape().as_list(), [None, 4096]):
        errorFlag=1
    if not np.array_equal(xTokens.get_shape().as_list(), [None, 20]):
        errorFlag = 1

    if not xVggFc7.dtype==tf.float32:
        errorFlag = 1
    if not xTokens.dtype==tf.int32:
        errorFlag = 1

    if errorFlag==1:
        print('"xVggFc7" and/or "xTokens" is not correct')
    else:
        print('Passed')
    return


########################################################################################################################
def checkGetInitialState(initial_state, hidden_state_sizes):
    #check names
    try:
        W_vggFc7 = tf.get_default_graph().get_tensor_by_name('W_vggFc7:0')
    except ValueError:
        raise ValueError('No tensor with name: "W_vggFc7"')

    try:
        b_vggFc7 = tf.get_default_graph().get_tensor_by_name('b_vggFc7:0')
    except ValueError:
        raise ValueError('No tensor with name: "b_vggFc7"')

    try:
        xVggFc7 = tf.get_default_graph().get_tensor_by_name('xVggFc7:0')
    except ValueError:
        raise ValueError('No tensor with name: "W_vggFc7"')

    batch_size, VggFc7Size = xVggFc7.get_shape().as_list()

    #Check sizes
    wrongSizes = False
    if not np.array_equal(W_vggFc7.get_shape().as_list(), [VggFc7Size, hidden_state_sizes]):
        wrongSizes = True
    if not np.array_equal(b_vggFc7.get_shape().as_list(), [1,hidden_state_sizes]):
        wrongSizes = True
    if not np.array_equal(initial_state.get_shape().as_list(), [batch_size, hidden_state_sizes]):
        wrongSizes = True

    if wrongSizes:
        raise ValueError('One or more tensors have wrong shape"')

    # Reference matrix
    initial_state_Ref = tf.layers.dense(inputs=xVggFc7, units=hidden_state_sizes, activation=tf.tanh)
    w_ref = tf.get_default_graph().get_tensor_by_name('dense/kernel:0')
    b_ref = tf.get_default_graph().get_tensor_by_name('dense/bias:0')

    with tf.control_dependencies([initial_state_Ref]):
        w_ref = tf.assign(ref=w_ref, value=W_vggFc7)
        b_ref = tf.assign(ref=b_ref, value=tf.squeeze(b_vggFc7, axis=0))

    #Checking the initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    _W_vggFc7, _b_vggFc7, _initial_state, _initial_state_Ref, _w_ref, _b_ref = sess.run([W_vggFc7, b_vggFc7, initial_state, initial_state_Ref, w_ref, b_ref])
    _initial_state, _initial_state_Ref,  = sess.run([initial_state, initial_state_Ref])

    if not np.array_equal(_initial_state, _initial_state_Ref):
        raise ValueError('Implementation is not correct"')


    if np.abs(np.std(_W_vggFc7)-1/np.sqrt(VggFc7Size))<0.0001 and np.abs(np.mean(_W_vggFc7))<0.0001:
        print('"W_vggFc7" looks to be initialized correctly')
    else:
        raise ValueError('"W_vggFc7" is not initialized correctly')
    if (np.std(_b_vggFc7)==0) and (np.mean(_b_vggFc7)==0):
        print('"b_vggFc7" looks to be initialized correctly')
    else:
        raise ValueError('"b_vggFc7" is not initialized correctly')

    print('Passed')
    return

########################################################################################################################
def checkgetWordEmbeddingMatrix(wordEmbeddingMatrix):

    #Check sizes
    if not np.array_equal(wordEmbeddingMatrix.get_shape().as_list(), [5000, 128]):
        raise ValueError('wordEmbeddingMatrix has wrong shape"')

    if not (str(wordEmbeddingMatrix.dtype)=="<dtype: 'float32_ref'>"):
        raise ValueError('wordEmbeddingMatrix has wrong datatype"')

    #Checking the initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    _wordEmbeddingMatrix = sess.run(wordEmbeddingMatrix)

    if np.abs(np.mean(_wordEmbeddingMatrix))<0.01 and np.abs(np.std(_wordEmbeddingMatrix)-1.0)<0.01:
        print('"wordEmbeddingMatrix" looks to be initialized correctly')
    else:
        raise ValueError('"wordEmbeddingMatrix" is not initialized correctly')

    print('Passed')
    return


########################################################################################################################
def checkGetInputs(inputs, wordEmbeddingMatrix, xTokens):

    sess = tf.Session()
    [_inputs, _wordEmbeddingMatrix, _xTokens] = sess.run([inputs, wordEmbeddingMatrix, xTokens])

    xEmbed     = _wordEmbeddingMatrix[_xTokens]
    inputs_ref = np.stack(xEmbed, axis=1)

    if not np.array_equal(_inputs, inputs_ref):
        raise ValueError('The list "_inputs" is not correct')
    else:
        print('Passed')

    return

########################################################################################################################
def checkGetRNNOutputWeights(W_hy, b_hy):

    #Check sizes
    if not np.array_equal(W_hy.get_shape().as_list(), [128, 1000]):
        raise ValueError('W_hy has wrong shape"')

    #Check sizes
    if not np.array_equal(b_hy.get_shape().as_list(), [1, 1000]):
        raise ValueError('W_hy has wrong shape"')

    #Checking the initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    _W_hy, _b_hy  = sess.run([W_hy, b_hy])

    if np.abs(np.std(_W_hy)-1/np.sqrt(128))<0.001 and np.abs(np.mean(_W_hy))<0.0005:
        print('"W_vggFc7" looks to be initialized correctly')
    else:
        raise ValueError('"W_vggFc7" is not initialized correctly')
    if (np.std(_b_hy)==0) and (np.mean(_b_hy)==0):
        print('"b_vggFc7" looks to be initialized correctly')
    else:
        raise ValueError('"b_vggFc7" is not initialized correctly')

    print('Passed')
    return

########################################################################################################################
def checkRNNcell(rnnCell):
    hidden_state_sizes = 512
    inputSize          = 648
    ind                = 0
    batch_size         = 2

    #dummy variables
    np.random.seed(seed=0)
    np_initial_state = np.random.randn(batch_size, hidden_state_sizes).astype(np.float32)
    np_inputs        = np.random.randn(batch_size, inputSize).astype(np.float32)

    np_W = np.random.randn(hidden_state_sizes+inputSize, hidden_state_sizes).astype(np.float32)
    np_b = np.random.randn(1, hidden_state_sizes).astype(np.float32)

    initial_state = tf.placeholder(shape=(batch_size, hidden_state_sizes), dtype=tf.float32)
    inputs        = tf.placeholder(shape=(batch_size, inputSize), dtype=tf.float32)


    my_hidden_state = rnnCell.forward(input=inputs, state_old=initial_state)


    #Check of dimentions
    if not np.array_equal(my_hidden_state.get_shape().as_list(), [batch_size, hidden_state_sizes]):
        raise ValueError('hidden_state has wrong shape"')

    if not np.array_equal(rnnCell.W.get_shape().as_list(), [hidden_state_sizes+inputSize, hidden_state_sizes]):
        raise ValueError('self.W has wrong shape"')

    if not np.array_equal(rnnCell.b.get_shape().as_list(), [1, hidden_state_sizes]):
        raise ValueError('self.b has wrong shape"')

    cells = tf.nn.rnn_cell.BasicRNNCell(hidden_state_sizes, activation=tf.tanh)
    cell = tf.contrib.rnn.MultiRNNCell([cells])
    tf_hidden_state, _ = tf.contrib.rnn.static_rnn(cell, [inputs], dtype="float32", initial_state=tuple([initial_state]))

    sess =  tf.Session()
    sess.run(tf.global_variables_initializer())

    [_W, _b] = sess.run([rnnCell.W, rnnCell.b])
    #Check initialization
    if np.abs(np.std(_W)-1/np.sqrt(hidden_state_sizes+inputSize))<0.0001 and np.abs(np.mean(_W))<0.0005:
        print('"self.W" looks to be initialized correctly')
    else:
        raise ValueError('"self.W" is not initialized correctly')

    if (np.std(_b)==0) and (np.mean(_b)==0):
        print('"self.b" looks to be initialized correctly')
    else:
        raise ValueError('"self.b" is not initialized correctly')

    #Check implementation
    graph = tf.get_default_graph()
    my_W = graph.get_tensor_by_name("layer%d/W:0" % ind)
    my_b = graph.get_tensor_by_name("layer%d/b:0" % ind)
    tf_W = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/basic_rnn_cell/kernel:0" % ind)
    tf_b = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/basic_rnn_cell/bias:0" % ind)

    feedDictData   = [np_inputs, np_initial_state, np_W, np_b, np_W, np_b[0,:]]
    feedDictInputs = [inputs,    initial_state   , my_W , my_b, tf_W, tf_b]


    [_my_hidden_state, _tf_hidden_state] = sess.run([my_hidden_state, tf_hidden_state],
                                                        feed_dict={i: d for i, d in zip(feedDictInputs, feedDictData)})

    if not np.array_equal(_my_hidden_state, _tf_hidden_state[0]):
        raise ValueError('The vanilla RNN class is NOT implemented correctly')

    print('Passed')

    return


########################################################################################################################
def checkGRUcell(gruCell):
    hidden_state_sizes = 512
    inputSize          = 648
    ind                = 0
    batch_size         = 128

    #dummy variables
    np.random.seed(seed=0)
    np_initial_state = np.random.randn(batch_size, hidden_state_sizes).astype(np.float32)
    np_inputs        = np.random.randn(batch_size, inputSize).astype(np.float32)

    np_Wu = np.random.randn(hidden_state_sizes + inputSize, hidden_state_sizes).astype(np.float32)
    np_Wr = np.random.randn(hidden_state_sizes + inputSize, hidden_state_sizes).astype(np.float32)
    np_W = np.random.randn(hidden_state_sizes + inputSize, hidden_state_sizes).astype(np.float32)

    np_bu = np.random.randn(1, hidden_state_sizes).astype(np.float32)
    np_br = np.random.randn(1, hidden_state_sizes).astype(np.float32)
    np_b = np.random.randn(1, hidden_state_sizes).astype(np.float32)

    initial_state = tf.placeholder(shape=(batch_size, hidden_state_sizes), dtype=tf.float32)
    inputs        = tf.placeholder(shape=(batch_size, inputSize), dtype=tf.float32)


    my_hidden_state = gruCell.forward(input=inputs, state_old=initial_state)


    #Check of dimentions
    if not np.array_equal(my_hidden_state.get_shape().as_list(), [batch_size, hidden_state_sizes]):
        raise ValueError('hidden_state has wrong shape"')

    if not np.array_equal(gruCell.W.get_shape().as_list(), [hidden_state_sizes+inputSize, hidden_state_sizes]):
        raise ValueError('self.W has wrong shape"')

    if not np.array_equal(gruCell.W_u.get_shape().as_list(), [hidden_state_sizes+inputSize, hidden_state_sizes]):
        raise ValueError('self.W_u has wrong shape"')

    if not np.array_equal(gruCell.W_r.get_shape().as_list(), [hidden_state_sizes+inputSize, hidden_state_sizes]):
        raise ValueError('self.W_r has wrong shape"')

    if not np.array_equal(gruCell.b.get_shape().as_list(), [1, hidden_state_sizes]):
        raise ValueError('self.b has wrong shape"')

    if not np.array_equal(gruCell.b_u.get_shape().as_list(), [1, hidden_state_sizes]):
        raise ValueError('self.b_u has wrong shape"')

    if not np.array_equal(gruCell.b_r.get_shape().as_list(), [1, hidden_state_sizes]):
        raise ValueError('self.b_r has wrong shape"')

    cells = tf.contrib.rnn.GRUCell(hidden_state_sizes, activation=tf.tanh)
    cell  = tf.contrib.rnn.MultiRNNCell([cells])
    tf_hidden_state, _ = tf.contrib.rnn.static_rnn(cell, [inputs], dtype="float64", initial_state=tuple([initial_state]))

    sess =  tf.Session()
    sess.run(tf.global_variables_initializer())

    [_W, _b, _W_u, _b_u, _W_r, _b_r] = sess.run([gruCell.W, gruCell.b, gruCell.W_u, gruCell.b_u, gruCell.W_r, gruCell.b_r])
    #Check initialization
    if np.abs(np.std(_W)-1/np.sqrt(hidden_state_sizes+inputSize))<0.0001 and np.abs(np.mean(_W))<0.0005:
        print('"self.W" looks to be initialized correctly')
    else:
        raise ValueError('"self.W" is not initialized correctly')

    if (np.std(_b)==0) and (np.mean(_b)==0):
        print('"self.b" looks to be initialized correctly')
    else:
        raise ValueError('"self.b" is not initialized correctly')

    if np.abs(np.std(_W_u)-1/np.sqrt(hidden_state_sizes+inputSize))<0.0001 and np.abs(np.mean(_W_u))<0.0005:
        print('"self.W_u" looks to be initialized correctly')
    else:
        raise ValueError('"self.W_u" is not initialized correctly')

    if (np.std(_b_u)==0) and (np.mean(_b_u)==0):
        print('"self.b_u" looks to be initialized correctly')
    else:
        raise ValueError('"self.b_u" is not initialized correctly')

    if np.abs(np.std(_W_r)-1/np.sqrt(hidden_state_sizes+inputSize))<0.0001 and np.abs(np.mean(_W_r))<0.0005:
        print('"self.W_u" looks to be initialized correctly')
    else:
        raise ValueError('"self.W_u" is not initialized correctly')

    if (np.std(_b_r)==0) and (np.mean(_b_r)==0):
        print('"self.b_r" looks to be initialized correctly')
    else:
        raise ValueError('"self.b_r" is not initialized correctly')

    #Check implementation
    graph = tf.get_default_graph()
    my_W = graph.get_tensor_by_name("layer%d/candidate/W:0" % ind)
    my_b = graph.get_tensor_by_name("layer%d/candidate/b:0" % ind)
    my_Wu = graph.get_tensor_by_name("layer%d/update/W:0" % ind)
    my_bu = graph.get_tensor_by_name("layer%d/update/b:0" % ind)
    my_Wr = graph.get_tensor_by_name("layer%d/reset/W:0" % ind)
    my_br = graph.get_tensor_by_name("layer%d/reset/b:0" % ind)

    tf_Wg = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/gru_cell/gates/kernel:0" % ind)
    tf_bg = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/gru_cell/gates/bias:0" % ind)
    tf_W = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/gru_cell/candidate/kernel:0" % ind)
    tf_b = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/gru_cell/candidate/bias:0" % ind)

    feedDictData   = [np_inputs, np_initial_state]
    feedDictInputs = [inputs,    initial_state]

    # feed to "my" weights
    feedDictData.append(np_bu)
    feedDictInputs.append(my_bu)
    feedDictData.append(np_br)
    feedDictInputs.append(my_br)
    feedDictData.append(np_b)
    feedDictInputs.append(my_b)
    feedDictData.append(np_Wu)
    feedDictInputs.append(my_Wu)
    feedDictData.append(np_Wr)
    feedDictInputs.append(my_Wr)
    feedDictData.append(np_W)
    feedDictInputs.append(my_W)

    # feed to "tf" weights
    feedDictData.append(np.concatenate((np_Wr, np_Wu), axis=1))
    feedDictInputs.append(tf_Wg)
    feedDictData.append(np.concatenate((np_br, np_bu), axis=1)[0, :])
    feedDictInputs.append(tf_bg)

    feedDictData.append(np_W)
    feedDictInputs.append(tf_W)
    feedDictData.append(np_b[0, :])
    feedDictInputs.append(tf_b)


    [_my_hidden_state, _tf_hidden_state] = sess.run([my_hidden_state, tf_hidden_state],
                                                        feed_dict={i: d for i, d in zip(feedDictInputs, feedDictData)})


    tmpDiff  = np.max(np.abs(_my_hidden_state - _tf_hidden_state[0]))
    tmpLimit = 7e-5
    if tmpDiff >= tmpLimit:
       raise ValueError('The GRU class is NOT implemented correctly')
    print('Passed')

    return



########################################################################################################################
def checkRNNbuild(networkConfig, tests, is_training, batch_size):

    # Note: The test script uses internal tensorflow classes/function.
    # - "tf.contrib.rnn.GRUCell",
    # - "cell = tf.nn.rnn_cell.BasicRNNCell",
    # - "tf.contrib.rnn.MultiRNNCell",
    # - "tf.contrib.rnn.static_rnn",

    embedding_size  = networkConfig['embedding_size']
    vocabulary_size = networkConfig['vocabulary_size']
    truncated_backprop_length = networkConfig['truncated_backprop_length']
    hidden_state_sizes = networkConfig['hidden_state_sizes']
    num_layers         = networkConfig['num_layers']
    cellType           = networkConfig['cellType']

    # Reset tensorflow graph
    tf.reset_default_graph()

    # Set seed
    np.random.seed(seed=0)

    #input  = tf.placeholder(tf.float32, [batch_size, embedding_size, truncated_backprop_length])
    tokens        = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
    wordEmbedding = tf.placeholder(tf.float32, [vocabulary_size, embedding_size])
    xEmbed        = tf.nn.embedding_lookup(wordEmbedding, tokens)

    inputs   = tf.unstack(xEmbed, truncated_backprop_length, axis=1, name='inputList')

    # Create initial state(s)
    initial_state  = tf.placeholder(tf.float32, [batch_size, hidden_state_sizes])
    initial_states = [initial_state for ii in range(num_layers)]

    # Create output weights
    W_hy = tf.placeholder(tf.float32, [hidden_state_sizes, vocabulary_size])
    b_hy = tf.placeholder(tf.float32, [1, vocabulary_size])

    # --------------------------------------------- tensorflow rnn build------------------------------------------------
    cells = []
    for ll in range(num_layers):
        if cellType=='GRU':
            cell = tf.contrib.rnn.GRUCell(hidden_state_sizes, activation=tf.tanh)
        elif cellType=='RNN':
            cell = tf.nn.rnn_cell.BasicRNNCell(hidden_state_sizes, activation=tf.tanh)
        else:
            raise ValueError('Illegal rnn cell type. Select GRU or RNN')
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    tf_states_series, tf_current_state = tf.contrib.rnn.static_rnn(cell, inputs, dtype="float32", initial_state=tuple(initial_states))
    tf_logits_series      = [tf.matmul(state, W_hy) + b_hy for state in tf_states_series]
    tf_predictions_series = [tf.nn.softmax(logits) for logits in tf_logits_series]

    # ----------------------------------------- Your rnn build----------------------------------------------------------
    my_logits_series, my_predictions_series, my_current_states, my_predicted_tokens = \
                                           RNN_network.buildRNN(networkConfig, inputs, initial_states, wordEmbedding, W_hy, b_hy, is_training)

    # -------- Check ---------------------------------------------------------------------------------------------------
    # Init session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    tf.global_variables()
    # --------------------------------------- Generate numpy arrays ----------------------------------------------------
    np_initial_state = np.random.randn(batch_size, hidden_state_sizes).astype(np.float32)
    np_tokens        = np.random.randint(low=0, high=vocabulary_size-1, size=(batch_size, truncated_backprop_length)).astype(dtype=np.int32)

    np_W_hy = np.random.randn(hidden_state_sizes, vocabulary_size).astype(np.float32)
    np_b_hy = np.random.randn(1, vocabulary_size).astype(np.float32)

    np_wordEmbedding = np.random.randn(vocabulary_size, embedding_size).astype(np.float32)

    feedDictData   = [np_tokens, np_initial_state, np_W_hy, np_b_hy, np_wordEmbedding]
    feedDictInputs = [tokens,    initial_state,       W_hy,    b_hy,    wordEmbedding]
    graph = tf.get_default_graph()
    if cellType=='RNN':
        for ii in range(num_layers):
            np_b = np.random.randn(1, hidden_state_sizes).astype(np.float32)
            if ii==0:
                np_W = np.random.randn(hidden_state_sizes + embedding_size, hidden_state_sizes).astype(np.float32)/np.sqrt(hidden_state_sizes + embedding_size)
            else:
                np_W = np.random.randn(hidden_state_sizes+hidden_state_sizes, hidden_state_sizes).astype(np.float32)/np.sqrt(hidden_state_sizes+hidden_state_sizes)

            # Get graph handles
            my_W = graph.get_tensor_by_name("layer%d/W:0" % ii)
            my_b = graph.get_tensor_by_name("layer%d/b:0" % ii)
            tf_W = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/basic_rnn_cell/kernel:0" % ii)
            tf_b = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/basic_rnn_cell/bias:0" % ii)

            #feed to "my" weights
            feedDictData.append(np_W)
            feedDictInputs.append(my_W)

            feedDictData.append(np_b)
            feedDictInputs.append(my_b)

            # feed to "tf" weights
            feedDictData.append(np_W)
            feedDictInputs.append(tf_W)

            feedDictData.append(np_b[0,:])
            feedDictInputs.append(tf_b)
    elif cellType == 'GRU':
        for ii in range(num_layers):
            np_bu = np.random.randn(1, hidden_state_sizes).astype(np.float32)
            np_br = np.random.randn(1, hidden_state_sizes).astype(np.float32)
            np_b = np.random.randn(1, hidden_state_sizes).astype(np.float32)
            if ii==0:
                np_Wu = np.random.randn(hidden_state_sizes + embedding_size, hidden_state_sizes).astype(np.float32) / np.sqrt(hidden_state_sizes + embedding_size)
                np_Wr = np.random.randn(hidden_state_sizes + embedding_size, hidden_state_sizes).astype(np.float32) / np.sqrt(hidden_state_sizes + embedding_size)
                np_W  = np.random.randn(hidden_state_sizes + embedding_size, hidden_state_sizes).astype(np.float32) / np.sqrt(hidden_state_sizes + embedding_size)
            else:
                np_Wu = np.random.randn(hidden_state_sizes + hidden_state_sizes, hidden_state_sizes).astype(np.float32) / np.sqrt(hidden_state_sizes + embedding_size)
                np_Wr = np.random.randn(hidden_state_sizes + hidden_state_sizes, hidden_state_sizes).astype(np.float32) / np.sqrt(hidden_state_sizes + embedding_size)
                np_W  = np.random.randn(hidden_state_sizes + hidden_state_sizes, hidden_state_sizes).astype(np.float32) / np.sqrt(hidden_state_sizes + embedding_size)

            #Get graph handles
            my_W  = graph.get_tensor_by_name("layer%d/candidate/W:0" % ii)
            my_b  = graph.get_tensor_by_name("layer%d/candidate/b:0" % ii)
            my_Wu = graph.get_tensor_by_name("layer%d/update/W:0" % ii)
            my_bu = graph.get_tensor_by_name("layer%d/update/b:0" % ii)
            my_Wr = graph.get_tensor_by_name("layer%d/reset/W:0" % ii)
            my_br = graph.get_tensor_by_name("layer%d/reset/b:0" % ii)

            tf_Wg = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/gru_cell/gates/kernel:0" % ii)
            tf_bg = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/gru_cell/gates/bias:0" % ii)
            tf_W  = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/gru_cell/candidate/kernel:0" % ii)
            tf_b  = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_%d/gru_cell/candidate/bias:0" % ii)


            #feed to "my" weights
            feedDictData.append(np_bu)
            feedDictInputs.append(my_bu)
            feedDictData.append(np_br)
            feedDictInputs.append(my_br)
            feedDictData.append(np_b)
            feedDictInputs.append(my_b)
            feedDictData.append(np_Wu)
            feedDictInputs.append(my_Wu)
            feedDictData.append(np_Wr)
            feedDictInputs.append(my_Wr)
            feedDictData.append(np_W)
            feedDictInputs.append(my_W)

            # feed to "tf" weights
            feedDictData.append(np.concatenate((np_Wr, np_Wu), axis=1))
            feedDictInputs.append(tf_Wg)
            feedDictData.append(np.concatenate((np_br, np_bu), axis=1)[0,:])
            feedDictInputs.append(tf_bg)
            feedDictData.append(np_W)
            feedDictInputs.append(tf_W)
            feedDictData.append(np_b[0,:])
            feedDictInputs.append(tf_b)

            #tf.global_variables()

    #-------------------------------------------------------------------------------------------------------------------
    [_tf_logits_series, _tf_current_state, _my_logits_series, _my_current_state, _tf_predictions_series, _my_predictions_series] = \
                                        sess.run([tf_logits_series, tf_current_state, my_logits_series, my_current_states, tf_predictions_series, my_predictions_series],
                                                        feed_dict={i: d for i, d in zip(feedDictInputs, feedDictData)})

    if 'check_current_state' in tests:
        print('\nCheck of the hidden states at the last time step')

        if len(_tf_current_state)==len(_my_current_state):
            print('The "current_state" has correctly %d layers' % len(_my_current_state))

        for ii in range(len(_tf_current_state)):
            tmpDiff = np.max(np.abs(_tf_current_state[ii]-_my_current_state[ii]))
            tmpLimit = 3e-6
            if tmpDiff <= tmpLimit:
            # if np.array_equal(_tf_current_state[ii], _my_current_state[ii]):
                print('"current_state" is correct at layer %d'%ii)
            else:
                print('"current_state" is incorrect at layer %d' % ii)


    if 'check_logits_series' in tests:
        print('\nChecking the output logits for every time step. Total of %d steps.' % truncated_backprop_length)
        for ii in range(truncated_backprop_length):
            tmpDiff  = np.max(np.abs(_tf_logits_series[ii]-_my_logits_series[ii]))
            tmpLimit = 1e-4
            if tmpDiff <= tmpLimit:
            # if np.array_equal(_tf_logits_series[ii], _my_logits_series[ii]):
                print('"logits_series[%d]" is correct' % ii)
            else:
                print('"logits_series[%d]" is incorrect' % ii)

    if 'predictions_series' in tests:
        print('\nChecking the output predictions for every time step. Total of %d steps.' % truncated_backprop_length)
        for ii in range(truncated_backprop_length):
            tmpDiff  = np.max(np.abs(_tf_predictions_series[ii]-_my_predictions_series[ii]))
            tmpLimit = 3e-5
            if tmpDiff <= tmpLimit:
            # if np.array_equal(_tf_predictions_series[ii], _my_predictions_series[ii]):
                print('"predictions_series[%d]" is correct' % ii)
            else:
                print('"predictions_series[%d]" is incorrect' % ii)

#-----------------------------------------------------------------------------------------------------------------------
    # calculate prediction
    #feedDictData[5]   = False  #change to is_training =False
    [_my_predicted_tokens] = sess.run([my_predicted_tokens], feed_dict={i: d for i, d in zip(feedDictInputs, feedDictData)})


    # get referance tf_predicted_tokens. "Updating the input tokens one by one in a loop"
    _tf_predicted_tokens = []
    for kk in range(truncated_backprop_length):
        [_tf_predictions_series] = sess.run([tf_predictions_series], feed_dict={i: d for i, d in zip(feedDictInputs, feedDictData)})
        inds = np.argmax(_tf_predictions_series[kk], axis=1)
        _tf_predicted_tokens.append(inds)
        if kk<truncated_backprop_length-1:
            feedDictData[0][:, kk + 1] = inds

    if 'predicted_tokens' in tests:
        print('\nChecking the predicted_tokens for every time step. Total of %d steps.' % truncated_backprop_length)
        for ii in range(truncated_backprop_length):
            if np.array_equal(_tf_predicted_tokens[ii], _my_predicted_tokens[ii]):
                print('"predicted_tokens[%d]" is correct' % ii)
            else:
                print('"predicted_tokens[%d]" is incorrect' % ii)

def getConfig(cellType):

    hidden_state_sizes = 512
    embedding_size     = 648
    batch_size         = 128

    vocabulary_size    = 64
    truncated_backprop_length = 6
    num_layers = 3

    networkConfig = {}
    networkConfig['embedding_size']  = embedding_size
    networkConfig['vocabulary_size'] = vocabulary_size
    networkConfig['truncated_backprop_length'] = truncated_backprop_length
    networkConfig['hidden_state_sizes'] = hidden_state_sizes
    networkConfig['num_layers']  = num_layers
    networkConfig['cellType']    = cellType

    initial_state = tf.convert_to_tensor(np.random.randn(batch_size, hidden_state_sizes).astype(np.float32))
    input         = tf.convert_to_tensor(np.random.randn(batch_size, embedding_size).astype(np.float32))
    wordEmbedding = tf.convert_to_tensor(np.random.randn(vocabulary_size, embedding_size).astype(np.float32))
    W_hy          = tf.convert_to_tensor(np.random.randn(hidden_state_sizes, vocabulary_size).astype(np.float32))
    b_hy          = tf.convert_to_tensor(np.random.randn(1, vocabulary_size).astype(np.float32))

    initial_states = [initial_state for ii in range(num_layers)]
    inputs         = [input for ii in range(truncated_backprop_length)]

    return networkConfig, inputs, initial_states, wordEmbedding, W_hy, b_hy, batch_size

########################################################################################################################
if __name__ == "__main__":

    is_training    = True
    cellType       = 'GRU'   # of RNN or GRU
    tests          = ['check_current_state', 'check_logits_series', 'predictions_series', 'predicted_tokens']

    # num_layers          = 3
    # hidden_state_sizes  = 6
    # embedding_size      = 8
    # vocabulary_size     = 24
    # truncated_backprop_length = 6
    #
    # networkConfig = {}
    # networkConfig['embedding_size']  = embedding_size
    # networkConfig['vocabulary_size'] = vocabulary_size
    # networkConfig['truncated_backprop_length'] = truncated_backprop_length
    # networkConfig['hidden_state_sizes'] = hidden_state_sizes
    # networkConfig['num_layers']         = num_layers
    # networkConfig['cellType']           = cellType

    # Config
    cellType = 'GRU'  # of RNN or GRU
    is_training = True  # True/False

    networkConfig, inputs, initial_states, wordEmbedding, W_hy, b_hy, batch_size = getConfig(cellType)

    checkRNNbuild(networkConfig, tests, is_training, batch_size)








