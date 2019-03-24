import tensorflow as tf
import numpy as np
from tqdm import trange, tqdm_notebook
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import sys
from time import sleep


from utils import coco
from sourceFiles import RNN_network
from generateVocabulary import loadVocabulary

class trainClass():
    def __init__(self, trainingConfig, networkConfig):

        self.trainingConfig = trainingConfig
        self.networkConfig  = networkConfig

        # training config
        self.trainBatchSize = trainingConfig['trainBatchSize']
        self.valBatchSize   = trainingConfig['valBatchSize']
        self.learning_rate  = trainingConfig['learning_rate']
        self.numberOfEpochs = trainingConfig['numberOfEpochs']

        # Network config
        self.VggFc7Size                = networkConfig['VggFc7Size']
        self.embedding_size            = networkConfig['embedding_size']
        self.vocabulary_size           = networkConfig['vocabulary_size']
        self.truncated_backprop_length = networkConfig['truncated_backprop_length']
        self.hidden_state_sizes        = networkConfig['hidden_state_sizes']
        self.num_layers                = networkConfig['num_layers']
        self.cellType                  = networkConfig['cellType']
        self.is_training               = networkConfig['is_training']

        # Class handling the import of data
        self.myCocoDataClass = coco.CocoVGG16FC7DataClass(self.trainBatchSize, self.valBatchSize,
                                                          self.truncated_backprop_length, self.vocabulary_size)

        self.global_step = tf.Variable(0, trainable=False)
        return

    ####################################################################################################################
    def buildGraph(self):
        # Define placeholders for being able to feed data to the tensorflow graph
        self.xVggFc7, self.xTokens = RNN_network.getInputPlaceholders(self.VggFc7Size, self.truncated_backprop_length)

        # Initialize Vgg fc7 input weights and get initial state
        initial_state       = RNN_network.getInitialState(self.xVggFc7, self.VggFc7Size, self.hidden_state_sizes)
        initial_states      = [initial_state for ii in range(self.num_layers)]
        self.initial_states = tf.stack(initial_states, axis=2)
        initial_statesList  = tf.unstack(self.initial_states, self.num_layers, axis=2)

        # Word embeding input weights
        wordEmbeddingMatrix = RNN_network.getWordEmbeddingMatrix(self.vocabulary_size, self.embedding_size)

        # Get the word embeddings from the input tokens
        inputs = RNN_network.getInputs(wordEmbeddingMatrix, self.xTokens)

        # Output weights
        W_hy, b_hy = RNN_network.getRNNOutputWeights(self.hidden_state_sizes, self.vocabulary_size)

        # Build rnn graph
        self.logits_series, self.predictions_series, self.current_state, self.predicted_tokens = \
                               RNN_network.buildRNN(self.networkConfig, inputs, initial_statesList, wordEmbeddingMatrix , W_hy, b_hy, self.is_training)

        return

    ####################################################################################################################
    def loss(self):
        self.yTokens    = tf.placeholder(tf.int32, [None, self.truncated_backprop_length])
        self.yWeights   = tf.placeholder(tf.float32, [None, self.truncated_backprop_length])
        self.mean_loss, self.sum_loss = RNN_network.loss(self.yTokens, self.yWeights, self.logits_series)
        return

    ####################################################################################################################
    def optimizer(self):
        optimizerType = self.trainingConfig['optimizerType']
        if optimizerType=='Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif optimizerType=='RMSProp':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif optimizerType=='GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise Warning("Unknown optimizer.... select Adam, RMSProp or GradientDescent")
        self.train_op = optimizer.minimize(self.mean_loss, global_step=self.global_step)
        return

    ####################################################################################################################
    def modelEvaluation(self):

        return 0

    ####################################################################################################################
    def tensorboard(self):
        tf.summary.scalar('cross_entropy', self.mean_loss)
        self.summary_op = tf.summary.merge_all()

        self.trainDir = 'TensorboardDir/train/'
        self.trainDir = self.trainDir + \
                        ('/%s' % self.cellType) + \
                        ('/num_layers_%d' % self.num_layers) + \
                        ('/hidden_state_sizes_%d' % self.hidden_state_sizes) + \
                        ('/embedding_size_%d/' % self.embedding_size)

        if not os.path.exists(self.trainDir):
            os.makedirs(self.trainDir)

        for f in glob.glob(self.trainDir+'*'):
            os.remove(f)
        return

    ####################################################################################################################
    def configureSession(self):
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        return

    ####################################################################################################################
    def saveWeights(self):
        self.saverAndRestore_dir = 'model/RNN/'

        self.saverAndRestore_dir = self.saverAndRestore_dir + \
                                   ('/%s' % self.cellType) +\
                                   ('/num_layers_%d' % self.num_layers) +\
                                   ('/hidden_state_sizes_%d' % self.hidden_state_sizes) +\
                                   ('/embedding_size_%d/' % self.embedding_size)

        if not os.path.exists(self.saverAndRestore_dir):
            os.makedirs(self.saverAndRestore_dir)
        for f in glob.glob(self.saverAndRestore_dir+'*'):
            os.remove(f)
        self.saver = tf.train.Saver(max_to_keep=1)
        return

    ####################################################################################################################
    def restoreWeights(self):
        self.saverAndRestore_dir = 'model/RNN/'

        self.saverAndRestore_dir = self.saverAndRestore_dir + \
                                   ('/%s' % self.cellType)  + \
                                   ('/num_layers_%d' % self.num_layers) +\
                                   ('/hidden_state_sizes_%d' % self.hidden_state_sizes) +\
                                   ('/embedding_size_%d/' % self.embedding_size)

        self.saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.latest_checkpoint(self.saverAndRestore_dir)
        if ckpt==None:
            raise ValueError('No stored weights for this network configuration')
        self.saver.restore(self.sess, ckpt)
        return

    ####################################################################################################################
    def trainingLoop(self):
        # Initialize figure
        #plt.figure()
        fig, ax = plt.subplots()
        fig.show()
        plt.ylabel('Loss')
        plt.xlabel('Batch_iterations [epochs]')
        train_line = plt.plot([],[],color='blue', label='Train', marker='.', linestyle="")
        val_line  = plt.plot([], [], color='red', label='Validation', marker='.', linestyle="")
        plt.legend(handles=[train_line[0], val_line[0]])
        ax.set_axisbelow(True)
        ax.grid()

        iterPerEpoch   = self.myCocoDataClass.trainIterPerEpoch
        numberOfEpochs = self.numberOfEpochs

        train_writer = tf.summary.FileWriter(logdir=self.trainDir, graph=self.sess.graph, flush_secs=45)
        loss_vec = []
        Smallest_val_loss = 9999999
        val_loss          = 99999999
        if self.trainingConfig['inNotebook']:
            t = tqdm_notebook(range(numberOfEpochs * iterPerEpoch), desc='Progress', leave=True, file=sys.stdout)
        else:
            t = trange(numberOfEpochs * iterPerEpoch, desc='Train: Loss', leave=True, file=sys.stdout)


        for ii in t:
            loss = 0
            vgg_fc7Batch, xTokensBatch, yTokensBatch, yWeightsBatch, captionsBatch, orig_captionsBatch, imgPathsBatch \
                                                                             = self.myCocoDataClass.train_next_batch()

            for jj in range(xTokensBatch.shape[2]):
                if jj == 0:
                    [_sum_loss, _current_state, train_summary, _] = self.sess.run([self.sum_loss, self.current_state, self.summary_op, self.train_op],
                                                                feed_dict={self.xVggFc7: vgg_fc7Batch,
                                                                           self.xTokens: xTokensBatch[:, :, jj],
                                                                           self.yTokens: yTokensBatch[:, :, jj],
                                                                           self.yWeights: yWeightsBatch[:, :, jj]})
                    loss = loss +_sum_loss
                else:
                    _current_state = np.stack(_current_state, axis=2)
                    [_sum_loss, _current_state, _]  = self.sess.run([self.sum_loss, self.current_state, self.train_op],
                                                                feed_dict={self.initial_states: _current_state,
                                                                           self.xVggFc7: vgg_fc7Batch,
                                                                           self.xTokens: xTokensBatch[:, :, jj],
                                                                           self.yTokens: yTokensBatch[:, :, jj],
                                                                           self.yWeights: yWeightsBatch[:, :, jj]})
                    loss = loss + _sum_loss
            loss_vec.append(loss/np.sum(yWeightsBatch))

            if ii % 15 == 0:
                train_writer.add_summary(summary=train_summary, global_step=ii)
                currentLoss = sum(loss_vec)/len(loss_vec)
                loss_vec = []
                desc = ('Train | Epcohs = %0.2f | loss=%0.5f' % (ii / self.myCocoDataClass.trainIterPerEpoch, currentLoss))
                t.set_description(desc)
                t.update()

                if self.trainingConfig['inNotebook']:
                    ax.scatter(ii / iterPerEpoch, currentLoss, c="b")
                    ax.set_ylim(bottom=0, top=plt.ylim()[1])
                    fig.canvas.draw()
                    sleep(0.1)
                else:
                    plt.scatter(ii/iterPerEpoch, currentLoss, c="b")
                    plt.ylim(0.0, plt.ylim()[1])
                    plt.draw()
                    plt.pause(0.01)

            if (ii % (self.trainingConfig['saveModelEvery#epoch']*iterPerEpoch) == 0) and ii!=0:
                if val_loss<=Smallest_val_loss:
                    Smallest_val_loss = val_loss
                    desc = ('Saving')
                    t.set_description(desc)
                    t.update()
                    sleep(0.1)
                    self.saver.save(self.sess, save_path=self.saverAndRestore_dir, global_step=ii)


            if ii % (int(self.trainingConfig['getLossOnValSetEvery#epoch']*iterPerEpoch)) == 0 and ii!=0:
                desc = ('Testing')
                t.set_description(desc)
                t.update()
                val_loss = self.testLoop()
                if self.trainingConfig['inNotebook']:
                    ax.scatter(ii / iterPerEpoch, val_loss, c="r")
                    fig.canvas.draw()
                    sleep(0.1)
                else:
                    plt.scatter(ii/iterPerEpoch, val_loss, c="r")
                    plt.ylim(0.0, plt.ylim()[1])
                    plt.draw()
                    plt.pause(0.01)
        return

    ####################################################################################################################
    def testLoop(self):
        loss_vec    = []
        for ii in range(self.myCocoDataClass.valIterPerEpoch):
            loss = 0
            vgg_fc7Batch, xTokensBatch, yTokensBatch, yWeightsBatch, captionsBatch, orig_captionsBatch, imgPathsBatch \
                                                                                = self.myCocoDataClass.val_next_batch()

            for jj in range(xTokensBatch.shape[2]):
                if jj == 0:
                    [_sum_loss, _current_state] = self.sess.run([self.sum_loss, self.current_state],
                                                                feed_dict={self.xVggFc7: vgg_fc7Batch,
                                                                           self.xTokens: xTokensBatch[:, :, jj],
                                                                           self.yTokens: yTokensBatch[:, :, jj],
                                                                           self.yWeights: yWeightsBatch[:, :, jj]})
                    loss = loss +_sum_loss
                else:
                    _current_state = np.stack(_current_state, axis=2)
                    [_sum_loss, _current_state]  = self.sess.run([self.sum_loss, self.current_state],
                                                                feed_dict={self.xVggFc7: vgg_fc7Batch,
                                                                           self.xTokens: xTokensBatch[:, :, jj],
                                                                           self.yTokens: yTokensBatch[:, :, jj],
                                                                           self.yWeights: yWeightsBatch[:, :, jj],
                                                                           self.initial_states: _current_state})
                    loss = loss + _sum_loss
            loss_vec.append(loss/np.sum(yWeightsBatch))
        return sum(loss_vec)/len(loss_vec)


    ####################################################################################################################
    def plotImagesAndCaptions(self):

        vgg_fc7Batch, xTokensBatch, yTokensBatch, yWeightsBatch, captionsBatch, orig_captionsBatch, imgPathsBatch = self.myCocoDataClass.getRandomValBatch(self.valBatchSize)

        tokens_est = []
        tokens_in  = np.array(())
        for jj in range(xTokensBatch.shape[2]):
            if jj == 0:
                [_predicted_tokens, _current_state] = self.sess.run([self.predicted_tokens, self.current_state],
                                                            feed_dict={self.xVggFc7: vgg_fc7Batch,
                                                                       self.xTokens: xTokensBatch[:, :, jj]})
                tokens_est = tokens_est + _predicted_tokens
                tokens_in  = np.concatenate((tokens_in, xTokensBatch[:, :, jj][0,:]), axis=0)
            else:
                _current_state = np.stack(_current_state, axis=2)
                xTokensBatch[:, 0, jj] = _predicted_tokens[-1]

                [_predicted_tokens, _current_state] = self.sess.run([self.predicted_tokens, self.current_state],
                                                            feed_dict={self.xVggFc7: vgg_fc7Batch,
                                                                       self.xTokens: xTokensBatch[:, :, jj],
                                                                       self.initial_states: _current_state})

                tokens_est = tokens_est + _predicted_tokens
                tokens_in  = np.concatenate((tokens_in, xTokensBatch[:, :, jj][0, :]), axis=0)

        vocabularyDict = loadVocabulary(self.myCocoDataClass.data_dir)
        TokenToWord    = vocabularyDict['TokenToWord']

        for jj in range(self.valBatchSize):
            sentence = []
            foundEnd = False
            for kk in range(len(tokens_est)):
                word = TokenToWord[tokens_est[kk][jj]]
                if word=='eeee':
                    foundEnd = True
                if foundEnd == False:
                    sentence.append(word)

            #print captions
            print('\n')
            print('Generated caption')
            print(" ".join(sentence))
            print('\n')
            print('Original captions:')
            for kk in range(len(orig_captionsBatch[jj])):
                print(orig_captionsBatch[jj][kk])

            # show image
            imgpath = self.myCocoDataClass.data_dir + 'val2017/'+imgPathsBatch[jj]
            img = mpimg.imread(imgpath)
            fig, ax = plt.subplots()
            # plt.ion()
            ax.imshow(img)
            plt.show()

        return
