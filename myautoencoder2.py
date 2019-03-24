from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib #added to solve a problem
matplotlib.use('PS') #added to solve a problem
import matplotlib.pyplot as plt 
import argparse
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True' #added to solve a problem

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=2048, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_resultsSkip', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
FLAGS = parser.parse_args()
#cooment
# MAIN SCRIPT
point_num = FLAGS.point_num
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir
    

# Training Parameters
learning_rate = 0.01
num_steps = 10000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2+num_input, num_input])),
}


npc = np.zeros((num_input,num_input), dtype="float32")
for i in range(num_input):
	npc[i,i] = 1.0
idmat=tf.constant(npc, dtype="float32")


biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder

    # Encoder Hidden layer with sigmoid activation #1
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']),biases['encoder_b1']))
#zero= tf.zeros([4,256], dtype="float32")

#tr=tf.equal(layer_1,zero)

    #test = tf.get_default_graph().get_tensor_by_name(layer_1.name)
    #print(test) #Tensor("example:0", shape=(2, 2), dtype=float32)
layer_2a = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
    #if tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']) != 0
layer_2b = tf.nn.relu(X)
layer_2=tf.concat( [ layer_2a, layer_2b ] ,1)
sec_layer=tf.where(tf.greater(layer_2,0)) 


# Building the decoder

    # Decoder Hidden layer with sigmoid activation #1
layer_1_dec = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h1']),biases['decoder_b1']))      
    #if  tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']) != 0
fin_layer=tf.where(tf.greater(layer_1_dec,0))
    
def logout ( data ):
	#print(data)
	LOG_FILE.write(data + '\n')
def logout2 ( data ):
	#print(data)
    #writ=tf.dtypes.as_string(data)
    #tf.io.write_file('neurons.txt', 'ciao')
    LOG_FILE2.write(data + '\n')
#parameters for visualization

lay1=tf.where(tf.greater(layer_1,0))
lay2=tf.where(tf.greater(layer_2,0))
lay3=tf.where(tf.greater(layer_1_dec,0))
   
    
   
LOG_STORAGE_PATH = os.path.join(output_dir, 'logsSkip')

# write logs to the disk
LOG_FILE = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')
LOG_FILE2 = open(os.path.join(LOG_STORAGE_PATH, 'neurons.txt'), 'w')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)    

# Construct model
#encoder_op = encoder(X)
#decoder_op = decoder(encoder_op)

# Prediction
y_pred = layer_1_dec
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

#define parameters to save
lr_op = tf.summary.scalar('learning_rate', learning_rate)

train_variables = tf.trainable_variables()
saver = tf.train.Saver()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
    
    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})

        sumL = tf.Summary(value=[tf.Summary.Value(tag="loss function", simple_value=l),])

        train_writer.add_summary(sumL, i )
        
        logout( 'Training Total Mean_loss: %f' % ( l ) )    
        
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
            LOG_FILE.flush()
        
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g,l1,l2,l3 = sess.run([layer_1_dec,lay1,lay2,lay3], feed_dict={X: batch_x})
        

        #l1=sess.run(lay1, feed_dict={X: batch_x})
        #l2=sess.run(lay2, feed_dict={X: batch_x})
        #l3=sess.run(lay3, feed_dict={X: batch_x})
        LOG_FILE2.write(str(i+1) + ' image:' + '\n')
        for k in l1:
            logout2(str(k))
        for l in l2:
            logout2(str(l))
        for m in l3:
            logout2(str(m))  
        
        #LOG_FILE2.write(str(f))
            #logout2(f[:,:])
        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    #plt.savefig('demo.png', bbox_inches='tight')

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    #plt.savefig('demo2.png', bbox_inches='tight')
LOG_FILE.close()