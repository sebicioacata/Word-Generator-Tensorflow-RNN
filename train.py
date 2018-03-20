# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import processing_utils as proc
import model
import os
import math
import time
import hyper_params as hyper

tf.reset_default_graph()

source_directory = "data"

# Hyperparameters
batch_size = hyper.BATCH_SIZE
seq_len = hyper.SEQ_LEN
n_layers = hyper.N_LAYERS
inner_size = hyper.INNER_SIZE
keep_dropout = hyper.KEEP_DROPOUT
learning_rate = hyper.LEARNING_RATE

encoded_text = proc.read_data_files(source_directory)    # The raw data
epoch_size = len(encoded_text) // (batch_size * seq_len)

alpha_size = proc.alpha_size    # Alphabet size (98 - ASCII printable, tab and new line)


lr = tf.placeholder(tf.float32, name='lr')    # learning rate
p_keep = tf.placeholder(tf.float32, name='p_keep')    # dropout parameter
batch_size_ = tf.placeholder(tf.int32, name='batch_size_')

# Inputs
X = tf.placeholder(tf.uint8, [None, None], name = 'X')    # [batch_size, seq_len]
X_oh = tf.one_hot(X, alpha_size)    # [batch_size, seq_len, alpha_size]

# Expected outputs
Y_ = tf.placeholder(tf.uint8, [None, None], name = 'Y_')    # same length shifted by 1
Y_oh_ = tf.one_hot(Y_, alpha_size)

# Initial state
H_init = tf.placeholder(tf.float32, [None, n_layers*inner_size], name = "H_init")    # [batch_size, n_layers*inner_size]

# The predicted outputs and final state
Y_pred, H_final = model.create_model(X_oh, H_init, n_layers, inner_size, p_keep)

H_final = tf.identity(H_final, name='H_final')     # just to give it a name

# Softmax layer implementation:

# Flatten the first two dimension of the output [ batch_size, seq_len, alpha_size ] => [ batch_size x seq_len, alpha_size ]
# then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.

Y_flat = tf.reshape(Y_pred, [-1, inner_size])    # [ batch_size x seq_len, inner_size ]
Y_logits = layers.linear(Y_flat, alpha_size)    # [ batch_size x seq_len, alpha_size ]
Y_flat_ = tf.reshape(Y_oh_, [-1, alpha_size])    # [ batch_size x seq_len, alpha_size ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_flat_)  # [ batch_size x seq_len ]
loss = tf.reshape(loss, [batch_size_, -1])    # [ batch_size, seq_len ]
Yo = tf.nn.softmax(Y_logits, name='Yo')    # [ batch_size x seq_len, alpha_size ]
Y = tf.argmax(Yo, 1)    # [ batch_size x seq_len ]
Y = tf.reshape(Y_, [batch_size, -1], name="Y")    # [ batch_size, seq_len ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Init for saving models. They will be saved into a directory named 'checkpoints'.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

timestamp = str(math.trunc(time.time()))

nb_batches = 60
total_batches_size = nb_batches * batch_size * seq_len

progress = proc.Progress(nb_batches, size=111+2, msg="Training the next " +  str(nb_batches) + " batches out of " + str(epoch_size))


# Session init
init_state = np.zeros([batch_size, inner_size * n_layers])    # Initial zero input state

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

step = 0    # Training loop step
prev_epoch = 0
epoch = 0
nb_epochs = 11    # Set the number of epochs

# Entering the training loop
for x, y_, epoch in proc.rnn_batch_sequencer(encoded_text, batch_size, seq_len, nb_epochs):

    # Creating the feed dictionary for the current batch
    feed_dict = {X: x, Y_: y_, H_init: init_state, lr: learning_rate, p_keep: keep_dropout, batch_size_: batch_size}
    _, y, out_state = sess.run([train_step, Y, H_final], feed_dict=feed_dict)
         
    # For every epoch display a sample text and save
    if prev_epoch == epoch:
        
        proc.print_text_generation_header()    # Display a short text from the current learned state in every epoch (first batch)
        
        # Init Y, H for the sample run
        sample_y = np.array([[proc.encode_char("C")]]) # Start with 'C' 
        sample_h = np.zeros([1, inner_size * n_layers])
        
        for _ in range(1000):
            sample_yo, sample_h = sess.run([Yo, H_final], feed_dict={X: sample_y, p_keep: 1.0, H_init: sample_h, batch_size_: 1})
            
            sample_chr = proc.sample_from_probabilities(sample_yo, topn=10 if epoch <= 8 else 2)
            
            print(chr(proc.decode_char(sample_chr)), end="")
            sample_y = np.array([[sample_chr]])
            
        proc.print_text_generation_footer()
        
        # Create a checkpoint every epoch
        saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
        print("Saved file: " + saved_file)
        
        prev_epoch += 1
        
    # Save for the last epoch
    if epoch == nb_epochs:
        saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
        print("Saved file: " + saved_file)
        
    progress.step(epoch, reset=step % total_batches_size == 0)    # Display progress bar

    init_state = out_state    # Loop state 
    step += batch_size * seq_len
