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


def create_model(X, H_init, layers, inner_size, cell_type=tf.contrib.rnn.LSTMCell, p_keep=1.0):
    """
        Returns the outputs and the final state of a size layers*inner_size RNN
        
        Args:
            X (tensor (uint8)): Tensor of inputs for the LSTM
            H (tensor (float32)): Initial state
            layers (int): The number of the networks' inner layers
            inner_size (int): The size of a layer (i.e. the number of neurons per layer)
            cell_type (a type of rnn cell, default: LSTMCell): The cell type
            p_keep (tensor (float32), default: 1.0): The dropout rate; default means no dropout
            
        Returns:
            Tensor (uint8): Prediction outputs from the RNN. Size is infered from X
            Tensor (float32): Final state
    """

    cells = [tf.contrib.rnn.GRUCell(inner_size) for _ in range(layers)]
    
    # naive dropout (i.e. drops only input/output connections)
    dropcells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p_keep) for cell in cells]
    multicell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=False)
    multicell = tf.contrib.rnn.DropoutWrapper(multicell, output_keep_prob=p_keep)
    
    Y, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32, initial_state=H_init)
    
    return Y, H
    
    
    
    
    
    
    