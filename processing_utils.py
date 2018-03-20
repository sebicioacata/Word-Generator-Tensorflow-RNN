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

import os
import numpy as np
import hyper_params as hyper

alpha_size = hyper.ALPHA_SIZE

def encode_char(ch):
    """
    Encodes the input arg (i.e. returns a numeric equivalent).
    Encoding: 0 is unknown
              1 is tab
              2 is LF
              3 to 126 ASCII printable characters
        
    Args:
        ch (str): character to be encoded
        
    Returns:
        int: the integer equivalent of 'ch'
    """
    
    ch = ord(ch)
    
    if ch == 9:
        return 1
    if ch == 10:
        return 2
    elif 32 <= ch <= 126:
        return ch - 29
    else:
        return 0

    
    
def decode_char(ch):
    """
    Restores the input arg to char
        
    Args:
        ch (int): numeric code of char to be decoded
        
    Returns:
        str: the character 'ch'
    """
    
    if ch == 1:
        return 9
    if ch == 2:
        return 10
    elif 32 <= ch + 29 <= 126:
        return ch + 29
    else:
        return 0
    
    
def encode_string(str_):
    """
    Encodes each char of a string
    
    Args:
        str_ (str): string to be encoded
        
    Returns:
        list of ints: a list of encoded chars
    """
    
    return list(map(lambda l: encode_char(l), str_))


def read_data_files(source_directory):
    """
    Scans every file in the source directory, gathers and encodes all text found
    
    Args:
        source_directory (str): the name of the source directory
    
    Returns:
        list of ints: a list of all input text as encoded chars
    """
    
    files_list = [os.path.join(source_directory, file) for file in os.listdir(source_directory)]
    
    encoded_text = []
    
    for file in files_list:
        with open(file) as file_contents:
            print("Reading file: " + file)
            encoded_text.extend(encode_string(file_contents.read()))
            
    return encoded_text

def rnn_batch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    
    Args:
        raw_data: the training text
        batch_size: the size of a training minibatch
        sequence_size: the unroll size of the RNN
        nb_epochs: number of epochs to train on
        
    Returns:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    
    data = np.array(raw_data)
    data_len = data.shape[0]
    
    nb_batches = (data_len - 1) // (batch_size * sequence_size)    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    
    rounded_data_len = nb_batches * batch_size * sequence_size
    
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch
            
def sample_from_probabilities(probabilities, topn=alpha_size):
    """Roll the dice to produce a random integer in the [0..alpha_size] range,
    according to the provided probabilities. If topn is specified, only the
    topn highest probabilities are taken into account.
    
    Args:
        probabilities: a list of size alpha_size with individual probabilities
        topn: the number of highest probabilities to consider. Defaults to all of them.
    
    Returns:
        int: a random integer
    """
    
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    
    return np.random.choice(alpha_size, 1, p=p)[0]

class Progress:
    """Text mode progress bar.
    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(start=True) # to restart form 0%
    The progress bar displays a new header at each restart.
    """
    
    def __init__(self, maxi, size=100, msg=""):
        """
        Progress constructor
        
        Args:
            maxi: the number of steps required to reach 100%
            size: the number of characters taken on the screen by the progress bar
            msg: the message displayed in the header of the progress bat
        """
        
        self.maxi = maxi
        self.p = self.__start_progress(maxi)()  # () to get the iterator from the generator
        self.header_printed = False
        self.epoch = 0
        self.msg = msg
        self.size = size

    def step(self, epoch, reset=False):
        
        if reset:
            self.__init__(self.maxi, self.size, self.msg)
            self.epoch = epoch
        if not self.header_printed:
            self.__print_header()
        
        next(self.p)

    def __print_header(self):
        print()
        
        format_string = "0%{: ^" + str(self.size - 6) + "}100%"
        print(format_string.format("Epoch: "+str(self.epoch)+" | "+self.msg))
        self.header_printed = True

    def __start_progress(self, maxi):
        def print_progress():
            """
            Bresenham's algorithm. Yields the number of dots printed.
            This will always print 100 dots in max invocations.
            """
            dx = maxi
            dy = self.size
            d = dy - dx
            for x in range(maxi):
                k = 0
                while d >= 0:
                    print('-', end="", flush=True)
                    k += 1
                    d -= dx
                d += dy
                yield k

        return print_progress

def print_text_generation_header():
    print()
    print("┌{:─^111}┐".format('Generating random text from learned state'))


def print_text_generation_footer():
    print()
    print("└{:─^111}┘".format('End'))