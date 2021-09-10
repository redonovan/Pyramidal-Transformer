# Transform Attend Spell speech recognition model - research.
# Written Aug-Sep 2021.


import os
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_probability as tfp
from tensorflow.keras.layers.experimental import preprocessing


# GPU memory hack
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# set random seed for reproducibility
tf.random.set_seed(1)


# Hyperparameters

# data
mel_dim      = 40    # free choice, number of logmels
max_len      = 1700  # max frames, must be >= loglim from filter_lengths() below
voc_dim      = 123   # one more than the highest unicode value in the vocabulary, which is 122 for 'z'.
                     # the vocab is {a,b,c...z,0...9,<space>,<comma>,<period>,<apost>,<unk>} + {<sos>,<eos>}
                     # where <unk> is ?, <sos> is ^, and <eos> is $.
sample_rate  = 16000 # from LibriSpeech
# model
lis_dim      = 256   # free choice, (half of the) listener output dimension
lis_layers   = 3     # free choice, number of layers in the listener (Transformer)
nheads       = 8     # free choice, number of Transformer heads
dropout_rate = 0.1   # free choice, dropout rate in Transformer
dec_dim      = 512   # free choice, dimension of the DecoderCell's internal LSTMs
att_dim      = 512   # free choice, dimension of DecoderCell MLPs used to compute attention queries and keys
# training
norm_count   = 1000  # free choice, the number of normalization adaptation examples, 1000 takes 47s
frac_pyp     = 0.1   # free choice, the fraction of previous y predictions to use as y input
                     # This enables me to train as in the LAS paper, either with frac_pyp = 0 or 0.1
batch_size   = 1     # Limited by the amount of memory in the GPU; most efficiently a power of 2
num_epochs   = 1     # free choice; 1 epoch takes ~8.5 hours on my machine
# decoding
max_dec      = 300   # free choice, maximum number of characters to decode for if <eos> token is not found


# Data

# I prepared librispeechpartial as a lower-diskspace version of the LibriSpeech tfds dataset.
# Load the train_clean100 split into train_ds

import tensorflow_datasets as tfds

builder  = tfds.builder("librispeechpartial")
info     = builder.info
# dev_ds   = builder.as_dataset(split="dev_clean")
# test_ds  = builder.as_dataset(split="test_clean")
train_ds = builder.as_dataset(split="train_clean100")


# Shuffle the training dataset.
# Buffer size 2048 adds about 4GB to the resident memory size (non GPU) taking it to 9GB.
train_ds = train_ds.shuffle(2048, reshuffle_each_iteration=True, seed=1)


# tfio versions compatible with TensorFlow 2.3.0 do not have tfio.audio.spectrogram()
# so I use tf.signal instead
def get_spectrogram(wt):
    """Adapted from help(tf.signal.mfccs_from_log_mel_spectrograms)
    inputs  : waveform tensor, shape (..., samples) with floating point values in [-1, 1]
    returns : log mel spectrograms, shape (..., frames, mel_dim) as float32s
              ... means the batch dimension is optional
    """
    #
    # These have to be tf.constant because if I just do the calculations in Python then tf complains:
    # "ValueError: Creating variables on a non-first call to a function decorated with tf.function."
    frame_length = tf.constant(sample_rate * 25 // 1000)
    frame_step   = tf.constant(sample_rate * 10 // 1000)
    # 
    # produces shape (..., frames, fftbins)
    stfts = tf.signal.stft(wt, frame_length, frame_step)
    spectrograms = tf.abs(stfts)
    #
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, mel_dim
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))
    #
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    #
    # return shape (..., frames, mel_dim)
    return log_mel_spectrograms


# map function to convert the Librispeech Dataset dictionary into {logmels, speaker_id, ygt}
# pyramid downsampling is not relevant for the Transformer version, but I leave it in anyway.
# @tf.autograph.experimental.do_not_convert
def transform(d):
    wi = d['speech']         # -> (samples)
    wf = tf.cast(wi, dtype=tf.float32)
    wf = wf / 32768.0
    sf = get_spectrogram(wf) # -> (frames, mel_dim)
    nf = tf.shape(sf)[0]     # number of frames
    pd = 2 ** lis_layers     # pyramid downsampling due to listener layers (N/A)
    nh = nf // pd            # number of hidden representations after pyramid downsampling
    nf = nh * pd             # number of original frames that will be used
    sp = sf[:nf,:]           # spectrogram of these frames only, shape (frames, mel_dim)
    t  = d['text']           # -> () with dtype=tf.string
    # permitted characters are {a,b,c...,z,0,...,9,<space>,<comma>,<period>,<apostrophe>,<unk>}
    # The unknown token I use is ?
    l  = tf.strings.lower(t)
    r  = tf.strings.regex_replace(l,'[^([:alnum:]| |,|.|\')]','?')
    # prefix the text with the <sos> token, for which I use ^
    p1 = tf.strings.join([b'^', r])
    # postfix the text with the <eos> token, for which I use $
    p2 = tf.strings.join([p1, b'$'])
    # Decode the text into numbers, using '?' (63) for any unknown conversions (unlikely).
    # The vocabulary above therefore spans values 32 (space) to 122 (z).
    ysp = tf.strings.unicode_decode(p2,'utf-8', replacement_char=63) # -> (nchars, )
    # To be used in neural networks the decoded values must be 1-hot encoded
    yoh = tf.one_hot(ysp, voc_dim) # -> (nchars, voc_dim) as float32s
    # this function returns a dictionary; gt means ground truth
    o = {'logmels' : sp, 'speaker_id' : d['speaker_id'], 'ygt' : yoh}
    return o


# convert the Librispeech elements into {logmels, speaker_id, ygt}
new_ds = train_ds.map(transform)

# Filter out a few extremely long examples that cause training to fall over (exceed GPU memory).
# Parameters loglim and ygtlim have been chosen to exclude less than 0.5% of my data.
# This enabled me to run with batch_size=8 on my machine (pyramidal bi-LSTM Listener).
# @tf.autograph.experimental.do_not_convert
def filter_lengths(d):
    logmels = d['logmels']         # (frames, mel_dim)
    ygt     = d['ygt']             # (nchars, voc_dim)
    logl    = tf.shape(logmels)[0]
    ygtl    = tf.shape(ygt)[0]
    loglim  = 1700
    ygtlim  = 300
    return tf.math.logical_and(logl < loglim, ygtl < ygtlim)


fil_ds = new_ds.filter(filter_lengths)


# Normalization computes and stores means and variances for each logmel dimension.
# Processing all of train_clean100 (~28,000 examples) takes a long time.
# However, the mean and variance vectors don't change much after 100 examples.
# The number to process is chosen with the training parameter norm_count.
print('Computing data normalization vectors...')
norm = preprocessing.Normalization(axis=-1)
norm.adapt(fil_ds.take(norm_count).map(lambda d: d['logmels']))

# normalize the logmels to have 0 mean and stdev 1 in each dimension
# @tf.autograph.experimental.do_not_convert
def normalize(d):
    logmels = norm(d['logmels'])
    o = {'logmels' : logmels, 'speaker_id' : d['speaker_id'], 'ygt' : d['ygt']}
    return o


nor_ds = fil_ds.map(normalize)


# padded_batch pads all arrays with 0s to make rectangular tensors
pad_ds = nor_ds.padded_batch(
    batch_size,
    padded_shapes=({'logmels' : (None, mel_dim), 'speaker_id' : (), 'ygt' : (None, voc_dim)}))


# map function to add logmel and character masks to the training dataset
# @tf.autograph.experimental.do_not_convert
def gen_masks(d):
    logmels = d['logmels']  # (batch, frames, mel_dim)
    # create a logmel mask where False indicates a padded value (0s in every logmel dimension)
    logmel_mask = tf.cast(tf.reduce_sum(logmels, axis=-1), tf.bool)  # (batch, frames)
    # add to the dictionary
    d['logmel_mask'] = logmel_mask
    # extract the ground truth y values
    ygt = d['ygt']  # (batch, nchars, voc_dim)
    # create a ground truth mask where False indicates a padded value
    ygt_mask = tf.cast(tf.reduce_sum(ygt, axis=-1), tf.bool)  # (batch, nchars)
    # add to the dictionary
    d['ygt_mask'] = ygt_mask
    return d


mask_ds = pad_ds.map(gen_masks)



# Model

# Multi-head Attention layer
# This class is necessary because TensorFlow 2.3.0 doesn't have a MultiHeadAttention layer.
# Since I wrote this I've read the TensorFlow Portuguese to English transformer tutorial.
# My implementation is slightly less efficient than theirs, with separate matrices for every head.
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, nheads=4, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.nheads = nheads
    #
    # build is called automatically the first time __call__() is called
    def build(self, input_shape):
        query_shape, value_shape = input_shape
        # the last dimension of both the query and value tensors should be the same
        assert(query_shape[-1] == value_shape[-1])
        # this is the model_dim
        self.model_dim = query_shape[-1]
        # compute the key (and query and value) dimension for each head
        self.dk        = self.model_dim // self.nheads
        # create query and value weights and scaled attention layers for each attention head
        self.qwl = []
        self.vwl = []
        self.sal = []
        for h in range(self.nheads):
            qw = self.add_weight(shape=(self.model_dim, self.dk),
                                 initializer="random_normal",
                                 name=f"mha_query{h}",
                                 trainable=True)
            vw = self.add_weight(shape=(self.model_dim, self.dk),
                                 initializer="random_normal",
                                 name=f"mha_value{h}",
                                 trainable=True)
            sa = layers.Attention(use_scale=True)
            self.qwl.append(qw)
            self.vwl.append(vw)
            self.sal.append(sa)
        # and the output weight matrix applied to the concatenated head vector
        self.wo = self.add_weight(shape=(self.nheads * self.dk, self.model_dim),
                                  initializer="random_normal",
                                  name="mha_output",
                                  trainable=True)
    #
    #
    # inputs should be a list of [query, value] tensors, each of shape (batch, timesteps, model_dim)
    # keys are assumed to be equal to values (the usual case)
    # mask should be a list of [query, value] masks where False means a pad
    def call(self, inputs, mask):
        query, value = inputs
        # loop over heads
        hl = []
        for h in range(self.nheads):
            # use the weight matrices for each head to compute that head's query and value tensors
            qi = tf.matmul(query, self.qwl[h])
            vi = tf.matmul(value, self.vwl[h])
            # apply this head's scaled attention layer
            hi = self.sal[h]([qi,vi], mask)
            # add this head's output to the list
            hl.append(hi)
        # concatenate the individual head outputs (if necessary) along their last axis
        if self.nheads == 1:
            h = hl[0]
        else:
            h = layers.concatenate(hl)
        # one final linear transformation produces the multi-head-attention output
        mha = tf.matmul(h, self.wo)
        return mha
    # enable serialization (not used)
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({"nheads": self.nheads})
        return config


class TransformerLayer(keras.layers.Layer):
    def __init__(self, nheads, layer_dim, dropout_rate, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        # record for debugging
        self.nheads       = nheads
        self.layer_dim    = layer_dim
        self.dropout_rate = dropout_rate
        # internal layers
        self.mha = MultiHeadAttention(nheads)
        self.dr1 = layers.Dropout(dropout_rate)
        self.nl1 = layers.LayerNormalization()
        self.dl1 = layers.Dense(layer_dim * 4, activation='relu')
        self.dl2 = layers.Dense(layer_dim)
        self.dr2 = layers.Dropout(dropout_rate)
        self.nl2 = layers.LayerNormalization()
        #
    def call(self, inputs, mask, training):
        # inputs    (batch, timesteps, layer_dim)
        # mask      (batch, timestep)
        # training  Python boolean specifying whether the layer is training
        #
        # apply masked multi-head self-attention
        sa = self.mha([inputs, inputs], [mask, mask])  # (batch, timesteps, layer_dim)
        # dropout
        p1 = self.dr1(sa, training=training)           # (batch, timesteps, layer_dim)
        # add and norm
        a1 = inputs + p1
        n1 = self.nl1(a1)                              # (batch, timesteps, layer_dim)
        #
        # pointwise feed forward network
        d1 = self.dl1(n1)                              # (batch, timesteps, layer_dim*4)
        d2 = self.dl2(d1)                              # (batch, timesteps, layer_dim)
        # dropout
        p2 = self.dr2(d2, training=training)           # (batch, timesteps, layer_dim)
        # add and norm
        a2 = n1 + p2                                   # (batch, timesteps, layer_dim)
        n2 = self.nl2(a2)                              # (batch, timesteps, layer_dim)
        #
        return n2 # (batch, timesteps, layer_dim)
    

class Transformer(keras.layers.Layer):
    def __init__(self, nlayers, nheads, trans_dim, dropout_rate, max_len, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        # store parameters
        self.nlayers      = nlayers
        self.nheads       = nheads
        self.trans_dim    = trans_dim
        self.dropout_rate = dropout_rate
        self.max_len      = max_len
        # positional encoding tensor, shape (max_len, trans_dim)
        self.pet   = self.compute_pet()
        # learnable input scaling variable
        self.scale = tf.Variable(0.05)
        # input dropout layer
        self.idr   = layers.Dropout(dropout_rate)
        # list of TransformerLayers
        self.lays  = [TransformerLayer(nheads, trans_dim, dropout_rate) for _ in range(nlayers)]
    #
    def call(self, inputs, mask, training):
        # inputs    (batch, timesteps, trans_dim)
        # mask      (batch, timesteps)
        # training  Python boolean specifying whether the layer is training
        #
        # scale the input to enable it to compete with the positional encoding
        # the scale is learnable, with initial value sqrt(trans_dim) chosen for Embedded inputs
        sci = inputs * tf.sqrt(tf.cast(self.trans_dim, tf.float32)) * 20.0 * self.scale
        # add the positional encoding, broadcasting over batch
        ips = sci + self.pet[0:tf.shape(inputs)[1]]
        # apply dropout to produce the transformer layer input, (batch, timesteps, trans_dim)
        x = self.idr(ips, training=training)
        # apply the transformer layers
        for i in range(self.nlayers):
            x = self.lays[i](x, mask, training)
        # return the final x, shape (batch, timesteps, trans_dim)
        return x
    #
    def compute_pet(self):
        # compute positional encoding tensor
        pos = np.array(range(self.max_len))
        pel = []
        for d in range(self.trans_dim):
            if d % 2 == 0:
                row = np.sin(pos/(10000**(d/self.trans_dim)))
            else:
                row = np.cos(pos/(10000**((d-1)/self.trans_dim)))
            pel.append(row)
        # pea has shape (max_len, trans_dim)
        pea = np.array(pel).T
        pet = tf.convert_to_tensor(pea, dtype=tf.float32)
        return pet


class Listener(keras.layers.Layer):
    def __init__(self, lis_layers, lis_dim, nheads, dropout_rate, max_len, **kwargs):
        super(Listener, self).__init__(**kwargs)
        # store parameters
        self.lis_layers   = lis_layers
        self.lis_dim      = lis_dim
        self.nheads       = nheads
        self.dropout_rate = dropout_rate
        self.max_len      = max_len
        # for backwards compatibility all representation dimensions are lis_dim*2
        self.rep_dim = lis_dim*2
        # a pointwise embedding layer encodes logmels into rep_dim
        self.dem = layers.Dense(self.rep_dim)
        # the Transformer encoder
        self.tr  = Transformer(lis_layers, nheads, self.rep_dim, dropout_rate, max_len)
        #
    def call(self, inputs, mask, training):
        # input  (batch, frames, mel_dim)
        # mask   (batch, frames)
        # 
        # embed the logmels in rep_dim dimensions -> (batch, frames, rep_dim)
        x = self.dem(inputs)
        # apply the Transformer encoder
        h = self.tr(x, mask, training)
        # output shapes h = (batch, frames, lis_dim*2) and mask = (batch, frames)
        return h, mask

    
# The Test Model exists as a sanity check on everything up to here.
# It predicts speaker_id from logmels.
class TestModel(keras.Model):
    """To predict speaker_ids from logmels"""
    def __init__(self, lis_layers, lis_dim, nheads, dropout_rate, max_len, num_speakers, **kwargs):
        super(TestModel, self).__init__(**kwargs)
        self.listener = Listener(lis_layers, lis_dim, nheads, dropout_rate, max_len)
        self.condenser = layers.LSTM(lis_dim, return_sequences=False)
        self.dense  = layers.Dense(num_speakers)
    def call(self, inputs, training):
        # models passed to fit() can only have one positional argument plus 'training'
        logmels, logmel_mask = inputs
        h, mask = self.listener(logmels, logmel_mask, training)
        c = self.condenser(h, mask=mask) # -> (batch, lis_dim)
        d = self.dense(c)                # -> (batch, num_speakers) (softmax logits)
        return d


# Calling fit() on the test model:
#
# instantiate the test model
# tmodel = TestModel(lis_layers, lis_dim, nheads, dropout_rate, max_len, 20000)
# tmodel.compile(
#     optimizer=keras.optimizers.RMSprop(1e-3),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
# )
# using fit requires a dataset structured as (inputs, outputs)
# def tmap(d):
#     return (d['logmels'], d['logmel_mask']), (d['speaker_id'],)
# 
# 
# test_ds = mask_ds.map(tmap)
# history = tmodel.fit(test_ds, epochs=1)
# 
# This works, but it needs a batch_size of 2 now I use Transformers.



# Attention can't be done as a layer, because attention ci is computed from si which isn't available
# until you've run the decoder RNN to compute it from step i-1.  In other words, the attention
# calculation has to take place at each timestep, just like the LSTM cell calculation.  These together
# form a layer.  You can't do them, attention and decoder RNN, as separate layer calculations.
# Turns out the same is true of y(i) calculation, which is sometimes fed back in to compute s(i+1).
# I think this means that I need a custom RNN Cell that does a single step of LSTM and attention.
# The required methods and attributes of custom RNN cells are described in tf.keras.layers.RNN documentation.
class DecoderCell(keras.layers.Layer):
    def __init__(self, dec_dim, att_dim, lis_dim, voc_dim, frac_pyp, **kwargs):
        super(DecoderCell, self).__init__(**kwargs)
        # dec_dim will be the dimension of the DecoderCell's internal LSTMs
        self.dec_dim = dec_dim
        # att_dim is used to construct the MLPs used to compute attention queries and keys
        self.att_dim = att_dim
        # lis_dim is (was) the LSTM dimension in the listener pyramid
        self.lis_dim = lis_dim
        # maximum character index plus one
        self.voc_dim = voc_dim
        # fraction of the time we should use the previous y prediction for y input
        self.frac_pyp = frac_pyp
        # call states are [memory, carry] tensors x2 for the internal LSTMs, all shaped (batch, dec_dim),
        # plus previous s vector (batch, dec_dim), previous context vector, (batch, lis_dim*2),
        # and previous y prediction (batch, voc_dim)
        self.state_size = [tf.TensorShape([dec_dim]), tf.TensorShape([dec_dim]),
                           tf.TensorShape([dec_dim]), tf.TensorShape([dec_dim]),
                           tf.TensorShape([dec_dim]), tf.TensorShape([lis_dim*2]),
                           tf.TensorShape([voc_dim])]
        # output size is a yp tensor, shape (batch, voc_dim)
        self.output_size = tf.TensorShape([voc_dim])
        # the LSTMs to be used in the DecoderCell
        self.lstm_cell1 = layers.LSTMCell(self.dec_dim)
        self.lstm_cell2 = layers.LSTMCell(self.dec_dim)
        # the phi attention MLP
        self.phi1 = layers.Dense(att_dim * 2, activation='relu')
        self.phi2 = layers.Dense(att_dim)
        # the psi attention MLP
        self.psi1 = layers.Dense(att_dim * 2, activation='relu')
        self.psi2 = layers.Dense(att_dim)
        # the chr character distribution MLP
        self.chr1 = layers.Dense(voc_dim * 4, activation='relu')
        self.chr2 = layers.Dense(voc_dim)
        #
    def call(self, input_at_t, states_at_t, training, constants=None):
        #
        # input_at_t should be the 1-hot ground truth character vector at timestep i-1
        yin = input_at_t      # shape (batch, voc_dim)
        # states_at_t should be [memory, carry]x2 [psv, pcv, pyp] tensors, see state_size above
        pmc1 = states_at_t[0:2] # previous memory and carry for internal LSTM 1
        pmc2 = states_at_t[2:4] # previous memory and carry for internal LSTM 2
        psv  = states_at_t[4]   # previous s vector, shape (batch, dec_dim)
        pcv  = states_at_t[5]   # previous context vector, shape (batch, lis_dim*2)
        pyp  = states_at_t[6]   # previous y prediction, shape (batch, voc_dim) (as logits)
        # training is a Python boolean used to control dropout, which is not used in this layer
        # constants is a keyword argument that can be passed to RNN.__call__() which contains constants:
        listener_features = constants[0] # shape (batch, frames, lis_dim*2)
        listener_mask     = constants[1] # shape (batch, frames)
        blend             = constants[2] # shape () tf.bool
        #
        # When blending inputs the y to use (ytu) is produced as either yin (teacher-forcing) or a
        # sample drawn from the previous y prediction (pyp) distribution (to improve model robustness).
        # The decision of which to use is determined randomly, such that frac_pyp are from pyp.
        pypdist = tfp.distributions.OneHotCategorical(logits=pyp)
        sample  = pypdist.sample()                                    # shape (batch, voc_dim) int32
        sample  = tf.cast(sample, tf.float32)                         # shape (batch, voc_dim) float32
        rut     = tf.random.uniform((tf.shape(yin)[0],), seed=1)      # shape (batch,)   floats in [0,1)
        rut     = rut[:, None]                                        # shape (batch, 1) floats in [0,1)
        ytu     = tf.cast(rut <  self.frac_pyp, tf.float32) * sample  # shape (batch, voc_dim) float32
        ytu    += tf.cast(rut >= self.frac_pyp, tf.float32) * yin     # shape (batch, voc_dim) float32
        ytu    *= tf.cast(blend, tf.float32)                          # shape (batch, voc_dim) float32
        # When not blending inputs the y to use (ytu) is always yin.
        ytu    += tf.cast(tf.logical_not(blend), tf.float32) * yin    # shape (batch, voc_dim) float32
        #
        # concatenate psv, ytu and pcv, to produce (batch, dec_dim + voc_dim + lis_dim*2)
        rnni = layers.concatenate([psv, ytu, pcv])
        #
        # Run the internal LSTM cells on the concatenated input to compute s(i), shape (batch, dec_dim).
        o1, nmc1 = self.lstm_cell1(rnni, pmc1)
        si, nmc2 = self.lstm_cell2(o1, pmc2)
        #
        # Apply the phi attention MLP to si, producing (batch, att_dim)
        m1 = self.phi1(si)
        m2 = self.phi2(m1)
        # Reshape m2 into the query, shape (batch, 1, att_dim)
        query = layers.Reshape((1, self.att_dim))(m2)
        #
        # Apply the psi attention MLP to the listener features to get the key, (batch, frames, att_dim)
        l1  = self.psi1(listener_features)
        key = self.psi2(l1)
        #
        # Compute attention context vector ci with argument [query, value, key]
        # This should yield shape (batch, 1, lis_dim*2)
        c1 = layers.Attention()([query, listener_features, key], mask=[None, listener_mask])
        #
        # Reshape c1 to produce (batch, lis_dim*2)
        ci = layers.Reshape((-1,))(c1)
        #
        # concatenate si and ci to produce (batch, dec_dim + lis_dim*2)
        sc = layers.concatenate([si,ci])
        #
        # The character distribution MLP predicts y as softmax logits over characters (batch, voc_dim)
        ch = self.chr1(sc)
        yp = self.chr2(ch)
        #
        # return outputs at time t, states at time t+1
        # see https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
        return yp, nmc1 + nmc2 + [si] + [ci] + [yp]
    # The cell could also define a get_initial_state() method, but it doesn't.
    # That means the rnn will feed zeros to call() for the initial state instead.


# The Listen (now Transform) Attend Spell Model
class LASModel(keras.Model):
    def __init__(self, lis_dim, lis_layers, nheads, dropout_rate, max_len, dec_dim, att_dim, voc_dim, frac_pyp, max_dec, **kwargs):
        super(LASModel, self).__init__(**kwargs)
        # maximum number of characters to decode in the decode() function
        self.max_dec  = max_dec
        # DecoderCell parameters required in the decode() function
        self.lis_dim  = lis_dim
        self.dec_dim  = dec_dim
        self.voc_dim  = voc_dim
        # the listener
        self.listener = Listener(lis_layers, lis_dim, nheads, dropout_rate, max_len)
        # the decoder cell and rnn
        self.cell     = DecoderCell(dec_dim, att_dim, lis_dim, voc_dim, frac_pyp)
        self.rnn      = layers.RNN(self.cell, return_sequences=True)
        #
    def call(self, inputs, training):
        # keras models like all their inputs in the first argument
        yins, ymask, logmels, logmel_mask, blend = inputs
        # The call() function operates an rnn, to be used for training/validation/teacher-forced-prediction.
        #
        # yins        shape (batch, nchars, voc_dim)
        # ymask       shape (batch, nchars)
        # logmels     shape (batch, frames, mel_dim)
        # logmel_mask shape (batch, frames)
        # blend       shape () tf boolean
        # training    Python boolean
        #
        # compute the listener representation h and its mask
        h, hmask = self.listener(logmels, logmel_mask, training)
        # h           shape (batch, frames, lis_dim*2)
        # hmask       shape (batch, frames)
        #
        # Compute the y predictions as softmax logits.
        # Since y is post-padded and masked outputs are not used the ymask is optional here.
        yps = self.rnn(yins, mask=ymask, training=training, constants=[h, hmask, blend])
        # yps          shape (batch, nchars, voc_dim)
        #
        return yps
    #
    def decode(self, logmels, logmel_mask):
        # The decode() function performs decoding, predicting unknown characters from logmels.
        # 
        # logmels     shape (batch, frames, mel_dim)
        # logmel_mask shape (batch, frames)
        #
        # where this funcion expects batch = 1
        tf.debugging.assert_equal(tf.shape(logmels)[0],     1, message="las.decode() expects batch_size 1")
        tf.debugging.assert_equal(tf.shape(logmel_mask)[0], 1, message="las.decode() expects batch_size 1")
        #
        # compute the listener representation h and its mask
        h, hmask = self.listener(logmels, logmel_mask, training=False)        
        # h           shape (1, frames, lis_dim*2)
        # hmask       shape (1, frames)
        #
        # the DecoderCell should not blend its inputs when decoding
        blend    = tf.constant(False)
        # the <sos> token is always the first input
        sos_text = b'^'
        sos_code = tf.strings.unicode_decode(sos_text,'utf-8')
        yin      = tf.one_hot(sos_code, self.voc_dim)            # (1, voc_dim)
        # the <eos> token tells us when to stop decoding
        eos_text = b'$'
        eos_code = tf.strings.unicode_decode(eos_text,'utf-8')
        # use TensorArray to accumulate a sparse array of unknown size of decoded int32s
        dec_ta   = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=True)
        dec_i    = 0
        # initial state for the DecoderCell with batch_size of 1
        istate   = [tf.zeros((1, self.dec_dim)), tf.zeros((1, self.dec_dim)),
                    tf.zeros((1, self.dec_dim)), tf.zeros((1, self.dec_dim)),
                    tf.zeros((1, self.dec_dim)), tf.zeros((1, self.lis_dim*2)),
                    tf.zeros((1, self.voc_dim))]
        state    = istate
        # initial y decoded (for loop comparisons)
        yd       = sos_code
        # prepare while loop functions
        def cond(yin, state, yd, dec_ta, dec_i):
            return yd != eos_code
        #
        def body(yin, state, yd, dec_ta, dec_i):
            yp, state = self.cell(yin, state, training=False, constants=[h, hmask, blend])
            # yp shape (1, voc_dim)
            #
            # in this simple decoder the most likely character becomes the decoded char for this timestep
            yd     = tf.argmax(yp, axis=-1)                      # (1,) sparse decoded int64
            yd     = tf.cast(yd, tf.int32)                       # (1,) sparse decoded int32
            # accumulate
            dec_ta = dec_ta.write(dec_i, yd)
            dec_i += 1
            # prepare next input
            yin    = tf.one_hot(yd, self.voc_dim)                # (1, voc_dim)
            # return loop vars
            return yin, state, yd, dec_ta, dec_i
        #
        yin, state, yd, dec_ta, dec_i = tf.while_loop(
            cond, body, (yin, state, yd, dec_ta, dec_i), maximum_iterations=self.max_dec)
        #
        dec_st = tf.squeeze(dec_ta.stack())                      # (nchars,) sparse decoded int32
        return dec_st



# instantiate the model
las = LASModel(lis_dim,lis_layers,nheads,dropout_rate,max_len,dec_dim,att_dim,voc_dim,frac_pyp,max_dec)


# optimizer
# Preliminary experiments have shown that a learning rate of 1e-3 is optimal for the first
# 1000 batches of training.  LAS experiments staircased to 0.7 every 1/3 epoch, and I do that
# here too.

initial_learning_rate = 1e-3

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=28368//(3 * batch_size),
    decay_rate=0.7,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# define accumulators
train_loss = tf.keras.metrics.Mean()
train_acc  = tf.keras.metrics.Mean()

val_loss   = tf.keras.metrics.Mean()
val_acc    = tf.keras.metrics.Mean()


# loss function
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

# @tf.autograph.experimental.do_not_convert
def loss_function(tars, mask, logits):
    # losses due to pads must not be counted
    lt   = loss_object(tars, logits)         # (batch, nchars)
    mask = tf.cast(mask, lt.dtype)           # (batch, nchars) 0.0 where there's a pad, 1.0 otherwise
    lt   = lt * mask
    return tf.reduce_sum(lt) / tf.reduce_sum(mask)


# accuracy function
# @tf.autograph.experimental.do_not_convert
def acc_function(tars, mask, logits):
    # accuracies of pads must not be counted
    preds = tf.argmax(logits, axis=-1)       # (batch, nchars)
    tars  = tf.argmax(tars, axis=-1)         # (batch, nchars)
    acc   = tf.equal(tars, preds)            # (batch, nchars) bools
    acc   = tf.cast(acc, tf.float32)         # (batch, nchars) float32s
    mask  = tf.cast(mask, tf.float32)        # (batch, nchars) float32s, 0 for pads
    acc   = acc * mask
    return tf.reduce_sum(acc) / tf.reduce_sum(mask)


# Adapted from https://www.tensorflow.org/text/tutorials/transformer:
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

signature_dict = { 'logmels'     : tf.TensorSpec(shape=(None, None, mel_dim), dtype=tf.float32),
                   'logmel_mask' : tf.TensorSpec(shape=(None, None),          dtype=tf.bool),
                   'ygt'         : tf.TensorSpec(shape=(None, None, voc_dim), dtype=tf.float32),
                   'ygt_mask'    : tf.TensorSpec(shape=(None, None),          dtype=tf.bool),
                   'speaker_id'  : tf.TensorSpec(shape=(None),                dtype=tf.int64) }


@tf.function(input_signature = [signature_dict])
# @tf.autograph.experimental.do_not_convert
def train_step(d):
    # ygt is ground truth chars, including start and end characters
    ygt        = d['ygt']         # (batch, nchars, voc_dim)
    ygt_mask   = d['ygt_mask']    # (batch, nchars)
    # y inputs exclude end character
    yins       = ygt[:, :-1]      # (batch, nchars, voc_dim)
    yins_mask  = ygt_mask[:, :-1] # (batch, nchars)
    # y targets exclude the start character
    ytars      = ygt[:, 1:]       # (batch, nchars, voc_dim)
    ytars_mask = ygt_mask[:, 1:]  # (batch, nchars)
    # training blends ground truth inputs with predictions
    blend = tf.constant(True)
    #
    with tf.GradientTape() as tape:
        # call the model to obtain the y predictions (logits) (batch, nchars, voc_dim)
        yps = las([yins, yins_mask, d['logmels'], d['logmel_mask'], blend], training=True)
        # compute the loss
        loss = loss_function(ytars, ytars_mask, yps)
    #
    # compute and apply the gradients
    gradients = tape.gradient(loss, las.trainable_variables)
    # clip gradients to stabilize training
    # with batch_size=1 gradient norms are only rarely over 3.0 after 300 batches
    gradients, gn = tf.clip_by_global_norm(gradients, 3.0)
    # tf.print(gn, tf.linalg.global_norm(gradients))
    optimizer.apply_gradients(zip(gradients, las.trainable_variables))
    # accumulate
    train_loss(loss)
    train_acc(acc_function(ytars, ytars_mask, yps))


@tf.function(input_signature = [signature_dict])
# @tf.autograph.experimental.do_not_convert
def val_step(d):
    # ygt is ground truth chars, including start and end characters
    ygt        = d['ygt']         # (batch, nchars, voc_dim)
    ygt_mask   = d['ygt_mask']    # (batch, nchars)
    # y inputs exclude end character
    yins       = ygt[:, :-1]      # (batch, nchars, voc_dim)
    yins_mask  = ygt_mask[:, :-1] # (batch, nchars)
    # y targets exclude the start character
    ytars      = ygt[:, 1:]       # (batch, nchars, voc_dim)
    ytars_mask = ygt_mask[:, 1:]  # (batch, nchars) 
    # blend is True so that validation loss/acc are comparable to training loss/acc
    # similarly training is True so that dropout operation does not differ with training
    # (validation results are only useful comparatively since this is not the ultimate task)
    blend = tf.constant(True)
    # call the model to obtain the y predictions (logits) (batch, nchars, voc_dim)
    yps = las([yins, yins_mask, d['logmels'], d['logmel_mask'], blend], training=True)
    # accumulate
    val_loss(loss_function(ytars, ytars_mask, yps))
    val_acc(acc_function(ytars, ytars_mask, yps))


# I'll use the dev set as validation data
dev_ds    = builder.as_dataset(split="dev_clean")
dtr_ds    = dev_ds.map(transform)
dfi_ds    = dtr_ds.filter(filter_lengths)
dno_ds    = dfi_ds.map(normalize)
dpd_ds    = dno_ds.padded_batch(
    batch_size,
    padded_shapes=({'logmels' : (None, mel_dim), 'speaker_id' : (), 'ygt' : (None, voc_dim)}))
dmk_ds    = dpd_ds.map(gen_masks)
# I'll validate on about 512 utterances, which should take about 2 minutes
val_steps = 512 // batch_size


# Create a Checkpoint that will manage objects with trackable state,
# one I name "optimizer" and the other I name "model".
checkpoint           = tf.train.Checkpoint(optimizer=optimizer, model=las)
checkpoint_directory = './checkpoints'
checkpoint_prefix    = os.path.join(checkpoint_directory, 'ckpt')


# setup tensorboard logging directories
# run "tensorboard --logdir logs" in shell
# point browser at http://localhost:6006/
current_time         = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir        = 'logs/' + current_time + '/train'
val_log_dir          = 'logs/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer   = tf.summary.create_file_writer(val_log_dir)


# global step counter
gstep = 0

# training loop
# If you've already trained a system and saved checkpoints you can kill the training loop after
# 1 batch (required to set up objects in the optimizer) and then load the latest checkpoint below.
for epoch in range(num_epochs):
    start = time.time()
    #
    # reset accumulators
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
    #
    for i,d in enumerate(mask_ds):
        train_step(d)
        # I will monitor training over groups of batches since there will not be many epochs
        if i % 10 == 0:
            print(f'epoch {epoch+1}  batch {i}  train_loss {train_loss.result():.4f}',
                  f'train_acc {train_acc.result():.4f}')
            # tensorboard log
            with train_summary_writer.as_default():
                _ = tf.summary.scalar('loss', train_loss.result(), step=gstep)
                _ = tf.summary.scalar('acc',  train_acc.result(),  step=gstep)
            # reset training accumulators
            train_loss.reset_states()
            train_acc.reset_states()
        gstep += 1
    # save epoch checkpoint
    cp = checkpoint.save(file_prefix=checkpoint_prefix)
    print('saving checkpoint to', cp)
    # validate epoch
    print('validating', end='', flush=True)
    for i,d in enumerate(dmk_ds):
        if i == val_steps:
            break
        val_step(d)
        if i % 10 == 0:
            print('.', end='', flush=True)
    print(f'\nepoch {epoch + 1} val loss {val_loss.result():.4f}  val acc {val_acc.result():.4f}')
    # tensorboard log
    with val_summary_writer.as_default():
        _ = tf.summary.scalar('loss', val_loss.result(), step=epoch)
        _ = tf.summary.scalar('acc',  val_acc.result(),  step=epoch)
    #
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    


# With the trained system, do some predictions.

# Load the latest checkpoint from disk
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
status.assert_consumed()

# First, look at model predictions when teacher-forcing each input character.

@tf.function(input_signature = [signature_dict])
# @tf.autograph.experimental.do_not_convert
def pred_step(d):
    # ygt is ground truth chars, including start and end characters
    ygt        = d['ygt']         # (batch, nchars, voc_dim)
    ygt_mask   = d['ygt_mask']    # (batch, nchars)
    # y inputs exclude end character
    yins       = ygt[:, :-1]      # (batch, nchars, voc_dim)
    yins_mask  = ygt_mask[:, :-1] # (batch, nchars)
    # y targets exclude the start character
    ytars      = ygt[:, 1:]       # (batch, nchars, voc_dim)
    ytars_mask = ygt_mask[:, 1:]  # (batch, nchars)
    # blend is False since we want to always use the teacher-forced input char
    blend = tf.constant(False)
    # call the model to obtain the y predictions (logits) (batch, nchars, voc_dim)
    yps = las([yins, yins_mask, d['logmels'], d['logmel_mask'], blend], training=False)
    return yps, ytars, ytars_mask



# Produce predictions for 4 batches
print('Predictions when teacher-forcing each input character:\n')
for i, d in enumerate(dmk_ds):
    # get the predictions, as logits
    yps, ytars, ytars_mask = pred_step(d)
    # get the targets as a sparse tensor (batch, nchars)
    sparse_tars = tf.argmax(ytars, axis=-1)
    sparse_tars = tf.cast(sparse_tars, dtype=tf.int32)
    # encode into text (batch)
    tar_text = tf.strings.unicode_encode(sparse_tars, 'UTF-8', replacement_char=63)
    # argmax over the probabilities of each character -> (batch, nchars)
    sparse_preds = tf.argmax(yps, axis=-1)
    sparse_preds = tf.cast(sparse_preds, dtype=tf.int32)
    # encode into text (batch)
    pred_text = tf.strings.unicode_encode(sparse_preds, 'UTF-8', replacement_char=63)
    # compute target lengths (batch)
    lengths = tf.reduce_sum(tf.cast(ytars_mask, tf.int32), axis=-1)
    # printouts
    for t,p,l in zip(tar_text, pred_text, lengths):
        print('predicted', p.numpy()[:l.numpy()])
        print('target   ', t.numpy()[:l.numpy()])
        print()
    if i == 3:
        break


# Second, look at predictions as pure decodings, from logmels and the <sos> token:

signature_list = [ tf.TensorSpec(shape=(1, None, mel_dim), dtype=tf.float32),
                   tf.TensorSpec(shape=(1, None),          dtype=tf.bool) ]

@tf.function(input_signature = signature_list)
# @tf.autograph.experimental.do_not_convert
def decode_step(logmels, logmel_mask):
    # call the model to obtain the decoded text
    dec_text = las.decode(logmels, logmel_mask)
    return dec_text


# Produce predictions for 4 examples
print('Predictions as pure decodings, starting from <sos>:\n')
for i, d in enumerate(dmk_ds.unbatch().batch(1)):
    logmels     = d['logmels']                      # (1, frames, mel_dim)
    logmel_mask = d['logmel_mask']                  # (1, frames)
    # call decoder
    dec_st      = decode_step(logmels, logmel_mask) # (nchars,)
    dec_text    = tf.strings.unicode_encode(dec_st, 'UTF-8', replacement_char=63) # ()
    # ygt is ground truth chars, including start and end characters
    ygt         = d['ygt']                          # (1, nchars, voc_dim)
    ygt_mask    = d['ygt_mask']                     # (1, nchars)
    # y targets exclude the start character
    ytars       = ygt[:, 1:]                        # (1, nchars, voc_dim)
    ytars_mask  = ygt_mask[:, 1:]                   # (1, nchars) 
    # get the targets as a sparse tensor              (nchars, )
    sparse_tars = tf.argmax(ytars, axis=-1)
    sparse_tars = tf.cast(sparse_tars, dtype=tf.int32)
    sparse_tars = tf.squeeze(sparse_tars)
    # encode the targets into text                    ()
    tar_text = tf.strings.unicode_encode(sparse_tars, 'UTF-8', replacement_char=63)
    # compute target length                           ()
    length = tf.squeeze(tf.reduce_sum(tf.cast(ytars_mask, tf.int32), axis=-1))
    # printouts
    print('decoded', dec_text.numpy())
    print('target ', tar_text.numpy()[:length.numpy()])
    print('\n')
    if i == 3:
        break


