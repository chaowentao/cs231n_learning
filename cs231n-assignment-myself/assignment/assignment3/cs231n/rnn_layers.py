from __future__ import print_function, division
from builtins import range
import numpy as np
"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # pass
    affine_h = np.dot(prev_h, Wh)  # (N, H)
    affine_x = np.dot(x, Wx) + b  # (N, H)
    new_h = affine_x + affine_h  # (N, H)
    next_h = np.tanh(new_h)  # (N, H)
    cache = (x, prev_h, next_h, Wx, Wh)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # pass
    x, prev_h, next_h, Wx, Wh = cache
    dnew_h = (1 - next_h**2) * dnext_h  # N*H
    dx = np.dot(dnew_h, Wx.T)  # N*H x H*D = N*D
    dWx = np.dot(x.T, dnew_h)  # N*H x N*D = D*H
    dprev_h = np.dot(dnew_h, Wh.T)  # N*H x H*H = N*H
    dWh = np.dot(prev_h.T, dnew_h)  # N*H x N*H = H*H
    db = np.sum(dnew_h, axis=0)

    # x, prev_h, Wx, Wh, next_h = cache
    # d_new_h = dnext_h * (1 - next_h ** 2)
    # dWx = np.dot(x.T, d_new_h)
    # db = np.sum(d_new_h, axis=0)
    # dx = np.dot(d_new_h, Wx.T)
    # dWh = np.dot(prev_h.T, d_new_h)
    # dprev_h = np.dot(d_new_h, Wh.T)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # pass
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros(N * T * H).reshape(N, T, H)
    h_part = h0
    for i in range(T):
        x_part = x[:, i:i + 1, :].reshape(N, D)
        h_part, _ = rnn_step_forward(x_part, h_part, Wx, Wh, b)
        h[:, i:i + 1, :] = h_part.reshape(N, 1, H)
    cache = (x, h0, Wh, Wx, b, h)

    # N, T, D = x.shape
    # hiddens = []
    # hidden = h0
    # for i in range(T):
    #     xt = x[:,i,:]
    #     hidden, _ = rnn_step_forward(xt, hidden, Wx, Wh, b)
    #     hiddens.append(hidden)
    # h = np.stack(hiddens, axis=1)
    # cache = (x, h0, Wh, Wx, b, h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # pass
    # x, h0, Wh, Wx, b, h = cache
    # N, T, D = x.shape
    # N, H = h0.shape
    # dx = np.zeros_like(x)
    # dh0 = np.zeros_like(h0)
    # dWx = np.zeros_like(Wx)
    # dWh = np.zeros_like(Wh)
    # db = np.zeros_like(b)
    # dh0_part = 0
    # for i in range(T):
    #   dh_part = dh[:,T-i-1,:]
    #   if T-i-1>0:
    #     h0_part = h[:,T-i-2,:]
    #   else:
    #     h0_part = h0
    #   cache_part = (x[:,T-i-1,:],h0_part,h[:,T-i-1,:],Wx,Wh)
    #   dx[:, T-i-1, :],dh0_part,dWx_part,dWh_part,db_part = rnn_step_backward(dh_part + dh0_part,cache_part)
    #   dh0 = dh0_part
    #   dWx = dWx + dWx_part
    #   dWh = dWh + dWh_part
    #   db = db + db_part

    x, h0, Wh, Wx, b, h = cache
    _, T, _ = dh.shape
    dx = np.zeros_like(x)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h = 0
    for i in range(T):
        t = T - 1 - i
        xt = x[:, t, :]
        dht = dh[:, t, :]
        if t > 0:
            prev_h = h[:, t - 1, :]
        else:
            prev_h = h0
        next_h = h[:, t, :]
        dx[:, t, :], dprev_h, dwx, dwh, db_ = rnn_step_backward(dht + dprev_h, (xt, prev_h, next_h,Wx, Wh))
        dWx += dwx
        dWh += dwh
        db += db_
        dh0 = dprev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # pass
    # 找出每个句子对应的向量 比如[0 3 1 2] ，0,3,1,2代表第一句话有4个单词，每个单词的长度为0,3,1,2，对应向量为W矩阵中0,3,1,2行
    out = W[x, :]
    cache = (x, W.shape)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # pass
    N,T,D = dout.shape
    x, W_shape = cache # x (N,T) W(V,D)
    dW = np.zeros(W_shape)
    np.add.at(dW,x,dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # pass
    N,H = prev_h.shape
    A = np.dot(x,Wx) + np.dot(prev_h,Wh) + b # (N,4H)
    ai, af, ao, ag = np.split(A, 4, axis=1)
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)
    next_c = f * prev_c + i * g # next_c (N,H)
    next_h = o*np.tanh(next_c) # next_h (N,H)
    cache = (x,prev_h,prev_c,i,f,o,g,next_c,Wx,Wh)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # pass
    # N,H = dnext_h.shape
    # x,prev_h,prev_c,i,f,o,g,next_c,Wx,Wh = cache
    # dnext_c = dnext_h*o*(1-np.tanh(next_c)**2) + dnext_c #dnext_c由两部分组成

    # do = dnext_h * np.tanh(next_c) # (N,H)
    # df = dnext_c * prev_c # (N,H)
    # di = dnext_c * g # (N,H)
    # dg = dnext_c * i # (N,H)
    # dx = np.dot(di*i*(1-i),Wx[:,0:H].T) # (N,H)*(D,H).T=(N,D)
    # dx = np.dot(df*f*(1-f),Wx[:,H:2*H].T)+dx # (N,H)*(D,H).T=(N,D)
    # dx = np.dot(do*o*(1-o),Wx[:,2*H:3*H].T)+dx # (N,H)*(D,H).T=(N,D)
    # dx = np.dot(dg*(1-g**2),Wx[:,3*H:].T)+dx # (N,H)*(D,H).T=(N,D)

    # dprev_h = np.dot(di*i*(1-i),Wh[:,0:H].T) # (N,H)*(H,H).T=(N,H)
    # dprev_h = np.dot(df*f*(1-f),Wh[:,H:2*H].T)+dprev_h # (N,H)*(H,H).T=(N,D)
    # dprev_h = np.dot(do*o*(1-o),Wh[:,2*H:3*H].T)+dprev_h # (N,H)*(H,H).T=(N,D)
    # dprev_h = np.dot(dg*(1-g**2),Wh[:,3*H:].T)+dprev_h # (N,H)*(H,H).T=(N,D)

    # dprev_c =(dnext_c) * f # (N,H)

    # dWx = np.concatenate((np.dot(x.T, di*i*(1-i)),np.dot(x.T, df*f*(1-f)),np.dot(x.T, do*o*(1-o)),np.dot(x.T, dg*(1-g**2))),axis=1) # (N,D).T*(N,H)=(N,H)
    # dWh = np.concatenate((np.dot(prev_h.T, di*i*(1-i)),np.dot(prev_h.T, df*f*(1-f)),np.dot(prev_h.T,do*o*(1-o)),
    #           np.dot(prev_h.T, dg*(1-g**2))),axis=1)  # (N,H).T*(N,H)=(H,H)
    # db = np.concatenate((np.sum(di*i*(1-i),axis=0),np.sum(df*f*(1-f),axis=0),np.sum(do*o*(1-o),axis=0),np.sum(dg*(1-g**2),axis=0)))


    N,H = dnext_h.shape
    x,prev_h,prev_c,i,f,o,g,next_c,Wx,Wh = cache
    dnext_c = dnext_h * o * (1 - np.tanh(next_c) ** 2) + dnext_c
    dprev_c = dnext_c * f
    do = dnext_h * np.tanh(next_c)
    df = dnext_c * prev_c
    di = dnext_c * g
    dg = dnext_c * i
    dai = di * i * (1 - i)
    daf = df * f * (1 - f)
    dao = do * o * (1 - o)
    dag = dg * (1 - g ** 2)
    da = np.concatenate([dai, daf, dao, dag], axis=1)  # (N, 4H)
    dx = np.dot(da, Wx.T) # (N,4H)*(D,4H).T = (N,D)
    dprev_h = np.dot(da, Wh.T) # (N,4H)*(H,4H).T = (N,H)
    dWx = np.dot(x.T, da) # (N,D).T*(N,4H).T = (D,4H)
    dWh = np.dot(prev_h.T, da) # (N,H).T*(N,4H).T = (H,4H)
    db = np.sum(da, axis=0) # (4H,) 按列求和
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # pass
    N, T, D = x.shape
    _, H = h0.shape
    prev_c = np.zeros((N,H))
    prev_h = h0
    h = np.zeros((N,T,H)) # (N, T, H)
    cache = []
    for i in range(T):
      next_h,next_c,c = lstm_step_forward(x[:,i,:],prev_h,prev_c,Wx,Wh,b)
      h[:,i,:] = next_h
      prev_h = next_h
      prev_c = next_c
      cache.append(c)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # pass
    N, T, H = dh.shape
    x,prev_h,prev_c,i,f,o,g,next_c,Wx,Wh = cache[0]
    D,_ = Wx.shape
    dx = np.zeros((N,T,D))
    dh0 = np.zeros_like(prev_h)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros((4*H,))
    for i in range(T):
      t = T - i - 1
      if t == T - 1:
          dht = dh[:, t, :]
          dnext_c = np.zeros((N, H))
      else:
          dht = dh[:, t, :] + dprev_h
      dx_, dprev_h, dnext_c, dWx_, dWh_, db_ = lstm_step_backward(dht, dnext_c, cache[t])
      dx[:, t, :] = dx_
      dWx += dWx_
      dWh += dWh_
      db += db_
      dh0 = dprev_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
