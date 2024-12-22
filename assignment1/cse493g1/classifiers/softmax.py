import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    num_dim = W.shape[0]
    for i in range(num_train):
      scores= X[i, :].dot(W)
      sum = 0
      for j in range(num_class):
         sum += np.exp(scores[j])
         scores[j] = np.exp(scores[j])
      scores_norm = scores/sum

      for c in range(num_class):
        if c == y[i]:
          dW[:,c] += X[i,:].T * (scores_norm[c]-1)
        else:
          dW[:,c] += X[i,:].T * (scores_norm[c])
      loss -= np.log(scores_norm[y[i]])
    
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    num_dim = W.shape[0]
    
    scores= X.dot(W) # N x C
    scores_exp = np.exp(scores)
    scores_sum = np.sum(scores_exp, axis=1, keepdims= True)

    scores_norm = scores_exp / scores_sum
    loss += np.sum(-np.log(scores_norm[np.arange(num_train),y]))
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW_mask = scores_norm
    dW_mask[np.arange(X.shape[0]), y] -= 1
    dW = X.T.dot(dW_mask)


    dW /= num_train
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
