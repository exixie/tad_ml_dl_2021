import numpy as np
import math


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    prediction = predictions.copy()
    new_exp = np.vectorize(lambda x: math.exp(x))
    if len(predictions.shape)==1:
        pred = prediction-np.max(prediction)
        exp_prob = new_exp(pred)
        probs = np.array(list(map(lambda x: x/np.sum(exp_prob),exp_prob)))
    else:
        pred = list(map(lambda x: x-np.max(x), prediction))
        exp_prob = new_exp(pred)
        probs = np.array(list(map(lambda x: x/np.sum(x),exp_prob)))
    
    return probs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    new_loss = np.vectorize(lambda x: -math.log(x))
    if len(probs.shape)==1:
        probs_target = probs[target_index]
        size_target = 1
    else:
        batch_size = np.arange(target_index.shape[0])
        probs_target = probs[batch_size,target_index.flatten()]
        size_target = target_index.shape[0]
    loss = np.sum(new_loss(probs_target))/size_target
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    prediction = predictions.copy()
    probs = softmax(prediction) 
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if len(predictions.shape)==1:
        dprediction[target_index] -= 1
    else:
        batch_size = np.arange(target_index.shape[0])
        dprediction[batch_size,target_index.flatten()] -= 1
        dprediction = dprediction/target_index.shape[0]

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    #loss = reg_strength*np.sum(np.dot(np.transpose(W),W))
    #batch_size = np.arange(W.shape[1])
    #grad = np.array((list(map(lambda x: np.sum(W,axis=1), batch_size))))
    #grad = 2*reg_strength*np.transpose(grad)
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W

    return loss, grad


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        result = X.copy()
        result[result<0] = 0
        return result

    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out
        d_result[self.X<0] = 0
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.dot(X,self.W.value)+self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_input = np.dot(d_out, np.transpose(self.W.value))
        self.W.grad += np.dot(np.transpose(self.X),d_out)
        self.B.grad += np.sum(d_out, axis=0)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}