'''This module contains an optimizer, which should work for
 all models, unless explicitly mentioned in the optimizer or model doc'''

import numpy as np

class AdamOptimizer():
    '''adam optimizer object
    This object in instanciated by the .fit() method of the model class
        each time it is triggered
    Unlike in Keras, this object should not be instanciated by the user

    WARNING : so far, this implementation is specific for the neural network model class
    '''

    def __init__(self, weights, bias, alpha_init=0.001, beta_1=0.9,
             beta_2=0.999, epsilon=1e-8):

        self.alpha_init=alpha_init
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.epsilon=epsilon

        self.t=0
        #initializing first and second momentum
        self.m_weights = [np.zeros_like(w) for w in weights]
        self.m_bias = [np.zeros_like(b) for b in bias]
        self.v_weights = self.m_weights.copy()
        self.v_bias = self.m_bias.copy()

    def get_update(self, gradient_weights, gradient_bias):
        '''computes the values to be added to weights and bias arrays at
        the end of the train step'''
        self.t+=1
        alpha=self.alpha_init*np.sqrt(1-self.beta_2**self.t)/(1-self.beta_1**self.t)

        # updating 1st and 2nd momenta
        self.m_weights=[self.beta_1 * m + (1-self.beta_1) * grad\
                   for m, grad in zip(self.m_weights, gradient_weights)]
        self.m_bias=[self.beta_1 * m + (1-self.beta_1) * grad\
                   for m, grad in zip(self.m_bias, gradient_bias)]
        self.v_weights=[self.beta_2 * v + (1-self.beta_2) * grad**2\
                   for v, grad in zip(self.v_weights, gradient_weights)]
        self.v_bias=[self.beta_2 * v + (1-self.beta_2) * grad**2\
                   for v, grad in zip(self.v_bias, gradient_bias)]

        #computing the updates
        weights_update = [- alpha * m / (np.sqrt(v) + self.epsilon)\
                                  for m, v in zip( self.m_weights, self.v_weights)]
        bias_update = [- alpha * m / (np.sqrt(v) + self.epsilon)\
                                  for m, v in zip( self.m_bias, self.v_bias)]

        return weights_update, bias_update

