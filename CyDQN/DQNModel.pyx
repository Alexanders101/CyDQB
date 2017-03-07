#!python
#cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False, cdivision=True
import numpy as np
from keras import optimizers
from keras.models import Model, model_from_json
from keras import backend as K

cdef object copy_model(object model):
    copy = model_from_json(model.to_json())
    copy.set_weights(model.get_weights())
    return copy

def huber_loss(y_true, y_pred, clip_value=1):
    x = K.abs(y_true - y_pred)

    condition = x < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (x - .5 * clip_value)
    return K.switch(condition, squared_loss, linear_loss)

def DQN_loss(y_true, y_pred, loss):
    return K.sum(loss(y_true, y_pred), axis=-1)

cdef list get_soft_target_model_updates(object target, object source, FLOAT_t tau):
    target_weights = target.trainable_weights + target.non_trainable_weights
    source_weights = source.trainable_weights + source.non_trainable_weights
    tau_p = 1.0-tau
    return [(tw, tau*sw + tau_p*tw) for tw, sw in zip(target_weights, source_weights)]

class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, constraints, loss):
        updates = self.optimizer.get_updates(params, constraints, loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()


cdef class DQNModel:
    def __cinit__(self, object model, object optimizer, object loss,
                  FLOAT_t gamma_=0.99, FLOAT_t tau=1.0, BOOL_t double_dqn=False):
        """ DQN loss and training implemented in Keras
        Parameters
        ----------
        model : Model
            The Keras model, must be functional for now
        optimizer : Optimizer
            An optimizer from keras.optimizers
        gamma_ : float32
            The Damping factor for score of future events

        """
        self.model = model
        self.target_model = copy_model(model)
        self.gamma = gamma_
        self.double_dqn = double_dqn

        objective = lambda y_true, y_pred: DQN_loss(y_true, y_pred, loss)
        if (tau < 1):
            optimizer = AdditionalUpdatesOptimizer(optimizer,
                                                   get_soft_target_model_updates(self.target_model, self.model, tau))

        self.target_model.compile('sgd', 'mse')
        self.model.compile(optimizer, objective)

    cdef FLOAT_t[:, ::1] predict(self, UINT8_t[:,:,:,::1] x):
        return self.model.predict(np.asarray(x))

    cdef FLOAT_t fit(self, UINT8_t[:,:,:,::1] S, UINT8_t[:,:,:,::1] NS, UINT16_t[::1] A, FLOAT_t[::1] R,
                          UINT8_t[::1] T, FLOAT_t[::1] Errors, FLOAT_t[::1] weights):
        """ Fit on a batch of data
        Parameters
        ----------
        S : np.ndarray[UINT8_t, ndim=4]
            The beginning states, Input into the nerual network
        NS : np.ndarray[UINT8_t, ndim=4]
            The states following the given actions
        A : np.ndarray[UINT16_t, ndim=1]
            The actions
        R : np.ndarray[FLOAT_t, ndim=1]
            The Score of the action
        T : np.ndarray[UINT8_t, ndim=1]
            Wether or not the Action resulted in the game ending

        Returns
        -------
        float32 :
            Loss of the current fit
        """
        cdef FLOAT_t[:, ::1] VS
        cdef FLOAT_t[:, ::1] VSS
        cdef FLOAT_t[:, ::1] VNS

        VS = self.model.predict(np.asarray(S))
        VNS = self.target_model.predict(np.asarray(NS))

        if self.double_dqn:
            VSS = self.model.predict(np.asarray(NS))
            double_q_value(VS, VSS, VNS, A, R, T, self.gamma, Errors)
        else:
            q_value(VS, VNS, A, R, T, self.gamma, Errors)

        return self.model.train_on_batch(S, VS, weights)

    cdef str summary(self):
        return self.model.summary()

    cdef void hard_update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    cdef void save_weights(self, str filepath, BOOL_t overwrite=True):
        self.model.save_weights(filepath, overwrite)

    cdef void load_weights(self, str filepath):
        self.model.load_weights(filepath)