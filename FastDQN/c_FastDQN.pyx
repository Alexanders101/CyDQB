#!python
#cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False

# TODO
# DDQN
# PRIORITIZED REPLAY
# Dynamic Frame Skip https://arxiv.org/pdf/1605.05365v2.pdf
#
import numpy as np
cimport numpy as np

from keras import optimizers
from keras.models import Model, model_from_config, model_from_json
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.optimizers import RMSprop
from keras import backend as K

######################
### Typedefs
######################

ctypedef np.float32_t FLOAT_t
ctypedef np.uint8_t UINT8_t
ctypedef np.uint16_t UINT16_t
ctypedef np.uint64_t UINT64_t
ctypedef np.float64_t DOUBLE_t

######################
### Cythonized Utils
######################

cdef FLOAT_t max_1d(np.ndarray[FLOAT_t, ndim=1] arr, int size):
    """ Calculate the maximum element in a 1-dim float array
    Parameters
    ----------
    arr: Numpy array
    size: size of array

    Returns
    -------
    Maximum Value
    """
    cdef int i
    cdef FLOAT_t max = arr[0]
    cdef FLOAT_t cur_val

    for i in range(1, size):
        cur_val = arr[i]
        if cur_val > max:
            max = cur_val

    return max

cdef void error_calc(np.ndarray[FLOAT_t, ndim=2] VS, np.ndarray[FLOAT_t, ndim=2] VNS,
                     np.ndarray[UINT16_t, ndim=1] A, np.ndarray[FLOAT_t, ndim=1] R,
                     np.ndarray[UINT8_t, ndim=1] T, FLOAT_t gamma):
    cdef int batch_size = VS.shape[0]
    cdef int action_count = VS.shape[1]
    cdef FLOAT_t target

    cdef int i
    cdef int j
    for i in range(batch_size):
        # target = R[i] + (gamma * (1 - T[i]) * max_1d(VNS[i], action_count)) - VS[i, A[i]]
        target = R[i] + (gamma * (1 - T[i]) * max_1d(VNS[i], action_count))
        VS[i, A[i]] = target
        # for j in range(action_count):
        #     VS[i, j] += target

######################
### Keras Utils
######################

cdef object copy_model(object model):
    # Requires Keras 1.0.7 since get_config has breaking changes.
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
    # loss =
    # mask = K.clip(K.abs(y_true - y_pred), 0, 1)
    # For now assume that loss is 0 based
    return K.sum(loss(y_true, y_pred), axis=-1)

def get_soft_target_model_updates(target, source, tau):
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

######################
### Helper Classes
######################

cdef class RingBuffer:
    "A ND ring buffer using numpy arrays"
    cdef public np.ndarray data
    cdef np.uint32_t index
    cdef np.uint32_t size

    def __init__(self, tuple shape, dtype):
        """ A fast ring buffer for numpy arrays of constant size
        Parameters
        ----------
        shape : tuple
            The shape of the buffer, the size is the first element of the shape
        dtype :
            Data type of elements
        """
        self.data = np.zeros(shape, dtype=dtype)
        self.index = 0
        self.size = shape[0]

    cdef void append(self, x):
        "adds array x to ring buffer"
        self.data[self.index] = x
        self.index = (self.index + 1) % self.size

    cdef np.ndarray pop(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index - 1) % self.size
        return self.data[idx]

    def __getitem__(self, key):
        return self.data[key]

cdef class DQNModel:
    # Damping factor for future events
    cdef FLOAT_t gamma
    # Base Keras model (Must be Functional)
    cdef public object model
    # The stop_gradient of the model
    cdef object target_model

    def __cinit__(self, object model, object optimizer, FLOAT_t gamma_=0.99, FLOAT_t tau=1.0):
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

        objective = lambda y_true, y_pred: DQN_loss(y_true, y_pred, huber_loss)
        if (tau < 1):
            optimizer = AdditionalUpdatesOptimizer(optimizer,
                                                   get_soft_target_model_updates(self.target_model, self.model, tau))

        self.target_model.compile('sgd', 'mse')
        self.model.compile(optimizer, objective)
        # NS = Input(shape=self.model.input_shape[1:], dtype='float32')
        # self.grad_model = K.function([NS], [K.stop_gradient(self.model(NS))])

    cdef np.ndarray[FLOAT_t, ndim=2] predict(self, np.ndarray[np.uint8_t, ndim=4] X):
        return self.model.predict(X)

    cdef np.float64_t fit(self, np.ndarray[UINT8_t, ndim=4] S, np.ndarray[UINT8_t, ndim=4] NS,
                          np.ndarray[UINT16_t, ndim=1] A,
                          np.ndarray[FLOAT_t, ndim=1] R, np.ndarray[UINT8_t, ndim=1] T):
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
        cdef np.ndarray[FLOAT_t, ndim=2] VS
        cdef np.ndarray[FLOAT_t, ndim=2] VNS

        VS = self.model.predict(S)
        # VNS = self.grad_model((NS.astype(np.float32),))[0].astype(np.float32)
        VNS = self.target_model.predict(NS)

        error_calc(VS, VNS, A, R, T, self.gamma)
        return self.model.train_on_batch(S, VS)

    cdef void hard_update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    cdef void save_weights(self, str filepath, np.uint8_t overwrite=True):
        self.model.save_weights(filepath, overwrite)

    cdef void load_weights(self, str filepath):
        self.model.load_weights(filepath)

######################
### Agent
######################

cdef class Agent_:
    cdef object game
    cdef DQNModel model

    cdef tuple frame_size
    cdef tuple state_size
    cdef UINT64_t num_actions
    cdef UINT64_t channel_axis
    cdef UINT64_t colors

    cdef UINT64_t frame_seq_count
    cdef UINT64_t save_freq
    cdef UINT64_t memory
    cdef UINT64_t batch_size
    cdef DOUBLE_t epsilon
    cdef DOUBLE_t delta_epsilon
    cdef FLOAT_t gamma
    cdef FLOAT_t tau

    cdef str save_name

    def __cinit__(self, object model, object game, tuple frame_size, tuple state_size, UINT64_t num_actions,
                  str save_name, DOUBLE_t epsilon=0.1,
                  DOUBLE_t delta_epsilon=0.00001, FLOAT_t gamma=0.99, UINT64_t batch_size=64,
                  UINT64_t frame_seq_count=1,
                  UINT64_t memory=5000, UINT64_t save_freq=10, FLOAT_t tau=1.0, object optimizer=RMSprop(0.0001)):
        """
        Parameters
        ----------
        model : Model
            Keras model
        game : Game
            Class following the requirements
        frame_size :
            Size of the output from the game
        state_size :
            Size of each state
        num_actions :
            Number of actions for the input to the game
        save_name :
            Name of the model
        epsilon :
        delta_epsilon :
            Random factor
        gamma :
            Damping Factor
        batch_size :
            Size of training batch
        frame_seq_count :
            Number of frames to tile on top of each other
        memory :
            Size of buffer for states
        save_freq :
            Amount of episodes to wait before saving
        tau :
            If >=1, then the rate at which to hard update the target model
            IF < 1 < 0, then the weight at which to incorporate the main model into target model
        optimizer :
            Keras Optimizer
        """
        self.game = game

        self.channel_axis = 2 if K.image_dim_ordering() == "tf" else 0
        self.colors = frame_size[self.channel_axis]

        self.frame_size = frame_size
        self.state_size = state_size

        self.num_actions = num_actions
        self.frame_seq_count = frame_seq_count

        self.epsilon = epsilon
        self.delta_epsilon = delta_epsilon
        self.gamma = gamma
        self.tau = tau

        self.save_freq = save_freq
        self.memory = memory
        self.batch_size = batch_size

        self.save_name = save_name

        print("Compiling Model...")
        self.model = DQNModel(model, optimizer, gamma, tau)
        print("Done")

        print("Starting Agent with the following parameters:")
        print("epsilon: {}".format(epsilon))
        print("delta_epsilon: {}".format(delta_epsilon))
        print("batch_size: {}".format(batch_size))
        print("frame_seq_count: {}".format(frame_seq_count))
        print("memory: {}".format(memory))
        print("optimizer: {}".format(optimizer))

    cdef void play_(self, UINT64_t num_episodes, UINT8_t train=True):
        #################
        ### Buffer Setup
        #################
        cdef RingBuffer D_S = RingBuffer((self.memory,) + self.state_size, np.uint8)
        cdef RingBuffer D_NS = RingBuffer((self.memory,) + self.state_size, np.uint8)
        cdef RingBuffer D_A = RingBuffer((self.memory,), np.uint16)
        cdef RingBuffer D_R = RingBuffer((self.memory,), np.float32)
        cdef RingBuffer D_T = RingBuffer((self.memory,), np.uint8)
        cdef np.ndarray batch_choices = np.arange(self.memory)
        cdef tuple frame_size = (1,) + self.frame_size

        #################
        ### Initial Game state
        #################
        cdef np.ndarray[UINT8_t, ndim=4] s_0 = np.repeat(self.game.reset().reshape(frame_size), self.frame_seq_count,
                                                         axis=self.channel_axis + 1)

        #################
        ### Loop Variables
        #################
        cdef UINT64_t t = 0
        cdef UINT64_t a_t
        cdef UINT64_t episode
        cdef np.ndarray[UINT8_t, ndim=4] s_t
        cdef np.ndarray[UINT8_t, ndim=4] s_t1
        cdef np.ndarray[UINT8_t, ndim=3] x_t1
        cdef UINT8_t terminal
        cdef FLOAT_t r_t
        cdef np.ndarray[np.int64_t, ndim=1] batch
        cdef np.ndarray[FLOAT_t, ndim=2] values
        cdef DOUBLE_t total_loss
        cdef DOUBLE_t total_score

        for episode in range(num_episodes):
            print("====================")
            print("Episode {} has begun".format(episode))

            # Reset game to start a new episode
            self.game.reset()
            s_t = s_0
            terminal = 0
            total_loss = 0
            total_score = 0

            # Update random factor
            if self.epsilon > 0 and t > self.memory:
                self.epsilon -= self.delta_epsilon

            while not terminal:
                # Get the Models chosen Action or random action
                values = self.model.predict(s_t)
                a_t = (np.random.randint(self.num_actions)
                       if np.random.rand() <= self.epsilon
                       else values.argmax())

                # Get the next frame of the game based on chosen action and create the next state
                x_t1, r_t, terminal = self.game.step(a_t, values)
                s_t1 = np.concatenate((s_t[:, :, :, self.colors:], x_t1.reshape(frame_size)),
                                      axis=self.channel_axis + 1)

                # Update Buffers
                total_score += r_t
                D_S.append(s_t[0])
                D_NS.append(s_t1[0])
                D_A.append(a_t)
                D_R.append(r_t)
                D_T.append(terminal)

                # Train Model
                if t > self.memory and train:
                    batch = np.random.choice(batch_choices, self.batch_size, replace=False)
                    total_loss += self.model.fit(D_S[batch], D_NS[batch], D_A[batch], D_R[batch], D_T[batch])
                    if self.tau >= 1.0 and int(t % self.tau) == 0:
                        self.model.hard_update_target_weights()

                # Update timestep
                s_t = s_t1
                t += 1

            print("Total Loss: {:.4f} | Total Score: {:.4f} | Epsilon: {:.5f} | Frames: {}".format(total_loss,
                                                                                                   total_score,
                                                                                                   self.epsilon, t))
            if episode % self.save_freq == 0:
                print("Saving Model to {}.j5".format(self.save_name))
                self.model.save_weights("{}.h5".format(self.save_name))

    def play(self, UINT64_t num_episodes, UINT8_t train=True):
        return self.play_(num_episodes, train)

cpdef Agent_ Agent(build_model, object game, tuple frame_size, UINT64_t num_actions, str save_name,
                   DOUBLE_t epsilon=0.1,
                   DOUBLE_t delta_epsilon=0.00001, FLOAT_t gamma=0.99, UINT64_t batch_size=64,
                   UINT64_t frame_seq_count=1,
                   UINT64_t memory=5000, UINT64_t save_freq=10, FLOAT_t tau=1.0, object optimizer=RMSprop(0.0001)):
    channel_axis = 2 if K.image_dim_ordering() == "tf" else 0
    colors = frame_size[channel_axis]
    state_size = list(frame_size)
    state_size[channel_axis] = frame_seq_count * colors
    state_size = tuple(state_size)

    model = build_model(state_size, num_actions)
    try:
        model.load_weights('{}.h5'.format(save_name))
        print("loading from {}.h5".format(save_name))
    except:
        print("Training a new model")

    return Agent_(model, game, frame_size, state_size, num_actions, save_name, epsilon, delta_epsilon, gamma,
                  batch_size, frame_seq_count,
                  memory, save_freq, tau, optimizer)

def test_methods():
    size = 5
    VS = np.random.rand(size, 8).astype(np.float32)
    VOS = VS.copy()
    VNS = np.random.rand(size, 8).astype(np.float32)
    R = np.random.rand(size).astype(np.float32)
    T = np.random.randint(0, 2, size, np.uint8)
    A = np.random.randint(0, 4, size, np.uint16)

    gamma = np.float32(0.99)

    target = R + (gamma * (1 - T) * np.max(VNS, axis=1))
    # VSS = (VS.transpose() + target).transpose()
    print(target)
    print(A)
    error_calc(VS, VNS, A, R, T, gamma)
    print(VOS)
    print(VS)
    for true, pred in zip(VS, VOS):
        print(DQN_loss(true, pred, huber_loss).eval().tolist())
    # print(target)
    # assert (target == VS).all()
