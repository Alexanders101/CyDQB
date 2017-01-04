#!python
#cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False, cdivision=True
from keras import optimizers
from keras import backend as K
from FastDQN.DQNModel import huber_loss
from time import time
import numpy as np

######################
### Agent
######################

cdef class Agent:
    def __cinit__(self, object model, object optimizer, object loss, object game,
                  UINT64_t num_actions, tuple frame_size, tuple state_size, str save_name, BOOL_t double_dqn=False,
                  DOUBLE_t epsilon=0.1, DOUBLE_t delta_epsilon=0.00001, FLOAT_t gamma=0.99, FLOAT_t tau=1.0, UINT64_t frame_skip=0,
                  UINT64_t batch_size=64, UINT64_t frame_seq_count=1, UINT64_t memory=5000, UINT64_t save_freq=10,
                  FLOAT_t alpha = 0.7, FLOAT_t beta = 0.5):
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

        self.frame_size = frame_size
        self.state_size = state_size
        self.num_actions = num_actions
        self.frame_seq_count = frame_seq_count
        self.channel_axis = 2 if K.image_dim_ordering() == "tf" else 0
        self.colors = frame_size[self.channel_axis]

        self.epsilon = epsilon
        self.delta_epsilon = delta_epsilon
        self.gamma = gamma
        self.tau = tau
        self.frame_skip = frame_skip+1

        self.alpha = alpha
        self.beta = -beta

        self.save_freq = save_freq
        self.memory = memory
        self.batch_size = batch_size

        self.save_name = save_name

        print("Compiling Model...")
        self.model = DQNModel(model, optimizer, loss, gamma, tau, double_dqn)
        print("Done")
        print(self.model.summary())

        print("Starting Agent with the following parameters:")
        print("Name: {}".format(save_name))
        print("Double DQN: {}".format(double_dqn))
        print("Frame Skip: {}".format(frame_skip))
        print("Batch size: {}".format(batch_size))
        print("Stacked Frames: {}".format(frame_seq_count))
        print("Buffer Size: {}".format(memory))
        print("Tau: {}".format(tau))
        print("Epsilon: {}".format(epsilon))
        print("Delta Epsilon: {}".format(delta_epsilon))
        print("Alpha: {}".format(alpha))
        print("Beta: {}".format(beta))


    cdef void play_(self, UINT64_t num_episodes, BOOL_t train=True):
        #################
        ### Buffer Setup
        #################
        cdef np.ndarray[UINT8_t, ndim=4] D_S = np.empty( (self.memory,) + self.state_size, np.uint8 )
        cdef np.ndarray[UINT8_t, ndim=4] D_NS = np.empty( (self.memory,) + self.state_size, np.uint8 )
        cdef np.ndarray[UINT16_t, ndim=1] D_A = np.empty( self.memory, np.uint16 )
        cdef np.ndarray[FLOAT_t, ndim=1] D_R = np.empty( self.memory, np.float32 )
        cdef np.ndarray[UINT8_t, ndim=1] D_T = np.empty( self.memory, np.uint8 )

        cdef np.ndarray[FLOAT_t, ndim=1] D_E = np.zeros( self.memory, np.float32 ) + 0.0001
        cdef np.ndarray[FLOAT_t, ndim=1] P = np.empty( self.memory, np.float32 )
        cdef np.ndarray[FLOAT_t, ndim=1] W = np.empty( self.batch_size, np.float32 )
        cdef np.ndarray[FLOAT_t, ndim=1] errors = np.empty(self.batch_size, np.float32)
        cdef UINT64_t buffer_idx

        cdef np.ndarray batch_choices = np.arange(self.memory, dtype=np.uint64)
        cdef tuple frame_size = (1,) + self.frame_size

        #################
        ### Initial Game state
        #################
        cdef UINT8_t[:,:,:,::1] s_0 = np.repeat(self.game.reset().reshape(frame_size), self.frame_seq_count,
                                                axis=self.channel_axis + 1)

        #################
        ### Loop Variables
        #################
        cdef UINT64_t t = 0
        cdef UINT64_t episode
        cdef UINT64_t explore_time = self.memory

        cdef UINT16_t a_t
        cdef FLOAT_t r_t
        cdef UINT8_t terminal

        cdef UINT8_t[:,:,:,::1] s_t
        cdef UINT8_t[:,:,:,::1] s_t1
        cdef UINT8_t[:,:,::1] x_t1

        cdef UINT64_t[::1] batch
        cdef FLOAT_t[:,::1] values
        cdef DOUBLE_t total_loss
        cdef DOUBLE_t total_score

        cdef DOUBLE_t epsilon = self.epsilon
        cdef DOUBLE_t delta_epsilon = self.delta_epsilon
        cdef DOUBLE_t start_time


        for episode in range(num_episodes):
            print("======================================")
            print("Episode {} has begun".format(episode))
            start_time = time()

            # Reset game to start a new episode
            self.game.reset()
            s_t = s_0
            terminal = 0
            total_loss = 0
            total_score = 0

            # Update random factor
            if epsilon > 0 and t > explore_time:
                epsilon -= delta_epsilon

            while not terminal:
                # Get the Models chosen Action or random action
                if t % self.frame_skip == 0:
                    values = self.model.predict(s_t)
                    if np.random.rand() <= epsilon:
                        a_t = np.random.randint(self.num_actions, dtype=np.uint16)
                    else:
                        a_t = argmax_2d(values)


                # Get the next frame of the game based on chosen action and create the next state
                x_t1, r_t, terminal = self.game.step(a_t, values)

                s_t1 = np.concatenate((s_t[:, :, :, self.colors:], np.reshape(x_t1, frame_size)),
                                      axis=self.channel_axis + 1)

                # Update Buffers
                total_score += r_t

                buffer_idx = t % explore_time
                D_S[buffer_idx] = s_t[0]
                D_NS[buffer_idx] = s_t1[0]
                D_A[buffer_idx] = a_t
                D_R[buffer_idx] = r_t
                D_T[buffer_idx] = terminal
                D_E[buffer_idx] = max_1d(D_E)

                # print(D_S)
                # print(D_NS)
                print(np.asarray(values))
                print(D_A)
                print(D_R)
                print(D_T)
                print(D_E)

                # Train Model
                if t > explore_time and train:
                    # Get probability of each state
                    P = D_E / sum_1d(D_E)

                    # Get the current batch indices
                    batch = np.random.choice(batch_choices, self.batch_size, replace=False, p=P)

                    # Calculate weight for each sample
                    W = (P[batch]*explore_time)**(self.beta)
                    W /= max_1d(W)

                    # Fit model
                    total_loss += self.model.fit(D_S[batch], D_NS[batch], D_A[batch], D_R[batch], D_T[batch], errors, W)

                    # Calculate new Errors for probabilities
                    D_E[batch] = errors**self.alpha + 1E-6

                    # Update target model
                    if self.tau >= 1.0 and int(t % self.tau) == 0:
                        self.model.hard_update_target_weights()

                # Update timestep
                s_t = s_t1
                t += 1

            print("Total Loss: {:.4f} | Total Score: {:.4f} | Epsilon: {:.5f} | Frames: {} | Time Taken: {:.2f}s".format(
                  total_loss, total_score, self.epsilon, t, time() - start_time))
            if episode % self.save_freq == 0:
                print("Saving Model to {}.j5".format(self.save_name))
                self.model.save_weights("{}.h5".format(self.save_name))

    cpdef play(self, UINT64_t num_episodes, BOOL_t train=True):
        return self.play_(num_episodes, train)


cpdef Agent MakeAgent(object game, tuple frame_size, UINT64_t num_actions, str save_name,
                      object build_model, object optimizer=optimizers.Adam(1E-6), object loss=huber_loss, BOOL_t double_dqn=False,
                      DOUBLE_t epsilon=0.1, DOUBLE_t delta_epsilon=0.00001, FLOAT_t gamma=0.99, FLOAT_t tau=1.0, UINT64_t frame_skip=0,
                      UINT64_t batch_size=64, UINT64_t frame_seq_count=1, UINT64_t memory=5000, UINT64_t save_freq=10,
                      FLOAT_t alpha = 0.7, FLOAT_t beta = 0.5):

    assert len(frame_size) == 3, "Frame size must be of the form (rows, cols, colors)."
    assert num_actions > 1, "Must have at least 2 actions for game"
    assert 1.0 >= epsilon >= 0.0, "Epsilon mut be a probability between 0 and 1"
    assert frame_skip >= 0
    assert frame_seq_count >= 1, "Must have at least 1 frame per iteration"

    channel_axis = 2 if K.image_dim_ordering() == "tf" else 0
    colors = frame_size[channel_axis]
    state_size = list(frame_size)
    state_size[channel_axis] = frame_seq_count * colors
    state_size = tuple(state_size)

    model = build_model(state_size, num_actions)

    try:
        model.load_weights('{}.h5'.format(save_name))
        print("Loaded previously trained model from {}.h5".format(save_name))
    except:
        pass

    return Agent(model, optimizer, loss, game, num_actions, frame_size, state_size, save_name, double_dqn,
                 epsilon, delta_epsilon, gamma, tau, frame_skip, batch_size, frame_seq_count, memory, save_freq, alpha, beta)