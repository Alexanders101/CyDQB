# CyDQN
Cython and Keras based Deep-Q Network for training nerual networks to play games and designed to use the OpenAI gym api.

## Requirements
- cython
- numpy
- keras
- tensorflow

In order to run example
- gym
- skimage

## Installation 
Run build.sh in order to compile the cython code. (build_2.sh is for python 2, this is not as guaranteed to succeed)


## Usage
The main functionality inside of the library is the MakeAgent function.

In order to play a game, the user must provide several things to the constructor
- build_model function: function with the api (state_size, number_of_actions) -> Keras Model, 
this will be called when constructing your Keras model in order to perform the training. The state size is the size
of an individual frame / group of frames inside of a game, and the number of actions is the output size of the network.

- game class: Class very similar to the GYM api for games, it must implement two functions: 
step(action, value) -> (observation, score, done) 
and reset() -> observation.

- frame_size : Tuple of the shape of a single frame (height, width, colors). (must be 3 dimensional)
- num_actions : number of possible actions to perform
- save_name : Base name for any model or log files to be saved. (Model will automatically load any previous model with the same save_name.)
- frame_seq_count : How many frames to stack on top of each other.
- save_freq : Number of episodes before saving model.
- memory : Size of memory bank for previous observations. This is also the delay before the model begins training as it requires a full memory before the model will start the training process.
- epsilon, delta_epsilon : probability of selecting a random action instead of predicted one, and the change every episode.
- gamma : Damping factor of previous observations.
- batch_size : Size of training batch, reduce if you run out of memory.
- tau : Factor of target model updates. if tau > 1, then its the amount of episodes before updating target model. if tau < 1, then its the weight of the update.
- optimizer : Either a keras or tensorflow optimizer.
- double_dqn : Whether or not to use a double_dqn model instead of single.

In order to play the game, call agent.play with the following parameters
- num_episodes : Number of episodes to train / play for.
- train : Whether or not to train or just observe.


## Example
Example usage is demonstrated in the pacman.py file in the head directory. This has an example of the model training
on the Ms. Pacman atari game with a simple or recurrent model.
