import gym
import cv2
from collections import deque
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

""" creating GIF """
def save_frames_as_gif(frames, path='./', filename='Pong-v0_1.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def create_nn(input_shape, output_shape):
    net_input = Input(shape=input_shape)
    x = ConvLSTM2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding="same", return_sequences=True, data_format='channels_last')(net_input)
    x = ConvLSTM2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", return_sequences=True)(x)
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", return_sequences=False)(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    net_output = Dense(units=output_shape, activation="softmax")(x)

    OPTIMIZER = Adam(lr=0.00005)

    model = Model(inputs=net_input, outputs=net_output)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["categorical_accuracy"])

    return model

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    return img

def resize(img):
    height = 84
    width = 84
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def sequence(state, seq):
    for i in range(3):
        seq[i] = seq[i+1]
    seq[3] = state

    return seq

if __name__=="__main__":
    setting = "nn_seq_1500Examples_85Epochs"
    game = "Breakout-v0"
    env = gym.make(game)
    model = create_nn((4, 84, 84, 1), env.action_space.n)

    # choose correct folder to load weights from
    weights = {"Pong-v0": "2021-03-12_00-14-20_Pong-v0", "Breakout-v0": "2021-03-12_16-13-42_Breakout-v0", "MsPacman-v0": "2021-03-12_21-47-12_MsPacman-v0"}
    folder = weights[game]
    model.load_weights("../../results/" + folder + "/nn_seq_1500Examples_85Epochs.h5")


    gif = []  # for a GIF
    reward_list = []



    for episode in range(20):
        done = False
        episode_reward = 0.0
        state = env.reset()
        state = resize(state)
        state = grayscale(state)
        lives = 5 # just for Breakout-v0
        
        # initialize deque for framestack
        seq = []
        for i in range(4):
            seq.append(state)
        seq = np.asarray(seq)
        if game == "Breakout-v0":
            env.step(1)

        # playing a episode
        while not done:
            env.render()
            #gif.append(env.render(mode="rgb_array"))
            action = np.argmax(model.predict(np.expand_dims(seq, 0)))
            state, reward, done, info = env.step(action)
            if game == "Breakout-v0":
                print(info["ale.lives"])
                if info["ale.lives"] < lives:
                    lives = info["ale.lives"]
                    env.step(1)
            state = resize(state)
            state = grayscale(state)
            seq = sequence(state, seq)


        
        print("Episode:", episode, "\tReward:", episode_reward)
        reward_list.append(episode_reward)

    """ Saving reward_list to pandas dataframe """
    import pandas as pd 
    df = pd.DataFrame(reward_list, columns=["Rewards"])
    df.to_csv(game + ".csv")
    

    """ Saving GIF 
    print("Saving GIF")
    save_frames_as_gif(gif, filename=game + ".gif")
    print("Finished saving gif!")
    """