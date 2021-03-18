import gym
import cv2
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
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
    x = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding="same")(net_input)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = Dense(output_shape)(x)
    net_output = Activation("softmax")(x)

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

if __name__=="__main__":
    game = "MsPacman-v0"
    env = gym.make(game)
    model = create_nn((84, 84, 1), env.action_space.n)
    model.load_weights("../../results/2021-03-12_21-47-12_MsPacman-v0/nn_1500Examples_85Epochs.h5")

    frames = []  # for a GIF
    reward_list = []
  

    for episode in range(1):
        done = False
        episode_reward = 0.0
        state = env.reset()
        state = resize(state)
        state = grayscale(state)
        state = np.reshape(state, (1, 84, 84, 1))    
        state = state.astype("float32")

        while not done:
            #env.render()
            frames.append(env.render(mode="rgb_array"))
            action = np.argmax(model.predict(state))
            state, reward, done, info = env.step(action)
    
            episode_reward += reward
            state = resize(state)
            state = grayscale(state)
            state = np.reshape(state, (1, 84, 84, 1))
            state = state.astype("float32")


                    
        print("Episode:", episode, "\tReward:", episode_reward)
        reward_list.append(episode_reward)

    """ Saving frames to a GIF
    print("saving GIF")
    save_frames_as_gif(frames, path='./', filename=game + '.gif')
    print("Finished saving GIF!")
    """
    
    
    """ Saving reward_list to a pandas dataframe
    import pandas as pd 
    df = pd.DataFrame(reward_list, columns=["Rewards"])
    df.to_csv(game + setting + ".csv")
    save_frames_as_gif(frames, filename=game + ".gif")
    """