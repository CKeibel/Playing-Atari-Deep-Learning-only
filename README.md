Playing Atari without Reinforcement Learning. (Deep Learning only) <br>
Projektarbeit -> Proof of Concept für [Bachelorarbeit](https://github.com/CKeibel/Deep_Reinforcement_Learning)
<br>
<br>

Training info:
* saving 1500 game sequences with reward > 0
* train 85 Epochs
<br>
<br>

# Attempt 1 - resized-grayscale images
Preprocessing:
* Input resized to 84x84 pixels
* Input converted to grayscale images (84x84x1)
<br>

![Pong-v0](https://github.com/CKeibel/Playing-Atari-Deep-Learning-only/blob/main/models_in_action/01_resized_and_grayscale_imgs/Pong-v0.gif)
![Breakout-v0](https://github.com/CKeibel/Playing-Atari-Deep-Learning-only/blob/main/models_in_action/01_resized_and_grayscale_imgs/Breakout-v0.gif)
![MsPacman-v0](https://github.com/CKeibel/Playing-Atari-Deep-Learning-only/blob/main/models_in_action/01_resized_and_grayscale_imgs/MsPacman-v0.gif)
<br>
<br>

# Attempt 2 - resized-grayscale images
Preprocessing:
* Preprocessing from attempt 1
* stacking 4 images together to one input (84x84x4)
<br>

![Pong-v0](https://github.com/CKeibel/Playing-Atari-Deep-Learning-only/blob/main/models_in_action/02_framestack_model/Pong-v0.gif)
![Breakout-v0](https://github.com/CKeibel/Playing-Atari-Deep-Learning-only/blob/main/models_in_action/02_framestack_model/Breakout-v0.gif)
![MsPacman-v0](https://github.com/CKeibel/Playing-Atari-Deep-Learning-only/blob/main/models_in_action/02_framestack_model/MsPacman-v0.gif)

