### Directory structure
- CS3316-RL
  - Policy-based
    - visualization
    - DDPG.py
    - run.py
    - run.sh
    - TD3.py
    - utils.py
  - Value-based
    - DQN
      - run.py
      - baselines
      - utils.py
    - Dueling_DQN
      - run.py
      - baselines
      - utils.py
  - Final_report.pdf

### Training Environment
- Operating system: Linux
- GPU: NVIDIA GeForce RTX 3090 
- Python: 3.7.16
- Pytorch: 1.10.1
- mujoco_py: 2.1.2.14
- Gym: 0.19.0
- Atari: 0.2.6


It's recommended to run the code on a **linux** environment, and the code is tested on Ubuntu 18.04.5 LTS. The code is also tested on Macos, but the performance is not as good as on Linux.

### Installation
please refer to the [official website](https://gym.openai.com/docs/) of OpenAI Gym for the installation of Gym.

for the installation of mujoco_py, look up the [guideline](https://github.com/openai/mujoco-py), note that the it depends on the mujoco library, which is opensourced on the [website](https://www.roboti.us/index.html). 

For the atari game, you can install the atari game by:
```pip install gym[atari]```

**Note:** the value-based methods relies on ```cpprb``` to create replay buffer and the ```cv2``` to process the atari environment. You can install them by: 
```pip install cpprb``` 
```pip install opencv-python```





### Run in Mujoco environment
To run in *mujoco* environment with policy-based methods, you can run the code by:
1. Enter the directory Policy-based
2. run the code with 
```python3 run.py --env_name $env --agent $agent```

- You can test on the following environments:
  - Hopper-v2
  - Humanoid-v2
  - HalfCheetah-v2
  - Ant-v2
and other environments in the [mujoco](https://www.gymlibrary.dev/environments/mujoco/).

- The availabel agents are "DDPG" and "TD3".

3. By default, the results will be saved in the directory ```results/``` and the model to ```models/```. Specify the directories after ```--model_path``` and ```--res_dir```  **without the '/' after the directory**. 

The entire list of **hyperparameters** can be found in the ```run.py``` file.

If the program operates normally, you will see the following information:
```
---------------------------------------
Starting training: Hopper-v2
---------------------------------------
Total T: 30 Episode Num: 0 Episode T: 30 Reward: 22.64282383453175
Total T: 58 Episode Num: 1 Episode T: 28 Reward: 31.71377777927209
Total T: 82 Episode Num: 2 Episode T: 24 Reward: 25.424505413498558
Total T: 102 Episode Num: 3 Episode T: 20 Reward: 19.00467753690162
Total T: 120 Episode Num: 4 Episode T: 18 Reward: 18.578618839868945
Total T: 134 Episode Num: 5 Episode T: 14 Reward: 6.437612635849992
Total T: 165 Episode Num: 6 Episode T: 31 Reward: 12.21320158286492
Total T: 180 Episode Num: 7 Episode T: 15 Reward: 12.767043940349637
```

**Note:** for the convenience to run the two agents in all four environments, you can directly use the following command:
```bash run.sh```

### Run the Atari task
To run the *Atari* game with value-based methods, you can run the code by:
1. Enter the directory Value-based
2. Enter the directory of the specific algorithm you want to run （DQN or Dueling_DQN）
3. run the code with the following command 
```python3 run.py --env $env```
- You can test on the following environments:
  - VideoPinball-ramNoFrameskip-v4
  - BreakoutNoFrameskip-v4
  - PongNoFrameskip-v4
  - BoxingNoFrameskip-v4
and other environments in the [Atari](https://www.gymlibrary.dev/environments/atari/).

4. By default, the results of DQN will be saved in the directory ```DQN_result/```, and the model to ```DQN_model/```, and the results of Dueling DQN will be saved in the directory ```Duel_DQN_result/```, and the model to ```Duel_DQN_model/``` specify the directory you want after the ```--data_path``` and ```--model_path```  and **Add a '/' after the directory**. Besides, you can specify the ```--print_freq``` as 1 to see the instantaneous results, or to 100 to reduce the information displayed. 
   
The entire list of **hyperparameters** can be found in the ```run.py``` files.

If the program operates normally, you will see the following information:
```
---------------------------------------
Starting training: BoxingNoFrameskip-v4
---------------------------------------
Episode: 1 Reward: 3.0
Episode: 2 Reward: -2.0
Episode: 3 Reward: -15.0
Episode: 4 Reward: -3.0
Episode: 5 Reward: -6.0
Episode: 6 Reward: -2.0
Episode: 7 Reward: 4.0
Episode: 8 Reward: -3.0
```

