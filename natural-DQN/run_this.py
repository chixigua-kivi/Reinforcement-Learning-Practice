from maze_env1 import Maze
from RL_brain import DeepQNetwork
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import pickle

episodes = []
steps = []

def run_maze():
    step = 0    # 用来控制什么时候学习
    for episode in range(600):
        # 初始化环境
        observation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # DQN 根据观测值选择行为 epison_greedy策略
            action = RL.choose_action(observation)

            # 环境根据行为给出下一个 state, reward, 是否终止
            observation_, reward, done = env.step(action)

            # DQN 存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            #首先在200步之后才开始学习，之后每5步学习一次
            if (step > 500) and (step % 5 == 0):
                RL.learn()

            # 将下一个 state_ 变为 下次循环的 state
            observation = observation_

            # 如果终止, 就跳出循环
            if done:
                steps.append(step)
                episodes.append(episode)
                break
            step += 1   # 总步数

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions,
                      env.n_features,#observation/state 的属性，如长宽高
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                      memory_size=2000, # 记忆上限
                      # output_graph=True   # 是否输出 tensorboard 文件
                      )
    env.after(100, run_maze)#进行强化学习训练
    env.mainloop()

    #观看训练时间曲线
    his_natural = np.vstack((episodes, steps))

    file = open('his_natural.pickle', 'wb')
    pickle.dump(his_natural, file)
    file.close()

    plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
    plt.legend(loc='best')  # legend图例，其中’loc’参数有多种，’best’表示自动分配最佳位置
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()  # 显示网格线 1=True=默认显示；0=False=不显示
    plt.show()

    RL.plot_cost()  # 观看神经网络的误差曲线










