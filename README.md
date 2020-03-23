# Reinforcement-Learning-Practice
机器人大创项目强化学习部分编程代码

主要包括：

natural-DQN

double-DQN

dueling-DQN

prioritized-DQN

dueling-DDQN

prioritized-DDQN

伪rainbow-DQN

待改进的points：

* e-greedy可以改为e不断衰减
* PDDQN中$w_{i}=\left(\frac{1}{N \cdot P_{\mathrm{PRB}}(i)}\right)^{\beta}$重要性采样系数可以让β 在训练开始时赋值为一个小于1 的数，然后随着训练迭代数的增加，让β数不断变大，并最终到达 1（变回replay buffer） 。这样我们既可以确保训练速度能够增加，又可以让模型的收敛性得到保证。
* 在硬件支持的情况下：a.加大记忆库的容量  b.减小learning rate但是不要太小 c.增大神经网络的规模：层数，神经元数目
* reward-engineering：修改为连续变化的reward：随着离目标点的距离的变化也可以获得reward，而不是仅仅在到达目标点才获得+1的reward（sparse reward）
* feature-engineering：创建能够反应state更多信息的feature；一般输入整个图像反应的信息较多，但是需要GPU加速支持

