import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(1)
tf.set_random_seed(1)

class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,#observation/state 的属性，如长宽高
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            dueling=True,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features #observation/state 的属性，如长宽高
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy  # epsilon 的最大值
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon_increment = e_greedy_increment  # epsilon 的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # 是否开启探索模式, 并逐步减少探索次数,e_greedy_increment=None-->self.epsilon = 0,
        # e_greedy_increment!=None-->self.epsilon=self.epsilon_max
        # TODO(xhx):探索模式后续如何启动？

        self.dueling = dueling  # decide to use dueling DQN or not

        # 记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0

        # 初始化全 0 记忆 [s, a, r, s_]
        # size = s特征数+s_特征数+a(0/1/2/3)+r
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # 和视频中不同, 因为 pandas 运算比较慢, 这里改为直接用 numpy

        # 创建 [target_net, evaluate_net]
        self._build_net()

        # 替换 target net 的参数
        # TODO(xhx):替换参数这四行代码没看懂
        #在build_net中，各自的w1,b1,w2,b2都放进collection 'target_net_params' 和'eval_net_params'
        t_params = tf.get_collection('target_net_params')  # 提取 target_net 的参数
        e_params = tf.get_collection('eval_net_params')  # 提取  eval_net 的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  # 更新 target_net 参数
        self.sess = tf.Session()

        # 输出 tensorboard 文件
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # 记录所有 cost 变化, 用于最后 plot 出来观看


    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):  # 第一层, 两种 DQN 都一样
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):  # 专门分析 state 的 Value  #所有动作都是一个数值self.V
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):  # 专门分析每种动作的 Advantage
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                ##关键部分！！！##
                with tf.variable_scope('Q'):  # 合并 V 和 A, 为了不让 A 直接学成了 Q, 我们减掉了 A 的均值 (self.A - tf.reduce_mean)
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q'):  # 普通的 DQN 第二层
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out
        # -------------- 创建 eval 神经网络, 及时提升参数 --------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # 用来接收 q_target 的值, 这个之后会通过计算得到
        # c_names(collections_names) 是在更新 target_net 参数时会用到
        #定义W,b的初始值
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'): # 求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):    # 梯度下降
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------------- 创建 target 神经网络, 提供 target Q ---------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # 接收下个 observation
        with tf.variable_scope('target_net'):
            #c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        #target_net 不需要求loss，然后train，只有有结构就行，直接使用eval_net的参数
        #target_net 不需要求loss，然后train，只有有结构就行，直接使用eval_net的参数

    def choose_action(self,observation):
        # 统一 observation 的 shape (1, size_of_observation)
        # 讲其从数组升级为矩阵/向量形式
        observation = observation[np.newaxis, :]
        #epsilon-greedy策略
        #选max_q
        if np.random.uniform() < self.epsilon:
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            #####################################DDQN改造#################################################
            ##用来画Q值变化
            # if not hasattr(self,'q'):# 记录选的 Qmax 值,让它记录下每次选择的 Q 值.
            #     self.q=[]
            #     self.running_q=0
            #     self.running_q=self.running_q*0.99+0.01*np.max(actions_value)
            #####################################DDQN改造#################################################
        #随机选
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self,s, a, r, s_):
        ##TODO:这两句是干啥的？
        #用来旧memory就被新memory替换之后重新计数用的
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 记录一条 [s, a, r, s_] 记录
        # horizontal stack左右合并成一个数组,便于存储
        transition = np.hstack((s, [a, r], s_))

        # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # 替换过程

        self.memory_counter += 1

    def learn(self):
        # 检查是否替换 target_net 参数
        # 每过 self.replace_target_iter次学习，就replace一下
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 从 memory 中随机抽取 batch_size 这么多记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        #####################################DDQN改造#################################################

        # q_next, q_eval = self.sess.run(
        #     [self.q_next, self.q_eval],
        #     feed_dict={
        #         self.s_: batch_memory[:, -self.n_features:],  # next observation
        #         self.s: batch_memory[:, :self.n_features]  # observation
        #     })

        #######DDQN#######
        q_next, q_evalxx = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],  # next observation
                       self.s: batch_memory[:, -self.n_features:]})  # next observation
        #######DDQN#######
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})# now observation

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        #######DDQN#######
        max_action_qevalxx = np.argmax(q_evalxx,axis=1) # q_eval 得出的最高奖励动作
        select_q_max_next = q_next[batch_index,max_action_qevalxx]
        q_target[batch_index, eval_act_index] = reward + self.gamma * select_q_max_next
        #######DDQN#######

        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        #####################################DDQN改造#################################################

        """
               假如在这个 batch 中, 我们有2个提取的记忆, 根据每个记忆可以生产3个 action 的值:
               q_eval =
               [[1, 2, 3],
                [4, 5, 6]]

               q_target = q_eval =
               [[1, 2, 3],
                [4, 5, 6]]

               然后根据 memory 当中的具体 action 位置来修改 q_target 对应 action 上的值:
               比如在:
                   记忆 0 的 q_target 计算值是 -1, 而且我用了 action 0;
                   记忆 1 的 q_target 计算值是 -2, 而且我用了 action 2:
               q_target =
               [[-1, 2, 3],
                [4, 5, -2]]

               所以 (q_target - q_eval) 就变成了:
               [[(-1)-(1), 0, 0],
                [0, 0, (-2)-(6)]]

               最后我们将这个 (q_target - q_eval) 当成误差, 反向传递会神经网络.
               所有为 0 的 action 值是当时没有选择的 action, 之前有选择的 action 才有不为0的值.
               我们只反向传递之前选择的 action 的值,
               """
        # 训练 eval_net
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)  # 记录 cost 误差

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        file = open('self.cost_his_dueling_DDQN.pickle', 'wb')
        pickle.dump(self.cost_his, file)
        file.close()


        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()








