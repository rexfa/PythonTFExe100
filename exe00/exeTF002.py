import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 练习在神经网络里建立层
# 增加层函数，输入，输入值size，输出值size，激励函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    # 标定Weights的 输入输出shape
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    # biaase 的输入输出 shape
    biases =tf.Variable(tf.zeros([1,out_size])+0.1)
    # tf.matmul的乘法，这里还是规定了目标是一个线性函数,直线
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        # 执行激励函数
        outputs = activation_function(Wx_plus_b)
    return outputs

# 创造数据，同时适当加入一些噪音
# 在线性空间内取-1到-1 的300的x,x_data.shape => (300,1)
# 注意最后的中括号没有的情况x_data.shape => (300,)，实际上是增加一个维度
x_data = np.linspace(-1,1,300)[:,np.newaxis]
#噪音0到0.05 容量结构和x_data一样
noise = np.random.normal(0,0.05,x_data.shape)
#x为平方和上一个例子不一样，baise为0.5，加入了噪音扰动
y_data = np.square(x_data)+0.5+noise

#Show data
##plt.scatter(x_data,y_data)
##plt.show()

# 为网络输入定义 占位符
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
# 增加隐藏层
layer1 = add_layer(xs,1,10,tf.nn.relu)
# 增加输出层 注意构建时 层间输出输入数量要匹配
prediction = add_layer(layer1,10,1,None)

# 定义损失函数 夹在输出层和数据之间
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#创建会话
session = tf.Session()
init = tf.initialize_all_variables()
session.run(init)

#画图
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

# 训练
for i in range(2000):
    session.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 ==0:
        try:
            ax.line.remove(lines[0])
        except Exception:
            pass
        prediction_value = session.run(prediction,feed_dict={xs:x_data})

        #显示预测
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(1)