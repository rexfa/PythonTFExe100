import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 创建线性的测试数据
#start
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+3
# 创建一个structure（结构体） 在已知x和y是线性关系后建立目标。定义weight斜率与biase偏差
# tf.Variable创建TF的变量
# random_uniform 控制变量，随机取值于-1.0 到 1.0之间容量为1的数组
# tf.zeros 取一个0的容量为1的数组
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
# 确定y和x_data和目标 Weights和 biases的线性关系 函数
y=Weights*x_data + biases 
# deviation (偏差) and optimizer（优化）
# 损失，降低平均值，就是降低方差的平均值
# 定义样本偏差 loss ，求的是y和y_data的方差，目标是做到loss 偏差最小
# ps 总喜欢说 差方，很奇怪啊我
loss = tf.reduce_mean(tf.square(y-y_data))
# 优化函数是 tf训练里的梯度下降 步长是0.5,就是用迭代误差函数的导数，尽可能找到最低值
# 步长选择很重要，步子太大可能迭代后误差反而变得更大，步子太小，计算量又会太大
# 另外初始值也很重要，如果初始值接近则很快
# 而在复杂的模型里，可能会有多个最小梯度的存在，这时初始值更加重要
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练目标就是不断优化到损失最小
train = optimizer.minimize(loss)
# 建立tf的初始化
init = tf.global_variables_initializer()
#end

#session 会话
session = tf.Session()
session.run(init)

# train 训练
# display 建立一个坐标系  x 0-1 y 0-1区间
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
t = np.arctan2(x_data,y_data)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for step in range(201):
    # 迭代进行训练
    session.run(train)
    if step%20 ==0:
        xp = np.linspace(0,1,3)
        yp = xp *session.run(Weights)+session.run(biases)
        ax.plot(xp,yp)
        # 显示 过程需要在会话里加run
        print(step,session.run(Weights),session.run(biases))
        plt.pause(1)
