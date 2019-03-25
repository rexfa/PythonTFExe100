import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 显示搭建图纸 

# 增加层函数，输入值，输入值大小，输出值大小，激励函数
# 增加图纸中的名称标记
def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


#make target data with some noise
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)+0.5+noise

# Define placeholder for inputs to the network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_in')
    ys = tf.placeholder(tf.float32,[None,1],name='y_in')

# Add a hide layer
l1 = add_layer(xs,1,10,tf.nn.relu)
# Add an output layer
prediction = add_layer(l1,10,1,None)

# The lose between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Session
session = tf.Session()

# writer logs
writer = tf.summary.FileWriter("logs/",session.graph)

inti = tf.initialize_all_variables()
session.run(inti)

# draw the real data              
fig1 = plt.figure()
ax =fig1.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

# train 2000 times
for i in range(2000):
    session.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = session.run(prediction,feed_dict={xs:x_data})

        # show the prediction
        lines= ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(1)