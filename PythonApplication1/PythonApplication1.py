import numpy
import scipy.special
import matplotlib.pyplot

class NeuralNetWork:

    #初始化函数
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        #初始化输入层、隐藏层、输出层节点、训练速率
        self.in_nodes = input_nodes
        self.hn_nodes = hidden_nodes
        self.ot_nodes = output_nodes
        self.lr = learning_rate

        #两个权重的矩阵
        self.wih = (numpy.random.rand(self.hn_nodes, self.in_nodes) - 0.5);
        self.who = (numpy.random.rand(self.ot_nodes, self.hn_nodes) - 0.5);

        #激活函数
        self.active_function = lambda x: scipy.special.expit(x);
    
    #训练函数
    def train(self, input_list, target_list):

        #计算输出
        inputs = numpy.array(input_list, ndmin = 2).T
        targets = numpy.array(target_list, ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.active_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.active_function(final_inputs)

        #结果对比，更新权重
        #输出层和隐藏层的误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #更新权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1-final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)), numpy.transpose(inputs))

    
    #查询函数
    def query(self, input_list):

        #将输入数组转化为二维的列表
        inputs = numpy.array(input_list, ndmin = 2).T

        #计算最后的输出矩阵
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.active_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.active_function(final_inputs)

        return final_outputs



#定义网络各层节点
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#定义学习率
learning_rate = 0.3

#创建神经网络对象
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#导入训练集
training_data_file = open("E:/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#训练测试集
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

#测试测试集

#导入测试集
test_data_file = open("E:/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#测试数据
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    #正确标签
    correct_label = int(all_values[0])
    print(correct_label, "correct_answer")
    #输入归一化
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    #查询网络
    outputs = n.query(inputs)
    #输出矩阵最大的值就是结果
    label = numpy.argmax(outputs)
    print(label, "network_answer")

    #匹配正确就加1
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

#输出最后正确率
scorecard_array = numpy.asarray(scorecard)
print(scorecard_array.sum() / scorecard_array.size)


print('a')