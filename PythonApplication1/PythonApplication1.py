import numpy
import scipy.special

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
        pass
    
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

        pass
    
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
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

#定义学习率
learning_rate = 0.3

#创建神经网络对象
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.query([1, 2, 1])
