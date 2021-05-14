import numpy
import scipy.special

class NeuralNetWork:

    #��ʼ������
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        #��ʼ������㡢���ز㡢�����ڵ㡢ѵ������
        self.in_nodes = input_nodes
        self.hn_nodes = hidden_nodes
        self.ot_nodes = output_nodes
        self.lr = learning_rate

        #����Ȩ�صľ���
        self.wih = (numpy.random.rand(self.hn_nodes, self.in_nodes) - 0.5);
        self.who = (numpy.random.rand(self.ot_nodes, self.hn_nodes) - 0.5);

        #�����
        self.active_function = lambda x: scipy.special.expit(x);
        pass
    
    #ѵ������
    def train(self, input_list, target_list):

        #�������
        inputs = numpy.array(input_list, ndmin = 2).T
        targets = numpy.array(target_list, ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.active_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.active_function(final_inputs)

        #����Աȣ�����Ȩ��
        #���������ز�����
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #����Ȩ��
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1-final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)), numpy.transpose(inputs))

        pass
    
    #��ѯ����
    def query(self, input_list):

        #����������ת��Ϊ��ά���б�
        inputs = numpy.array(input_list, ndmin = 2).T

        #���������������
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.active_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.active_function(final_inputs)

        return final_outputs



#�����������ڵ�
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

#����ѧϰ��
learning_rate = 0.3

#�������������
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.query([1, 2, 1])
