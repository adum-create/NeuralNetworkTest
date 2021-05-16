import numpy
import scipy.special
import matplotlib.pyplot

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
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#����ѧϰ��
learning_rate = 0.3

#�������������
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#����ѵ����
training_data_file = open("E:/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#ѵ�����Լ�
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

#���Բ��Լ�

#������Լ�
test_data_file = open("E:/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#��������
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    #��ȷ��ǩ
    correct_label = int(all_values[0])
    print(correct_label, "correct_answer")
    #�����һ��
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    #��ѯ����
    outputs = n.query(inputs)
    #�����������ֵ���ǽ��
    label = numpy.argmax(outputs)
    print(label, "network_answer")

    #ƥ����ȷ�ͼ�1
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

#��������ȷ��
scorecard_array = numpy.asarray(scorecard)
print(scorecard_array.sum() / scorecard_array.size)


print('a')