import numpy as np


def activationFunction(x):
    return 1 / (1 + np.exp(-x))

def squareIt(x):
    return x * x

class NeuralNetwork():
    def __init__(self , num_input_nodes , num_hidden_nodes , num_output_nodes , learning_rate):
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes
        self.learning_rate = learning_rate

        self.wih = np.random.rand(num_hidden_nodes , num_input_nodes)
        self.who = np.random.rand(num_output_nodes , num_hidden_nodes)


    def train(self , inputs_list , targets_list):
        input_layer = np.array(inputs_list , ndmin=2).transpose()
        targets = np.array(targets_list , ndmin=2).transpose()

        hidden_inputs = np.dot(self.wih , input_layer)
        hidden_outputs = activationFunction(hidden_inputs)

        final_inputs = np.dot(self.who , hidden_outputs)
        final_outputs = activationFunction(final_inputs)

        output_errors = squareIt(targets - final_outputs)
        hidden_errors = np.dot(self.who.transpose() , output_errors)

        self.who += self.learning_rate * np.dot((output_errors * final_outputs * (1-final_outputs)) , hidden_outputs.transpose())
        self.wih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)) , input_layer.transpose())


    def query(self , inputs_list):
        input_layer = np.array(inputs_list , ndmin=2).transpose()

        hidden_inputs = np.dot(self.wih , input_layer)
        hidden_outputs = activationFunction(hidden_inputs)

        final_inputs = np.dot(self.who , hidden_outputs)
        final_outputs = activationFunction(final_inputs)

        return final_outputs
    
    def saveWeights(self):
        with open('weights_input_hidden.csv' , 'w') as file:
            for row in self.wih:
                file.write(','.join(np.asarray(row , str)) + '\n')
        

        with open('weights_hidden_output.csv' , 'w') as file:
            for row in self.who:
                file.write(','.join(np.asarray(row , str)) + '\n')

    def loadWeights(self):
        self.wih = np.loadtxt('weights_input_hidden.csv' , dtype=np.float64 , delimiter=',' , ndmin=2)
        self.who = np.loadtxt('weights_hidden_output.csv' , dtype=np.float64 , delimiter=',' , ndmin=2)

            


# nn = NeuralNetwork(9 , 6 , 9 , 0.3)



