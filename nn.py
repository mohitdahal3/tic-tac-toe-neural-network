import numpy as np


def activationFunction(x):
    return 1 / (1 + np.exp(-x))



class NeuralNetwork():
    def __init__(self , num_input_nodes , num_hidden_nodes , num_output_nodes , learning_rate):
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes
        self.learning_rate = learning_rate

        # Initialize weights from a normal distribution centered at 0
        self.wih = np.random.normal(0.0, pow(self.num_hidden_nodes, -0.5), (self.num_hidden_nodes, self.num_input_nodes))
        self.who = np.random.normal(0.0, pow(self.num_output_nodes, -0.5), (self.num_output_nodes, self.num_hidden_nodes))


    def train(self , inputs_list , targets_list):
        input_layer = np.array(inputs_list , ndmin=2).transpose()
        targets = np.array(targets_list , ndmin=2).transpose()

        hidden_inputs = np.dot(self.wih , input_layer)
        hidden_outputs = activationFunction(hidden_inputs)

        final_inputs = np.dot(self.who , hidden_outputs)
        final_outputs = activationFunction(final_inputs)

        # ERROR = TARGET - ACTUAL (Simple difference, not squared for gradient)
        output_errors = targets - final_outputs
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
    
    def saveWeights(self, input_hidden_path, hidden_output_path):
        with open(input_hidden_path, 'w') as file:
            for row in self.wih:
                file.write(','.join(np.asarray(row, str)) + '\n')

        with open(hidden_output_path, 'w') as file:
            for row in self.who:
                file.write(','.join(np.asarray(row, str)) + '\n')

    def loadWeights(self, input_hidden_path, hidden_output_path):
        self.wih = np.loadtxt(input_hidden_path, dtype=np.float64, delimiter=',', ndmin=2)
        self.who = np.loadtxt(hidden_output_path, dtype=np.float64, delimiter=',', ndmin=2)
