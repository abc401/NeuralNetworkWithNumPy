import numpy as np
import numpy.typing as npt
from math import ceil

class NeuralNetwork:
    def __init__(self, *neuron_num_list: list[int]):
        self.learning_rate = 0.05
        self.weights = []
        self.biases = []
        self.activations = []
        self.z = []
        
        for i, num in enumerate(neuron_num_list):
            try:
                self.weights.append(np.random.random((neuron_num_list[i+1], num)))
            except IndexError:
                print(f"\n\nWeights -- Out of range in iteration: {i} ({i - len(neuron_num_list)})\n\n")

            try:
                self.biases.append(np.random.random(neuron_num_list[i+1]))
            except IndexError:
                print(f"\n\nBiases -- Out of range in iteration: {i} ({i - len(neuron_num_list)})\n\n")
            
            self.z.append(np.ndarray(num))
            self.activations.append(np.ndarray(num))
            
        
        self.weights = np.asarray(self.weights, dtype=object)
        self.biases = np.asarray(self.biases, dtype=object) * 2 - 1
        self.activations = np.asarray(self.activations, dtype=object)
        self.z = np.asarray(self.z, dtype=object)
        self.gradient = np.array([
            np.array(self.weights.shape()),
            np.array(self.biases.shape())
        ])
    
    def del_sigmoid(self, x):
        self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def _del_cost(self, n_layer: int, required: npt.NDArray):
        return 2 * (self.activations[n_layer] - required)
    
    def apply_gradient(self, batch_size):
        self.gradient /= batch_size
        self.weights -= self.learning_rate * self.gradient[0]
        self.biases -= self.learning_rate * self.gradient[1]
    
    def train(self, features: npt.NDArray, labels: npt.NDArray, batch_size: int):
        for i, f in enumerate(features):
            self._calc_activations(f)
            self.gradient += self.backprop(labels[i])
            if i % batch_size == 0:
                self.apply_gradient(batch_size)
        self.apply_gradient(batch_size)
                
            
    def backprop(self, required: npt.NDArray, n_layer: int = -1):
        gradient_single = np.zeros(self.gradient.shape())
        
        # Adjusting Biases
        del_b = map(self.del_sigmoid, self.z) * self._del_cost(n_layer, required)
        gradient_single[1] += del_b
        
        # Adjusting Activations
        try:
            gradient_single += self.backprop(self.weights[n_layer] * del_b, n_layer-1)
        except IndexError:
            pass
        
        # Adjusting Weights
        del_w = self.activations[n_layer - 1] * del_b
        gradient_single[0] += del_w
        
        return gradient_single
        

    
    def _calc_activations(self, input: npt.NDArray):
        self.activations[0] = input
        for i, a in enumerate(self.activations[1:]):
            self.z[i] = self.weights[i] * self.activations[i-1] + self.biases[i]
            a = self.sigmoid(self.z[i])
    
    def predict(self, input: npt.NDArray):
        self._calc_activations(input)
        return self.activations[-1]