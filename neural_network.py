import numpy as np
import numpy.typing as npt

class NeuralNetwork:
    def __init__(self, *neuron_num_list: list[int]):
        self.learning_rate = 0.05
        self.weights = []
        self.biases = []
        self.activations = []
        self.z = []
        self.n_layers = len(neuron_num_list)
        
        for i, num in enumerate(neuron_num_list):
            try:
                self.weights.append(np.random.randn((neuron_num_list[i+1], num)))
            except IndexError:
                print(f"\n\nWeights -- Out of range in iteration: {i} ({i - len(neuron_num_list)})\n\n")

            try:
                self.biases.append(np.random.randn(neuron_num_list[i+1]))
            except IndexError:
                print(f"\n\nBiases -- Out of range in iteration: {i} ({i - len(neuron_num_list)})\n\n")
            
            try:
                self.z.append(np.ndarray(neuron_num_list[i+1]))
            except IndexError:
                print(f"\n\nZ -- Out of range in iteration: {i} ({i - len(neuron_num_list)})\n\n")
            
            self.activations.append(np.ndarray(num))
            
        
        self.weights = np.asarray(self.weights, dtype=object)
        self.biases = np.asarray(self.biases, dtype=object) * 2 - 1
        self.activations = np.asarray(self.activations, dtype=object)
        
        self.z = np.asarray(self.z, dtype=object)
        self.gradient = np.array([
            self.weights.copy(),
            self.biases.copy()
        ])
    
    def del_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def del_cost(self, n_layer: int, required: npt.NDArray):
        return 2 * (self.activations[n_layer] - required)
    
    def apply_gradient(self, batch_size):
        if not batch_size:
            return
        self.gradient /= batch_size
        self.weights -= self.learning_rate * self.gradient[0]
        self.biases -= self.learning_rate * self.gradient[1]
    
    def train(self, features: npt.NDArray, labels: npt.NDArray, batch_size: int, label_vectors: list):
        print('\n\n--------------------------------')
        print('-----------Training-------------')
        print('--------------------------------\n\n')
        for i, (f, l) in enumerate(zip(features, labels)):
            self._calc_activations(f)
            self.gradient += self.backprop(label_vectors[l])
            if i % batch_size == 0:
                self.apply_gradient(batch_size)
        self.apply_gradient(len(features) % batch_size)
                
            
    def backprop(self, required: npt.NDArray, n_layer: int = -1):
        gradient_single = self.gradient.copy()
        if -n_layer > self.n_layers:
            return gradient_single
        
        # Adjusting Biases
        
        del_b = self.del_sigmoid(self.z[n_layer]) * self.del_cost(n_layer, required)
        print(n_layer)
        print(del_b.shape)
        print(gradient_single[1][n_layer].shape)
        gradient_single[1][n_layer] += del_b
        
        # Adjusting Activations
        try:
            print(self.weights[n_layer])
            print(del_b)
            gradient_single += self.backprop(np.dot(self.weights[n_layer], del_b), n_layer-1)
        except IndexError:
            pass
        
        # Adjusting Weights
        del_w = self.activations[n_layer - 1] * del_b
        gradient_single[0][n_layer] += del_w
        
        return gradient_single
        

    
    def _calc_activations(self, input: npt.NDArray):
        self.activations[0] = input
        for i, a in enumerate(self.activations[:-1]):
            self.z[i] = np.dot(self.weights[i], a) + self.biases[i]
            self.activations[i+1] = self.sigmoid(self.z[i])
    
    def predict(self, input: npt.NDArray):
        self._calc_activations(input)
        return np.argmax(self.activations[-1])