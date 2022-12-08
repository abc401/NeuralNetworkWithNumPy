from neural_network import NeuralNetwork
import numpy as np

def main():
    nn = NeuralNetwork(2, 3, 4)
    print(nn.weights, end="\n\n")
    print(nn.biases, end="\n\n")
    print(nn.activations, end="\n\n")
    print(nn.weights[-1])

if __name__ == '__main__':
    main()
