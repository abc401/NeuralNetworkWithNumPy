from neural_network import NeuralNetwork
import numpy as np
from mnist import MNIST
import logging
import os

def make_label_vector(labels: list):
    label_vectors = {l: np.zeros(len(labels)) for l in labels}
    for i, label_vector in enumerate(label_vectors.values()):
        label_vector[i] = 1.0
    
    return label_vectors

def main():
    mndata = MNIST('data')
    
    log.info("Loading data.")
    
    training_images, training_labels = mndata.load_training()
    testing_images, testing_labels = mndata.load_testing()
    training_images, training_labels = np.asarray(training_images), np.asarray(training_labels)
    testing_images, testing_labels = np.asarray(testing_images), np.asarray(testing_labels)
    
    log.info("Creating labels.")
    
    labels = [i for i in range(10)]
    nn = NeuralNetwork(784, 16, 17, 10)
    
    log.info("Training Network.")
    
    nn.train(training_images, training_labels, 100, make_label_vector(labels))
    
    log.info("Training Done.")
    
    label = labels[nn.predict(testing_images[0])]
    print(label, labels[0])



if __name__ == '__main__':
    log = logging.getLogger("log")
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
