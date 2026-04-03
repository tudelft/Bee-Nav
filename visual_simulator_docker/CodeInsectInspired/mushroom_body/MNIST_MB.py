import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from matplotlib import pyplot as plt
import json
from simple_network import SimpleCNN, CompactCNN_rgb, IncepAttentionCNN_rgb
import torch
from torchvision import datasets, transforms

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_set, test_set, train_loader, test_loader


def train_MBON_MNIST(n_epochs = 1, KC_dimension = 5000, n_nonzero_weights = 4, train_set=None, test_set=None, train_loader=None, test_loader=None):

    input_dimension = 28 * 28  # MNIST image size
    n_classes = 10

    if train_loader is None or test_loader is None:
        train_set, test_set, train_loader, test_loader = get_data_loaders(batch_size=64)

    # Instantiate weights randomly, allowing for different numbers of non-zero weights
    # PN_to_KC_weights = np.random.choice([0, 1], size=(KC_dimension, input_dimension), p=[0.99, 0.01])
    # Instantiate weights with a fixed number of non-zero weights per KC neuron
    PN_to_KC_weights = np.zeros((KC_dimension, input_dimension), dtype=np.float32)
    PN_to_KC_weights[:n_nonzero_weights, :] = 1
    for col in np.arange(input_dimension):
        np.random.shuffle(PN_to_KC_weights[:, col])
    # Instantiate KC to MBON weights
    KC_to_MBON_weights = np.ones((n_classes, KC_dimension))
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}:")
        for batch_idx, (data, target) in enumerate(train_loader):
            if np.mod(batch_idx, 50) == 0:
                print('.', end='')
            batch_size = data.shape[0]
            data_np = data.numpy().reshape(batch_size, -1) 
            # map to PN representation (here we just flatten the image)
            PN_representation = data_np
            # map to KC representation
            KC_representation = (PN_to_KC_weights @ PN_representation.T).T
            
            # Train MBONs
            for i in range(batch_size):
                label = target[i].item()
                KC_vector = KC_representation[i]
                # Set the weights to the most active KC neurons to zero for this label:
                # determine the top 50 indices:
                top_indices = np.argsort(KC_vector)[-50:]
                KC_to_MBON_weights[label, top_indices] = 0
        
    # At the end, evaluate on the test set
    correct_MBON = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        batch_size = data.shape[0]
        data_np = data.numpy().reshape(batch_size, -1) 
        PN_representation = data_np
        KC_representation = (PN_to_KC_weights @ PN_representation.T).T
        
        for i in range(batch_size):
            KC_vector = KC_representation[i]

            # Compute MBON activations
            MBON_activations = KC_to_MBON_weights @ KC_vector
            predicted_label = np.argmin(MBON_activations)
            true_label = target[i].item()
            if predicted_label == true_label:
                correct_MBON += 1

            total += 1
    performance = correct_MBON / total
    
    print(f"\nTest Accuracy MBON: {correct_MBON}/{total} = {(correct_MBON/total)*100:.2f}%")
    
    return performance

def train_KC_NN_MNIST(n_epochs = 1, KC_dimension = 5000, n_nonzero_weights = 4, train_set=None, test_set=None, train_loader=None, test_loader=None):

    input_dimension = 28 * 28  # MNIST image size
    n_classes = 10

    if train_loader is None or test_loader is None:
        train_set, test_set, train_loader, test_loader = get_data_loaders(batch_size=64)

    # Instantiate weights randomly, allowing for different numbers of non-zero weights
    # PN_to_KC_weights = np.random.choice([0, 1], size=(KC_dimension, input_dimension), p=[0.99, 0.01])
    # Instantiate weights with a fixed number of non-zero weights per KC neuron
    PN_to_KC_weights = np.zeros((KC_dimension, input_dimension), dtype=np.float32)
    PN_to_KC_weights[:n_nonzero_weights, :] = 1
    for col in np.arange(input_dimension):
        np.random.shuffle(PN_to_KC_weights[:, col])
    
    NN_database = [[] for _ in range(10)]  # list of lists to store KC representations per class

    performance = np.zeros((n_epochs, 1))  # columns: NN accuracy
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Training Epoch {epoch}, Batch {batch_idx}")
            batch_size = data.shape[0]
            data_np = data.numpy().reshape(batch_size, -1) 
            # map to PN representation (here we just flatten the image)
            PN_representation = data_np
            # map to KC representation
            KC_representation = (PN_to_KC_weights @ PN_representation.T).T
            
            for i in range(batch_size):
                label = target[i].item()
                KC_vector = KC_representation[i]
                NN_database[label].append(KC_vector)
        
        # At the end of each epoch, evaluate on the test set
        correct_NN = 0
        total = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            print(f"Testing Batch {batch_idx}")
            batch_size = data.shape[0]
            data_np = data.numpy().reshape(batch_size, -1) 
            PN_representation = data_np
            KC_representation = (PN_to_KC_weights @ PN_representation.T).T
            
            for i in range(batch_size):
                KC_vector = KC_representation[i]
                min_distance_per_class = []
                for c in range(10):
                    # determine the closest match in the NN_database for class c
                    min_distance = float('inf')
                    for stored_KC in NN_database[c]:
                        distance = np.linalg.norm(KC_vector - stored_KC)
                        if distance < min_distance:
                            min_distance = distance
                    min_distance_per_class.append(min_distance)
                predicted_label_NN = np.argmin(min_distance_per_class)
                if predicted_label_NN == true_label:
                    correct_NN += 1
                total += 1

        performance[epoch, 1] = correct_NN / total
        # store the KC dimension and performance over epochs to a json file:
        with open(f'mb_mnist_performance_kc{KC_dimension}_nw{n_nonzero_weights}_rseed{rseed}_NN.json', 'w') as f:
            json.dump({'KC_dimension': KC_dimension, 'performance': performance[:epoch+1, :].tolist()}, f)

        print(f"Test Accuracy NN: {correct_NN}/{total} = {correct_NN/total*100:.2f}%")


def train_KC_readout_MNIST(n_epochs = 1, KC_dimension = 5000, n_nonzero_weights = 4, train_set=None, test_set=None, train_loader=None, test_loader=None):

    input_dimension = 28 * 28  # MNIST image size
    n_classes = 10

    if train_loader is None or test_loader is None:
        train_set, test_set, train_loader, test_loader = get_data_loaders(batch_size=64)

    # Instantiate weights randomly, allowing for different numbers of non-zero weights
    # PN_to_KC_weights = np.random.choice([0, 1], size=(KC_dimension, input_dimension), p=[0.99, 0.01])
    # Instantiate weights with a fixed number of non-zero weights per KC neuron
    PN_to_KC_weights = np.zeros((KC_dimension, input_dimension), dtype=np.float32)
    PN_to_KC_weights[:n_nonzero_weights, :] = 1
    for col in np.arange(input_dimension):
        np.random.shuffle(PN_to_KC_weights[:, col])
    
    # super simple linear model for reading out KCs
    model = torch.nn.Linear(KC_dimension, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    performance = np.zeros((n_epochs, 1))  # columns: MBON accuracy, NN accuracy
    for epoch in range(n_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Training Epoch {epoch}, Batch {batch_idx}")
            batch_size = data.shape[0]
            data_np = data.numpy().reshape(batch_size, -1) 
            # map to PN representation (here we just flatten the image)
            PN_representation = data_np
            # map to KC representation
            KC_representation = (PN_to_KC_weights @ PN_representation.T).T
            
            # Train simple linear model
            inputs_tensor = torch.tensor(KC_representation, dtype=torch.float32)
            target_tensor = target
            optimizer.zero_grad()
            outputs = model(inputs_tensor)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            optimizer.step()

        # At the end of each epoch, evaluate on the test set
        correct_linear = 0
        total = 0
        model.eval()
        for batch_idx, (data, target) in enumerate(test_loader):
            print(f"Testing Batch {batch_idx}")
            batch_size = data.shape[0]
            data_np = data.numpy().reshape(batch_size, -1) 
            PN_representation = data_np
            KC_representation = (PN_to_KC_weights @ PN_representation.T).T
            
            for i in range(batch_size):
                KC_vector = KC_representation[i]

                # compute network output:
                inputs_tensor = torch.tensor(KC_vector, dtype=torch.float32).unsqueeze(0)
                outputs = model(inputs_tensor)
                _, predicted_label_NN = torch.max(outputs, 1)
                predicted_label_NN = predicted_label_NN.item()
                true_label = target[i].item()
                if predicted_label_NN == true_label:
                    correct_linear += 1

                total += 1

        performance[epoch, 0] = correct_linear / total
        # store the KC dimension and performance over epochs to a json file:
        with open(f'mb_mnist_performance_kc{KC_dimension}_nw{n_nonzero_weights}_rseed{rseed}_linear_ANN.json', 'w') as f:
            json.dump({'KC_dimension': KC_dimension, 'performance': performance[:epoch+1, :].tolist()}, f)

        print(f"Test Accuracy Linear Model: {correct_linear}/{total} = {correct_linear/total*100:.2f}%")

def train_CNN_MNIST(n_epochs = 25, network_type = 'attention'):
    train_set, test_set, train_loader, test_loader = get_data_loaders(batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if network_type == 'attention':
        model = IncepAttentionCNN_rgb(n_outputs=10, n_input_channels=1)
    elif network_type == 'compact':
        model = CompactCNN_rgb(n_outputs = 10, n_input_channels=1)
    else:
        model = SimpleCNN(n_outputs = 10, n_input_channels=1)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch}')
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data.to(device))
            loss = criterion(outputs, target.to(device))
            loss.backward()
            optimizer.step()

            if np.mod(batch_idx, 50) == 0:
                print('.', end='')
        
    correct_CNN = 0
    total = 0
    model.eval()
    print('Testing on test set.')
    for batch_idx, (data, target) in enumerate(test_loader):
        outputs = model(data.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct_CNN += (predicted == target.to(device)).sum().item()
        if np.mod(batch_idx, 50) == 0:
            print('.', end='')
    performance = correct_CNN / total

    print(f"\nTest Accuracy CNN: {correct_CNN}/{total} = {correct_CNN/total*100:.2f}%")
    
    return performance

def train_MNIST(network_type='attention', KC_dimension = 20000, n_nonzero_weights = 10):
    """ Train different network types on MNIST and report performance. 
        network_type: 'attention', 'compact', 'simple', 'MBON' """

    # set a random seed:
    rseed = 33
    np.random.seed(rseed)
    torch.manual_seed(rseed)
    
    n_tests = 10
    performance = np.zeros((n_tests, 1))

    if network_type == 'MBON':
        add_str = f'_KC{KC_dimension}_nz{n_nonzero_weights}'
    else:
        add_str = ''

    for t in range(n_tests):
        print(f'Experiment {t+1}/{n_tests}:')
        if network_type == 'MBON':
            performance[t] = train_MBON_MNIST(KC_dimension=KC_dimension, n_nonzero_weights=n_nonzero_weights)
        else:
            performance[t] = train_CNN_MNIST(n_epochs=5, network_type=network_type)

        # store the performance over epochs to a json file:
        with open(f'mb_mnist_performance_{network_type}{add_str}_rseed{rseed}.json', 'w') as f:
            json.dump({'performance': performance[:t+1, :].tolist()}, f)

    print(f'Network type {network_type} performance: {np.mean(performance)*100}% +- ({np.std(performance)*100})')

if __name__ == "__main__":
    train_MNIST()
