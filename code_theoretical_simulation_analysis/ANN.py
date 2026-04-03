import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
import observation_simulator
from output_utils import _save_or_show

class SymmetryNetwork(torch.nn.Module):
    def __init__(self, n_landmarks, n_hidden, biases):
        super(SymmetryNetwork, self).__init__()
        self.n_landmarks = n_landmarks
        self.n_hidden = n_hidden
        self.biases = biases

        # The weights from each pair of inputs should be the same:
        self.fc1 = torch.nn.Linear(2, n_hidden, bias = biases)
        self.fc2 = torch.nn.Linear(n_hidden, 2, bias = biases)
    
    def forward(self, x):
        for lm in range(self.n_landmarks):
            if len(x.shape) == 1:
                x_lm = x[2*lm:2*lm+2]
            else:
                x_lm = x[:, 2*lm:2*lm+2]
            if(lm == 0):
                h = self.fc1(x_lm)
            else:
                h = h + self.fc1(x_lm)
        hr = torch.relu(h)
        self.hiddens = hr
        o = self.fc2(hr)
        return o

def test_symmetry_network(inputs, model, parameters = None, output_dir = None):
    # Test whether the inputs can effectively be shuffled, and lead to the same outputs:
    n_samples = inputs.shape[0]
    n_landmarks = inputs.shape[1] // 2

    avg_distances = np.zeros(n_samples)
    # loop over all samples:
    for i in range(n_samples):
        # Test out all orders of the landmarks:
        permutations_landmarks = list(itertools.permutations(range(n_landmarks)))
        n_permutations = len(permutations_landmarks)
        out_perm = np.zeros([n_permutations, 2])
        if(isinstance(model, SymmetryNetwork)):
            hiddens_perm = np.zeros([n_permutations, parameters['n_hidden']])
        for j, perm in enumerate(permutations_landmarks):
            inputs_perm = np.zeros(n_landmarks * 2)
            for k in range(n_landmarks):
                inputs_perm[2*k:2*k+2] = inputs[i, 2*perm[k]:2*perm[k]+2]
            inputs_perm = torch.autograd.Variable(torch.Tensor(inputs_perm).float())
            out_perm[j] = model(inputs_perm).detach().numpy()
            if(isinstance(model, SymmetryNetwork)):
                hiddens_perm[j] = model.hiddens.detach().numpy()
        # check how close all outputs are from each other:
        distances = []
        for j in range(n_permutations):
            for k in range(j+1, n_permutations):
                distances.append(np.linalg.norm(out_perm[j] - out_perm[k]))

        if(isinstance(model, SymmetryNetwork)):
            # check how close all hidden layers are from each other:
            distances_hiddens = []
            for j in range(n_permutations):
                for k in range(j+1, n_permutations):
                    distances_hiddens.append(np.linalg.norm(hiddens_perm[j] - hiddens_perm[k]))
            
            distances_hiddens = np.asarray(distances_hiddens)
            avg_distances_hiddens = np.mean(distances_hiddens)
        
        distances = np.asarray(distances)
        avg_distances[i] = np.mean(distances)
        
    
    # plot the average distances in a histogram:
    plt.figure()
    plt.hist(avg_distances, bins=100)
    plt.xlabel('Average distance between outputs')
    plt.ylabel('Frequency')
    plt.title('Histogram of average distances between outputs')
    _save_or_show(output_dir, 'symmetry_output_distances')

    # print the average distance:
    print(f'Average distance between outputs: {np.mean(avg_distances)}')

    if(isinstance(model, SymmetryNetwork)):
        plt.figure()
        plt.hist(avg_distances_hiddens, bins=100)
        plt.xlabel('Average distance between hidden layers')
        plt.ylabel('Frequency')
        plt.title('Histogram of average distances between hidden layers')
        _save_or_show(output_dir, 'symmetry_hidden_distances')
        print(f'Average distance between hidden layers: {avg_distances_hiddens}')

def train_network(model, X_lm, t, n_epochs = 200, learning_rate = 0.05, parameters = None, output_dir = None):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []
    
    saved_10 = False

    for epoch in range(n_epochs):

        if(not parameters['ID']):
            # shuffle the landmarks
            X_lm = observation_simulator.shuffle_landmarks(X_lm)

        if(parameters['angular_noise']):
            X_lm = observation_simulator.add_angular_noise(X_lm)

        print('Epoch:', epoch)
        inputs = torch.autograd.Variable(torch.Tensor(X_lm).float())
        targets = torch.autograd.Variable(torch.Tensor(t).float())
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, targets)
        np_loss = loss.data.numpy()
        print('loss:', np_loss)
        if(not saved_10 and np_loss < 9):
            # save the model:
            torch.save(model.state_dict(), 'model_10.pt')
            saved_10 = True
        training_losses.append(loss.data.numpy())
        loss.backward()
        optimizer.step()
    
    training_losses = np.asarray(training_losses)

    plt.figure()
    plt.plot(training_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss')
    _save_or_show(output_dir, 'training_loss')


def run_MLP_hidden_activations(model, inputs):
    # Container for the activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hook for the second layer
    model[1].register_forward_hook(get_activation('second_layer'))
    out = model(inputs)
    hiddens = activations['second_layer']

    return hiddens, out

def print_weights_hidden_unit(model, hidden_unit):
    # get the weights corresponding to the hidden unit:
    weights = model[0].weight.data.numpy()[hidden_unit, :]
    bias = model[0].bias.data.numpy()[hidden_unit]

    # print the weights:
    print(f'Weights for hidden unit {hidden_unit}:')
    print(weights)
    print(f'Bias for hidden unit {hidden_unit}:')
    print(bias)

    #  get the weights from the hidden neuron to the outputs:
    weights_out = model[2].weight.data.numpy()[:, hidden_unit]
    print(f'Weights from hidden unit {hidden_unit} to the outputs:')
    print(weights_out)

def get_weights_hidden_unit(model, hidden_unit):
    # get the weights corresponding to the hidden unit:
    weights = model[0].weight.data.numpy()[hidden_unit, :]
    bias = model[0].bias.data.numpy()[hidden_unit]

    #  get the weights from the hidden neuron to the outputs:
    weights_out = model[2].weight.data.numpy()[:, hidden_unit]

    return weights, bias, weights_out

def analyze_ReLU_headings_orders(weights, bias, headings = [0, 45, 90, 135, 180, 225, 270, 315],
                                 n_grid_points = 50, orders = 'all', cb_limits = None,
                                 weights_out = None, parameters = None, output_dir = None):
    if orders == 'all':
        orders = list(itertools.permutations(range(parameters['n_landmarks'])))
    
    # make a plot with subplots for all headings and orders:
    n_headings = len(headings)
    n_orders = len(orders)
    fig, ax = plt.subplots(n_headings, n_orders, figsize=(15, 15))
    ax = ax.flatten()
    for i, heading in enumerate(headings):
        for j, order in enumerate(orders):
            analyze_ReLU(weights, bias, heading, n_grid_points, order, sub_plot = True, ax = ax[i*n_orders + j], 
                         cb_limits = cb_limits, weights_out = weights_out)
            ax[i*n_orders + j].set_title(f'{heading}$^\circ$ with order {order}')
    # make all titles and texts smaller:
    text_size = 8
    for a in ax:
        a.title.set_fontsize(text_size)
        a.xaxis.label.set_fontsize(text_size)
        a.yaxis.label.set_fontsize(text_size)
        a.tick_params(axis='both', which='major', labelsize=text_size)
    _save_or_show(output_dir, 'relu_headings_orders')

def analyze_ReLU(weights, bias, heading = 0, n_grid_points = 50, order = None,
                 sub_plot = False, ax = None, cb_limits = None, weights_out = None, parameters = None, output_dir = None):
    from observation_simulator import get_samples_locations

    # make a heatplot in space for the hidden unit:
    x = np.linspace(-parameters['size_region'], parameters['size_region'], n_grid_points)
    y = np.linspace(-parameters['size_region'], parameters['size_region'], n_grid_points)
    x, y = np.meshgrid(x, y)
    xy = np.column_stack([x.flatten(), y.flatten()])
    n_samples_grid = xy.shape[0]
    headings = np.ones(n_samples_grid) * heading

    if order is None:
        order = list(range(parameters['n_landmarks']))

    inputs = get_samples_locations(xy, np.deg2rad(heading), shuffle_order = order)

    # Run our own ReLU function:
    # Multiply the inputs with the weights:
    h = np.dot(inputs, weights)
    # add the bias:
    h += bias
    # apply the ReLU function:
    h = np.maximum(h, 0)
    if(weights_out is not None):
        # multiply with the weights from the hidden neuron to the outputs:
        h_out_x = h * weights_out[0]
        h_out_y = h * weights_out[1]
        h_out = np.column_stack([h_out_x, h_out_y])
        avg_h_out = np.mean(h_out, axis = 0)


    # show the activations with a scatter plot:
    if not sub_plot:
        plt.figure()
        plt.scatter(xy[:,0], xy[:,1], c=h, cmap='inferno') # other colormaps are: 'viridis', 'plasma', 'inferno', 'magma'
        plt.colorbar()
        # plot the landmarks:
        if not ID:
            plt.plot(parameters['landmarks'][:,0], parameters['landmarks'][:,1], 'bo')
        else:
            color_values = np.linspace(0, 1, parameters['n_landmarks'])
            plt.scatter(parameters['landmarks'][:,0], parameters['landmarks'][:,1], c=color_values, cmap='brg')
        if order is not None:
            plt.title(f'Heatmap for {heading}$^\circ$ with order {order}')
        else:
            plt.title(f'Heatmap for {heading}$^\circ$')
        _save_or_show(output_dir, f'relu_heatmap_{heading}deg')
    else:
        ax.scatter(xy[:,0], xy[:,1], c=h, cmap='inferno')
        # show the colorbar:
        text_size = 8
        if cb_limits is None:
            cbar = plt.colorbar(ax.scatter(xy[:,0], xy[:,1], c=h, cmap='inferno'), ax=ax)
        else:
            cbar = plt.colorbar(ax.scatter(xy[:,0], xy[:,1], c=h, cmap='inferno', vmin = cb_limits[0],
                                           vmax = cb_limits[1]), ax=ax)
        # adapt the font size of the colorbar:
        cbar.ax.tick_params(labelsize=text_size)
        # plot the landmarks:
        if not ID:
            ax.plot(parameters['landmarks'][:,0], parameters['landmarks'][:,1], 'bo')
        else:
            color_values = np.linspace(0, 1, parameters['n_landmarks'])
            ax.scatter(parameters['landmarks'][:,0], parameters['landmarks'][:,1], c=color_values, cmap='brg')
        if weights_out is not None:
            avg_loc_landmarks = np.mean(parameters['landmarks'], axis=0)
            # draw an arrow from the average location of the landmarks in the direction of the average output, rotated by the heading:
            # rotate the average output by the heading:
            rotated_h_out = np.dot(np.array([[np.cos(np.deg2rad(-heading)), -np.sin(np.deg2rad(-heading))],
                                             [np.sin(np.deg2rad(-heading)), np.cos(np.deg2rad(-heading))]]), avg_h_out)
            ax.arrow(avg_loc_landmarks[0], avg_loc_landmarks[1], rotated_h_out[0], rotated_h_out[1], head_width=0.5, head_length=0.5, fc='k', ec='k')

def analyze_hidden_unit(model, hidden_unit, heading = 0, n_grid_points = 50, order = None, parameters=None, output_dir = None):
    # get the weights corresponding to the hidden unit:
    weights = model[0].weight.data.numpy()[hidden_unit, :]
    bias = model[0].bias.data.numpy()[hidden_unit]

    # print the weights:
    print(f'Weights for hidden unit {hidden_unit}:')
    print(weights)
    print(f'Bias for hidden unit {hidden_unit}:')
    print(bias)

    # make a heatplot in space for the hidden unit:
    x = np.linspace(-parameters['size_region'], parameters['size_region'], n_grid_points)
    y = np.linspace(-parameters['size_region'], parameters['size_region'], n_grid_points)
    x, y = np.meshgrid(x, y)
    xy = np.column_stack([x.flatten(), y.flatten()])
    n_samples_grid = xy.shape[0]
    headings = np.ones(n_samples_grid) * heading

    if order is None:
        order = list(range(parameters['n_landmarks']))

    inputs = get_samples_locations(xy, np.deg2rad(heading), shuffle_order = order)

    # Run our own ReLU function:
    # Multiply the inputs with the weights:
    h = np.dot(inputs, weights)
    # add the bias:
    h += bias
    # apply the ReLU function:
    h = np.maximum(h, 0)

    # show the activations with a scatter plot:
    plt.figure()
    plt.scatter(xy[:,0], xy[:,1], c=h, cmap='inferno') # other colormaps are: 'viridis', 'plasma', 'inferno', 'magma'
    plt.colorbar()
    # plot the landmarks:
    if not parameters['ID']:
        plt.plot(parameters['landmarks'][:,0], parameters['landmarks'][:,1], 'bo')
    else:
        color_values = np.linspace(0, 1, parameters['n_landmarks'])
        plt.scatter(parameters['landmarks'][:,0], parameters['landmarks'][:,1], c=color_values, cmap='brg')
    if order is not None:
        plt.title(f'Heatmap for h = {hidden_unit}, {heading}$^\circ$ with order {order}')
    else:
        plt.title(f'Heatmap for h = {hidden_unit}, {heading}$^\circ$')
    _save_or_show(output_dir, f'hidden_unit_{hidden_unit}_{heading}deg')

def analyze_network(model, parameters = None, output_dir = None):
    biases = parameters['biases']

    if parameters['perceptron']:
        precision = 2
        print('Perceptron weights:')
        weights = model.weight.data.numpy()
        print(np.round(weights, precision))
        print('Perceptron bias:')
        biases = model.bias.data.numpy()
        print(np.round(biases, precision))
        print('Landmark locations:')
        print(np.round(parameters['landmarks'], precision))


        if(not parameters['N'] and parameters['ID'] and parameters['AD'] and parameters['n_landmarks'] == 2):
            # Predict the weights of the perceptron:
            # (1) move towards the center of the landmarks:
            w_1 = np.zeros([2,4])
            # average x and y:
            w_1[0,0] = 0.5
            w_1[0,2] = 0.5
            w_1[1,1] = 0.5
            w_1[1,3] = 0.5
            # (2) when at the center, move towards the nest:
            # u = alpha * R(beta) * k
            # where k is the vector from one landmark to the other,
            # beta is the angle from that vector to the nest,
            # and alpha is a scaling factor that brings the vector to the nest

            # k = landmarks[0] - landmarks[1]
            w_2 = np.zeros([2,4])
            # subtract x:
            w_2[0,0] = 1
            w_2[0,2] = -1
            # subtract y:
            w_2[1,1] = 1
            w_2[1,3] = -1

            # this part depends only on the fixed landmarks:
            landmarks = np.asarray(parameters['landmarks'])
            landmark_center = np.mean(landmarks, axis=0)
            k = landmarks[0] - landmarks[1]
            cos_beta = np.dot(k, -landmark_center) / (np.linalg.norm(k) * np.linalg.norm(landmark_center))
            beta = -np.arccos(cos_beta)
            sin_beta = np.sin(beta)
            R = np.array([[cos_beta, -sin_beta], [sin_beta, cos_beta]])
            alpha = np.linalg.norm(landmark_center) / np.linalg.norm(k)
            u = alpha * R
            w_2[:,:2] = np.dot(u, w_2[:,:2])
            w_2[:,2:] = np.dot(u, w_2[:,2:])

            w_3 = w_1 + w_2
            print('Predicted weights:')
            print(np.round(w_3, precision))

        print('Done with analysis!')
    else:
        # MLP
        if not parameters['symmetry_network']:
            # print the weights:
            for i, layer in enumerate(model):
                if(isinstance(layer, torch.nn.Linear)):
                    print(f'Layer {i}:')
                    print('Weights:')
                    print(np.round(layer.weight.data.numpy(), 2))
                    if layer.bias is not None:
                        print('Biases:')
                        print(np.round(layer.bias.data.numpy(), 2))
                    # Analysis of output weights:
                    # if i == 2:
                    #     # plot two bar plots one under the other with the weights for x and y:
                    #     plt.figure()
                    #     plt.subplot(2,1,1)
                    #     weights_x = layer.weight.data.numpy()[0,:n_hidden]
                    #     plt.bar(range(n_hidden), weights_x)
                    #     plt.title('Weights for x')
                    #     plt.subplot(2,1,2)
                    #     weights_y = layer.weight.data.numpy()[1,:n_hidden] 
                    #     plt.bar(range(n_hidden), weights_y)
                    #     plt.title('Weights for y')

                    #     awx = np.abs(weights_x)
                    #     sorted_awx = np.argsort(awx)
                    #     awy = np.abs(weights_y)
                    #     sorted_awy = np.argsort(awy)
                    #     print('Most important weights for x:')
                    #     print(sorted_awx[-10:])
                    #     print('Most important weights for y:')
                    #     print(sorted_awy[-10:])

                    #     # determine percentage overlap of "large" weights, >= 0.25:
                    #     large = 0.25
                    #     n_large_xy = np.sum(np.int8(awx >= large) * np.int8(awy >= large))
                    #     print(f'Percentage of large weights in common: {np.round(n_large_xy / n_hidden * 100, 2)}%')
                    #     print(f'Percentage of common large weights wrt x own large weights: {np.round(n_large_xy / np.sum(np.int8(awx >= large)) * 100, 2)}%')
                    #     print(f'Percentage of common large weights wrt y own large weights: {np.round(n_large_xy / np.sum(np.int8(awy >= large)) * 100, 2)}%')
            # check if the file 'model_10.pt' exists:
            import os
            if os.path.exists('model_10.pt'):
                saved_10 = True

            if saved_10:
                # load the model:
                # number of inputs of model:
                n_inputs = model[0].weight.data.numpy().shape[1]
                if(biases):
                    b = True
                model_10 = torch.nn.Sequential(torch.nn.Linear(n_inputs, parameters['n_hidden'], bias=b), torch.nn.ReLU(),
                        torch.nn.Linear(parameters['n_hidden'], 2, bias=b))
                model_10.load_state_dict(torch.load('model_10.pt'))
                print('Model loaded')
                # show 4 subplots, for the weights, biases of both layers:
                fig, ax = plt.subplots(2, 2, figsize=(15, 10))
                ax = ax.flatten()
                weights = model[0].weight.data.numpy() 
                weights_10 = model_10[0].weight.data.numpy()
                ax[0].imshow(weights - weights_10, cmap='coolwarm')
                ax[0].set_title('Layer 0 weights')
                ax[0].set_xlabel('Input')
                ax[0].set_ylabel('Hidden')
                # add the color bar:
                plt.colorbar(ax[0].imshow(weights - weights_10, cmap='coolwarm'), ax=ax[0])
                if model[0].bias is not None:
                    biases = model[0].bias.data.numpy()
                    biases_10 = model_10[0].bias.data.numpy()
                    ax[1].imshow(np.reshape(biases - biases_10, [1,parameters['n_hidden']]), cmap='coolwarm')
                    ax[1].set_title('Layer 0 biases')
                    ax[1].set_xlabel('Hidden')
                    ax[1].set_ylabel('Bias')
                    # add the color bar:
                    plt.colorbar(ax[1].imshow(np.reshape(biases - biases_10, [1,parameters['n_hidden']]), cmap='coolwarm'), ax=ax[1])
                weights = model[2].weight.data.numpy()
                weights_10 = model_10[2].weight.data.numpy()
                ax[2].imshow(weights - weights_10, cmap='coolwarm')
                ax[2].set_title('Layer 2 weights')
                ax[2].set_xlabel('Hidden')
                ax[2].set_ylabel('Output')
                # add the color bar:
                plt.colorbar(ax[2].imshow(weights - weights_10, cmap='coolwarm'), ax=ax[2])
                if model[2].bias is not None:
                    biases = model[2].bias.data.numpy()
                    biases_10 = model_10[2].bias.data.numpy()
                    ax[3].imshow(np.reshape(biases - biases_10, [1,2]), cmap='coolwarm')
                    ax[3].set_title('Layer 2 biases')
                    ax[3].set_xlabel('Output')
                    ax[3].set_ylabel('Bias')
                    # add the color bar:
                    plt.colorbar(ax[3].imshow(np.reshape(biases - biases_10, [1,2]), cmap='coolwarm'), ax=ax[3])
                _save_or_show(output_dir, 'weight_comparison')

        print('Done with analysis')

def run_model_ablating_hiddens(inputs, model, hidden_ids, retain_bias=False):
    first_layer = model[0]
    hiddens = torch.matmul(inputs, first_layer.weight.t()) + first_layer.bias
    activation_function = model[1]
    hiddens = activation_function(hiddens)

    for h in hidden_ids:
        if retain_bias:
            hiddens[:,h] = first_layer.bias[h]
        else:
            hiddens[:,h] = 0
    
    second_layer = model[2]
    out = torch.matmul(hiddens, second_layer.weight.t()) + second_layer.bias
    return out

def get_checkered_ids(model, sum_threshold = 0.02, print_ids = False):
    layer_0 = model[0]
    weights = layer_0.weight.data.numpy()
    n_hiddens = weights.shape[0]
    checkered_ids_w = []
    for h in range(n_hiddens):
        weights, bias, weights_out = get_weights_hidden_unit(model, h)
        sum_x_weights = np.sum(weights[0::2])
        sum_y_weights = np.sum(weights[1::2])
        if(abs(sum_x_weights) < sum_threshold and abs(sum_y_weights) < sum_threshold):
            if print_ids:
                print(f'Hidden unit {h} is a checkered neuron, bias = {bias}, weights = {weights}, weights_out = {weights_out}')
            checkered_ids_w.append(h)
    checkered_ids_w = np.array(checkered_ids_w)
    return checkered_ids_w
