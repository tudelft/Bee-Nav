import numpy as np
import torch
import matplotlib.pyplot as plt
from observation_simulator import *
from theory import *
from ANN import *
from plotting import *
from output_utils import _save_or_show
import json

def insect_navigation(parameters=None, output_dir=None):
    if parameters is None:
        # Load the parameters from the JSON file
        with open("parameters.json", "r") as file:
            parameters = json.load(file)

    if parameters['train']:
        # 1) Generate the dataset
        print('Generating the dataset')
        n_samples = parameters['n_samples']
        X, X_lm, t, heading = get_samples(n_samples, parameters=parameters, output_dir=output_dir)

        # 2) Create the neural network
        print('Creating the neural network')
        if(parameters['perceptron']):
            model = torch.nn.Linear(X_lm.shape[1], 2, bias = parameters['biases'])
        else:
            if not parameters['symmetry_network']:
                model  = torch.nn.Sequential(torch.nn.Linear(X_lm.shape[1], parameters['n_hidden'], bias=parameters['biases']), torch.nn.ReLU(),
                                torch.nn.Linear(parameters['n_hidden'], 2, bias=parameters['biases']))
            else:
                model = SymmetryNetwork(parameters['n_landmarks'], parameters['n_hidden'], parameters['biases'])

        # 3) Train the neural network
        print('Training the neural network')
        train_network(model, X_lm, t, n_epochs = parameters['n_epochs'], parameters=parameters, output_dir=output_dir)
        # save the model:
        torch.save(model, 'model.pt')
    else:
        # load the model:
        model = torch.load('model_investigated.pt')

    if parameters['network_analysis']:
        # analyze the network weights:
        analyze_network(model, parameters=parameters, output_dir=output_dir)

    # 4) Test the neural network
    # generate points on a grid:
    print('Testing the neural network')
    x, y, xy, heading, inputs, n_samples_grid, n_headings, headings, n_grid_points = \
        get_grid_samples(single_heading=parameters['single_heading'], parameters=parameters)

    # Show the inputs for different orders:
    # x, y, xy, heading, inputs, n_samples_grid, n_headings, headings, n_grid_points = \
    #     get_grid_samples(order = [0,1,2], show_inputs = True)
    # x, y, xy, heading, inputs, n_samples_grid, n_headings, headings, n_grid_points = \
    #     get_grid_samples(order = [1,2,0], show_inputs = True)

    # compute the model prediction:
    inputs = torch.autograd.Variable(torch.Tensor(inputs).float())
    if parameters['analyze_hidden_activations'] and not parameters['perceptron']:
        hiddens, out = run_MLP_hidden_activations(model, inputs)
    else:
        out = model(inputs)

    #if parameters['network_analysis'] and not parameters['N'] and not parameters['ID']:
        # test whether inputs can effectively be shuffled, and lead to the same outputs:
        # test_parameters.symmetry_network(inputs, model) 

    if(not parameters['N']):

        if(parameters['analyze_hidden_activations']):
            if parameters['cancel_checkered_neurons']:
                checkered_ids_w = get_checkered_ids(model)
                out = run_model_ablating_hiddens(inputs, model, checkered_ids_w, retain_bias=False)

        # plot the prediction:
        plot_not_N(n_headings, out, n_samples_grid, headings, n_grid_points, x, y, parameters=parameters, output_dir=output_dir)

        if parameters['plot_theory']:
            if(parameters['ID'] and parameters['AD'] and not parameters['N'] and not parameters['OCC']):
                plot_theory_ID_AD(inputs, heading, xy, n_headings, n_samples_grid, headings, n_grid_points, x, y, parameters =parameters, output_dir=output_dir)

        # Make a figure that shows the end points of the arrows:
        plot_end_points_not_N(n_headings, n_samples_grid, out, headings, xy, parameters = parameters, output_dir=output_dir)

        if(parameters['analyze_hidden_activations']):
            
            if parameters['show_histogram']:
                # show a histogram of the hidden unit activations:
                plt.figure()
                plt.hist(hiddens[0].detach().numpy(), bins = 50)
                plt.xlabel('Activation')
                plt.ylabel('Count')
                plt.title('Histogram of hidden unit activations')
                _save_or_show(output_dir, 'hidden_activation_histogram')

            if parameters['show_hidden_units']:
                act_min = 0 # np.min(hiddens[0].detach().numpy())
                # get the p-th percentile:
                p = 90
                act_max = np.percentile(hiddens[0].detach().numpy(), p)
                # np.max(hiddens[0].detach().numpy())
                print(f'act_min = {act_min}, act_max = {act_max}')

                # Make a list of all permutations of landmark indices:
                perms = list(itertools.permutations(range(parameters['n_landmarks'])))
                for perm in perms:
                    shuffle_order = list(perm)
                    x, y, xy, heading, inputs, n_samples_grid, n_headings, headings, n_grid_points = get_grid_samples(order = shuffle_order, parameters = parameters)
                    inputs = torch.autograd.Variable(torch.Tensor(inputs).float())
                    hiddens, out = run_MLP_hidden_activations(model, inputs)
                    plot_hidden_activations(hiddens, xy, headings, n_samples_grid, n_headings, order = shuffle_order, act_min = act_min, act_max = act_max, output_dir=output_dir)

                for h in range(10):
                    print(f'Hidden unit {h}:')
                    print_weights_hidden_unit(model, h)


            if parameters['analyze_single_unit']:

                hiddens = [0,1,4,6] #[2,3,5]
                # generate a dataset to see what happens when we ablate the hidden units:
                n_samples = 10000
                X, X_lm, t, heading = get_samples(n_samples, parameters=parameters, output_dir=output_dir)
                X_lm = torch.Tensor(X_lm).float()

                for h in hiddens:
                    # analyze a single hidden unit:
                    weights, bias, weights_out = get_weights_hidden_unit(model, h)
                    analyze_ReLU_headings_orders(weights, bias, weights_out = weights_out, output_dir=output_dir)
                    print(f'weights = {weights}, bias = {bias}, weights_out = {weights_out}')
                    orders = [[0,1,2], [0,2,1], [1,2,0], [1,0,2], [2,0,1], [2,1,0]]

                    for order in orders:
                        ox, oy, oxy, oheading, oinputs, on_samples_grid, on_headings, oheadings, on_grid_points = get_grid_samples(n_grid_points = 25, order = order)
                        oinputs = torch.autograd.Variable(torch.Tensor(oinputs).float())
                        oout_without = run_model_ablating_hiddens(oinputs, model, [h])
                        oout_with = model(oinputs)
                        plot_end_points_not_N(on_headings, on_samples_grid, oout_without, oheadings, oxy, \
                                            col = 'red', out_2 = oout_with, col_2 = 'grey', sup_title_postfix = f' order {order}, hidden {h}', output_dir=output_dir)


                    plt.figure()
                    plt.title(f'Headings for which hidden unit {h} improves or worsens the result')

                    for order in orders:
                        print(f'Order = {order}')
                        X_lm_shuffled = shuffle_landmarks(X_lm, order=order) 
                        X_lm_shuffled = torch.Tensor(X_lm_shuffled).float()

                        out_normal = model(X_lm_shuffled)
                        out_ablated = run_model_ablating_hiddens(X_lm_shuffled, model, [h])

                        # compute the error:
                        out_normal = out_normal.detach().numpy()
                        out_ablated = out_ablated.detach().numpy()
                        errors_normal = (out_normal - t)**2
                        errors_ablated = (out_ablated - t)**2
                        
                        mean_errors_normal = np.mean(errors_normal, axis = 0)
                        mean_errors_ablated = np.mean(errors_ablated, axis = 0)
                        print(f'Error normal, (x,y) = ({mean_errors_normal[0], mean_errors_normal[1]})')
                        print(f'Error ablated, (x,y) = ({mean_errors_ablated[0], mean_errors_ablated[1]})')

                        inds_better = np.where(errors_ablated > errors_normal)[0]
                        inds_worse = np.where(errors_ablated < errors_normal)[0]
                        print(f'Number of samples where the hidden unit {h} improves the result: {len(inds_better)}')
                        print(f'Number of samples where the hidden unit {h} worsens the result: {len(inds_worse)}')

                        plt.subplot(2,3,orders.index(order)+1)
                        plt.hist(np.rad2deg(heading[inds_better]), bins = 50, alpha = 0.5, label = 'Better')
                        plt.hist(np.rad2deg(heading[inds_worse]), bins = 50, alpha = 0.5, label = 'Worse')
                        plt.xlabel('Heading')
                        plt.ylabel('Count')
                        plt.title(f'Order = {order}')
                        plt.legend()

                    _save_or_show(output_dir, f'hidden_unit_{h}_improvement_histogram')



            if parameters['analyze_own_weights']:
                # Analyze manually chosen weights:
                # own_weights = [0, 1, 0, -0.5, 0, -0.5]
                own_weights = [0, 1, -0.2, -0.5, 0.2, -0.5]
                heading_angle = 135
                R = np.array([[np.cos(np.deg2rad(heading_angle)), -np.sin(np.deg2rad(heading_angle))],
                            [np.sin(np.deg2rad(heading_angle)), np.cos(np.deg2rad(heading_angle))]])
                for lm in range(parameters['n_landmarks']):
                    landm = np.array([own_weights[2*lm], own_weights[2*lm+1]])
                    new_weights = np.dot(R, landm)
                    own_weights[2*lm] = new_weights[0]
                    own_weights[2*lm+1] = new_weights[1]
                
                own_bias = -6.5
                analyze_ReLU_headings_orders(own_weights, own_bias, cb_limits=[0,3], output_dir=output_dir)
                
                own_weights = [0, 0.33, 0, 0.33, 0, 0.33]
                own_bias = 0
                analyze_ReLU_headings_orders(own_weights, own_bias, parameters=parameters, output_dir=output_dir)

            if parameters['identify_checkered_neurons']:
                hiddens = hiddens.detach().numpy()
               
                checkered_ids_w = get_checkered_ids(model, print_ids=True)
                checkered_ids_w = np.array(checkered_ids_w)
                print(f'Number of checkered neurons based on weights = {len(checkered_ids_w)}')
                # plot arrows for all the checkered neuron weights:
                plot_arrows_for_weights(checkered_ids_w, model, parameters=parameters, output_dir=output_dir)

                # determine the non-checkered neurons:
                non_checkered_ids_w = np.setdiff1d(range(parameters['n_hidden']), checkered_ids_w)
                # go over all non-checkered neurons, get their weights, and determine if there are negative output weights:
                for h in non_checkered_ids_w:
                    weights, bias, weights_out = get_weights_hidden_unit(model, h)
                    if np.any(weights_out < 0):
                        print(f'Hidden unit {h} has negative output weights: {weights_out}')
                plot_arrows_for_weights(non_checkered_ids_w, model, plot_output_weights = True, parameters=parameters, output_dir=output_dir)

            # analyze_hidden_unit(model, 2, heading = 90, order = [0,1,2])
            # analyze_hidden_unit(model, 2, heading = 90, order = [0,2,1])
            # analyze_hidden_unit(model, 2, heading = 90, order = [1,2,0])
            # analyze_hidden_unit(model, 2, heading = 90, order = [1,0,2])
            # analyze_hidden_unit(model, 2, heading = 90, order = [2,0,1])
            # analyze_hidden_unit(model, 2, heading = 90, order = [2,1,0])

    else:
        plot_N(out, n_grid_points, parameters['nest_location'], x, y, parameters=parameters, output_dir=output_dir)


    print('Done')




