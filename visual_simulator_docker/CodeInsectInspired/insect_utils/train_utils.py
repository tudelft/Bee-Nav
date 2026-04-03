import torch
import torch.nn as nn
import torch.optim as optim
# import wandb
from tqdm import tqdm
from matplotlib import pyplot as plt
from insect_utils.augment_and_save import load_dataset_from_csv
import os
from PIL import Image
from torchvision import transforms
import insect_utils.plot_utils
from insect_utils.flight_path_functions import rotate_vector_by_yaw
import numpy as np
import json
import pickle
from data_loader import augment_image_vector, show_image_and_target

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    # if wandb.run is not None:
    #     wandb.log({"epoch": epoch+1, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


def train(model, loader, val_loader, criterion, optimizer, config, plot_graphs = False):

    # Is the model using the GPU?
    if torch.cuda.is_available():
        print("GPU available")
        model = model.cuda()
        CUDA = True
    else:
        print("Using CPU")
        CUDA = False

    # run a dummy batch through the model to initialize the parameters:
    images, labels = next(iter(loader))
    if CUDA:
        images = images.cuda()
        labels = labels.cuda()
    model(images)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    # if wandb.run is not None:
    #     wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config['epochs']
    example_ct = 0  # number of examples seen
    batch_ct = 0
    val_ct = 0
    validation_losses = []
    for epoch in tqdm(range(config['epochs'])):
        for _, (images, labels) in enumerate(loader):

            if CUDA:
                images = images.cuda()
                labels = labels.cuda()
                
            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

                # Determine and log the validation loss:
                with torch.no_grad():
                    val_loss = 0
                    for images, labels in val_loader:
                        if CUDA:
                            images = images.cuda()
                            labels = labels.cuda()
                        outputs = model(images)
                        val_loss += criterion(outputs, labels).item() * images.size(0)
                    val_loss /= len(val_loader.dataset)
                    # if wandb.run is not None:
                    #     wandb.log({"epoch": epoch+1, "val_loss": val_loss})
                    print(f"Validation loss after {str(example_ct).zfill(5)} examples: {val_loss:.3f}")
                    validation_losses.append(val_loss)
                    val_ct += 1
    
    # plot the validation losses, save as figure and add to wandb:
    if plot_graphs:
        plt.plot(validation_losses)
        plt.xlabel("Validation steps")
        plt.ylabel("Validation loss")
        plt.title("Validation loss over training")
        plt.savefig("validation_loss.png")
        # if wandb.run is not None:
        #     wandb.log({"validation_loss": wandb.Image("validation_loss.png")})
        #     wandb.save("validation_loss.png")
    
    print("Training complete")

    # Save the model together with the architecture:
    # torch.save(model, "model.pt")

    # Create models folder if it doesn't exist
    models_folder = "models"
    os.makedirs(models_folder, exist_ok=True)

    # Save the model as a path file:
    # check how many models already exist with the appropriate name:
    existing_files = [f for f in os.listdir(models_folder) if f.startswith("model" + config["model_suffix"]) and f.endswith('.pth')]
    n_existing_files = len(existing_files)
    if n_existing_files > 0:
        model_name = "model" + config["model_suffix"] + "_" + str(n_existing_files)
    else:
        model_name = "model" + config["model_suffix"]
    # Save the model:
    model_path = os.path.join(models_folder, model_name + ".pth")
    torch.save(model.state_dict(), model_path)

    # Save the model in the exchangeable ONNX format
    onnx_path = os.path.join(models_folder, model_name + ".onnx")
    torch.onnx.export(model, images, onnx_path)
    # if wandb.run is not None:
    #     wandb.save(onnx_path)

    return model_path

def train_batch(images, labels, model, optimizer, criterion):
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def test(model, test_loader, criterion, graphics = False):
    model = model.cpu()  # Ensure the model is on CPU for evaluation
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        distance_errors = []
        angular_errors = []
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total += loss.item() * images.size(0)

            # Additional metrics:
            predictions = outputs.detach().numpy()
            targets = labels.detach().numpy()
            # Calculate the angular errors between the predictions and targets:
            for i in range(len(predictions)):
                pred = predictions[i]
                target = targets[i]
                # Calculate the distance error:
                distance_error = np.linalg.norm(pred[:2]) - np.linalg.norm(target[:2])
                distance_errors.append(distance_error)
                # Calculate the angular error:
                angular_error = np.arctan2(pred[1], pred[0]) - np.arctan2(target[1], target[0])
                angular_error = np.degrees(angular_error)  # Convert to degrees
                angular_errors.append(angular_error)

        total /= len(test_loader.dataset)
        print(f"Test loss of the model  {total} ")
        print(f"Mean absolute distance error: {np.mean(np.abs(distance_errors))}")
        print(f"Mean absolute angular error: {np.mean(np.abs(angular_errors))}")
        print(f'Based on N = {len(distance_errors)} samples')

    if graphics:
        plt.figure()
        # plot the histogram with a brown color:
        plt.hist(distance_errors, bins=20, alpha=0.5, label='Distance errors', color='brown')
        plt.xlabel('Distance error')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure()
        # plot the histogram with an orange color:
        plt.hist(angular_errors, bins=20, alpha=0.5, label='Angular errors', color='orange')
        plt.xlabel('Angular error (degrees)')
        plt.ylabel('Frequency')
        plt.show()

    # check if wandb has been initialized:
    # if wandb.run is not None:
    #     wandb.log({"test_loss":  total})

def evaluate(model, train_loader, position_map, rotation_map, noisy_position_map, target_map, train_indices, n_samples = 500, append_name = ''):

     # Run the model on all images in the dataset, storing the resulting predictions:
    predictions = []
    model.eval()

    # Load the config parameters:
    config = json.load(open("config/config_training.json"))
    image_dir = config['image_folder']

    filenames = list(position_map.keys())
    if n_samples > len(filenames):
        n_samples = len(filenames)
    
    for i in range(n_samples):
        f = os.path.join(image_dir, filenames[train_indices[i]])
        image = Image.open(f).convert('RGB')

        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        # Run the model:
        prediction = model(image)
        # Store the result:
        predictions.append(prediction)

    n_predictions = len(predictions)
    train_indices = train_indices[:n_predictions]
    # select train_indices from all the maps:
    positions = np.array(list(position_map.values())) 
    position_map = positions[train_indices, :]
    rotations = np.array(list(rotation_map.values()))
    rotation_map = rotations[train_indices, :]
    targets = np.array(list(target_map.values()))
    target_map = targets[train_indices, :2]
    if noisy_position_map is not None:
        noisy_positions = np.array(list(noisy_position_map.values()))
        noisy_position_map = noisy_positions[train_indices, :]

    # Transform the ground truth to a numpy array:
    if type(target_map) is dict:
        ground_truths = np.array(list(target_map.values()))
        ground_truths = ground_truths[train_indices, :]
    else:
        ground_truths = target_map

    # Transform the predictions to a numpy array:
    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.detach().numpy()

    insect_utils.plot_utils.make_nice_plot(predictions, position_map, rotation_map, ground_truths, noisy_position_map, append_name = append_name, config = config)

def evaluate_on_dataset(model, data_folder, image_folder, csv_filename, load_indices = False, config = None, evaluation_set='test', augment = False, graphics = False, append_name_evaluation_figure=''):

    # load the dataset:
    filenames, target_x, target_y, position_map, rotation_map, target_map, noisy_position_map, noisy_rotation_map, noisy_target_map =  \
        load_dataset_from_csv(os.path.join(data_folder, csv_filename))
    
    # if load_indices is True, we need to load the indices from the csv file:
    
    if load_indices:
        indices_pkl_name = os.path.join(config["dataset_folder"], "indices.pkl")
        if not os.path.exists(indices_pkl_name):
            load_indices = False

    if load_indices:
        # after training, we want to load the same indices:
        with open(indices_pkl_name, "rb") as file:
            indices = pickle.load(file)
        if evaluation_set == 'test':
            test_indices = indices["test_indices"]
        elif evaluation_set == 'validation':
            test_indices = indices["val_indices"]
        else:
            test_indices = indices["train_indices"]
        # select test_indices from all the maps:
        filenames = [filenames[i] for i in test_indices]
        target_x = [target_x[i] for i in test_indices]
        target_y = [target_y[i] for i in test_indices]
        # Use the indices to access the corresponding keys
        keys = list(position_map.keys())
        position_map = {keys[i]: position_map[keys[i]] for i in test_indices}
        rotation_map = {keys[i]: rotation_map[keys[i]] for i in test_indices}
        target_map = {keys[i]: target_map[keys[i]] for i in test_indices}
        if noisy_position_map is not None:
            noisy_position_map = {keys[i]: noisy_position_map[keys[i]] for i in test_indices}
            noisy_rotation_map = {keys[i]: noisy_rotation_map[keys[i]] for i in test_indices}
            noisy_target_map = {keys[i]: noisy_target_map[keys[i]] for i in test_indices}
    
    ground_truths = np.array(list(target_map.values()))

    # Run the model on all images in the dataset, storing the resulting predictions:
    predictions = []
    model.eval()

    for f, filename in enumerate(tqdm(filenames, desc="Running network on evaluation images.")):
        # Load the image:
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        if augment:
            # show_image_and_target(image, ground_truths[f])
            degree = np.random.randint(0, 360)
            wrapped_image, vector_new = augment_image_vector(image, ground_truths[f], degree)
            image = wrapped_image
            # ground_truths[f] = vector_new
            # show_image_and_target(image, ground_truths[f])
            image = image.unsqueeze(0)
            # Run the model:
            prediction = model(image)
            # show_image_and_target(image.squeeze(0), ground_truths[f], prediction[0].detach().numpy())
            prediction = rotate_vector_by_yaw(prediction[0].detach().numpy(), -degree)
            prediction = [prediction]
            prediction = torch.tensor(prediction, dtype=torch.float32)
        else:
            image = image.unsqueeze(0)
            # Run the model:
            prediction = model(image)     

        # Store the result:
        predictions.append(prediction)

    # Transform the predictions to a numpy array:
    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.detach().numpy()
    
    if graphics:
        insect_utils.plot_utils.make_nice_plot(predictions, position_map, rotation_map, ground_truths, noisy_position_map, \
                                               config = config, append_name=append_name_evaluation_figure, show_plot=False)

    # print the mean squared error:
    mse = np.mean(np.sum((predictions - ground_truths)**2, axis=1))
    print(f"Mean squared error on evaluation set: {mse}")


    return predictions, ground_truths, position_map, rotation_map, noisy_position_map

    

    
