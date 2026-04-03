# plot the trees for a given experiment
import os
import csv
from matplotlib import pyplot as plt
import numpy as np


# determines which files will be loaded and shown.
n_trees = 40
n_experiments = 10
n_rows = int(np.ceil(np.sqrt(n_experiments)))
n_cols = int(np.ceil(n_experiments / n_rows))

plt.figure()
for exp in range(n_experiments):
      
    # make a subplot for each experiment
    ax = plt.subplot(n_rows, n_cols, exp + 1)
    ax.set_title(f'Experiment {exp + 1}')
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_aspect('equal', adjustable='box')

    landmark_filename = f'./maps/forest_{n_trees}_trees_20x20_area_locations_{exp}.csv'

    if os.path.exists(landmark_filename):
        with open(landmark_filename, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            X = []
            Y = []
            Z = []
            for row in reader:
                x, y, z = row
                X.append(float(x))
                Y.append(float(y))
                Z.append(float(z))
            x = np.asarray(X)
            y = np.asarray(Y)
            z = np.asarray(Z)
            n_landmarks = len(X)
            landmark_positions = np.zeros([n_landmarks, 3])
            landmark_positions[:, 0] = x
            landmark_positions[:, 1] = y
            landmark_positions[:, 2] = z
    else:
        print(f"File {landmark_filename} does not exist.")
        landmark_positions = None
        break

    # plot the landmarks:
    if landmark_positions is not None:
        for l in range(len(landmark_positions)):
            # make the radius 2.0
            circle = plt.Circle((landmark_positions[l, 0], landmark_positions[l, 1]), 2.0, color='lightgreen', alpha=0.5, fill=True)
            ax.add_artist(circle)
    plt.tight_layout()

plt.savefig(f'landmark_maps_{n_trees}_trees.png', dpi=300)
plt.show()

print("Done.")