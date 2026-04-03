# import wandb

# Login to W&B
wandb.login()

# Initialize the API
api = wandb.Api()

# Specify your project
project = "g-c-h-e-decroon-tu-delft/VisualHoming"  

# Fetch all runs
runs = api.runs(project)

# Loop through and delete each run
for run in runs:
    print(f"Deleting run {run.id}")
    run.delete()
    
print("All runs have been deleted.")
