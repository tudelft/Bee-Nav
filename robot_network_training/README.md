# Robot Network Training

This folder contains code for training the navigation network offline using gathered learning data, either on a laptop or onboard the Raspberry Pi 4.

## Structure

- `home-learning_laptop/` — Training on a laptop (full version)
- `home-learning_rpi/` — Training onboard the Raspberry Pi 4 (optimised for efficient execution)

Both folders have a similar structure; however, the onboard version is optimised for the Raspberry Pi 4 hardware.

## Setup

1. Download the example data from [SURFdrive](https://surfdrive.surf.nl/files/index.php/s/qv4lwhh6PS630fG) (Password: `InsectMav`).
2. Navigate to the `learning&homing` folder. Choose one of the datasets. Inside, there is a `learning_xxx` folder and a `homing_xxx` folder.
3. Place the learning folder inside `data/train/` (relative to the chosen subfolder).
4. Place the homing folder inside `data/test/`.
5. Place `mask.pkl` inside the `utils/` folder.
6. Open `config.yaml` to set the correct paths and configurations. When using outdoor data, set wind correction to `True`. For indoor data, set it to `False`.

## Training and Testing

```bash
cd home-learning_laptop   # or home-learning_rpi
pip install -r requirements.txt
python3 train.py
```

After training, update `config.yaml` with the correct network path and test dataset path, then:

```bash
python3 test.py
```

## Evaluation

1. In `config.yaml`, set the evaluation data path to the preprocessed testing folder/CSV file.
2. Set `ground_truth` to `mocap` for CyberZoo data or `manual` for outdoor data.
3. Set the network to the name of the trained network.
4. Run `python3 evaluate.py`. Results are saved in the `evaluation_results/` folder.
