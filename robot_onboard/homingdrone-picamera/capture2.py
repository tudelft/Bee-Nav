#!/usr/bin/python3

import os
import cv2
import time
import threading
import argparse
from datetime import datetime
from flask import Flask, request
import requests
import numpy as np
import torch
import atexit
from collections import deque
from queue import Queue
import random
from torch.amp import autocast
import torch.nn.functional as F

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from reclinear import read_image, apply_mask, convert_image, after_process, rotate_image, get_mask
from single_image_test import predict
from single_image_test_onnx import predict_onnx
import learning_utils

# set thread limits for performance
cv2.setNumThreads(1)
torch.set_num_threads(2)

config = learning_utils.load_config("./online_learning_config.yaml")
BATCH_SIZE = config['training'].get('batch_size', 4)
REPLAY_BUFFER_SIZE = config['training'].get('replay_buffer_size', 20000)
LOG_EVERY_N_STEPS = config['training'].get('log_interval', 50)
SAVE_EVERY_N_STEPS = config['training'].get('save_interval', 50)

LOG_INTERVAL = LOG_EVERY_N_STEPS

# Global cache to store images in RAM for faster access
global_ram_cache = {} 

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='homing', choices=['homing', 'learning']) # 'homing' for inference, 'learning' for online training
parser.add_argument('--net_path', type=str, default='./gaze_net.pth')
parser.add_argument('--net_size', type=str, default='small')
parser.add_argument('--crop_type', type=str, default='full')
parser.add_argument('--wind_correction', type=str, default='on')
parser.add_argument('--onnx_model_path', type=str, default=None) # Path to ONNX model for inference faster than PyTorch
args, _ = parser.parse_known_args()

file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
file_date = datetime.now().strftime("%Y%m%d")
# Note: /home/data/ is the data directory on the drone's Raspberry Pi
folder_location = f'/home/data/{file_date}/{file_id}/'
rawdata_location = f'/home/data/{file_date}/{file_id}/{file_id}_rawdata/'
rectilinear_location = f'/home/data/{file_date}/{file_id}/{file_id}_rectilinear/'
os.makedirs(rawdata_location, exist_ok=True)
os.makedirs(rectilinear_location, exist_ok=True)
online_learning_dir = os.path.join(folder_location, "online_learning")
os.makedirs(online_learning_dir, exist_ok=True)

log_file_path = os.path.join(online_learning_dir, 'training_log.csv')

csv_path = f'{rectilinear_location}data.csv'
csv_path_raw = f'{rawdata_location}data.csv'

if args.mode == 'homing':
    model_name = args.net_path.split('/')[-1].split('.')[0]
else:
    model_name = "ONLINE_LEARNING"

csv_header = f'path_idx,recti_path,pos_x,pos_y,heading,pitch,roll,pos_x_mocap,pos_y_mocap,heading_mocap,{model_name}_output,{model_name}_distance\n'
csv_header_raw = 'path_idx,pos_x,pos_y,heading,pitch,roll,pos_x_mocap,pos_y_mocap,heading_mocap\n'

with open(csv_path, 'w') as f: f.write(csv_header)
with open(csv_path_raw, 'w') as f: f.write(csv_header_raw)

# Global state for the learning process
global_learning_state = {
    "net": None,
    "optimizer": None,
    "dataset_factory": None,
    "replay_buffer": None,
    "step_counter": 0,
    "model_save_path": os.path.join(online_learning_dir, 'gazenet_online'),
    "log_file_path": os.path.join(online_learning_dir, 'training_log.csv'),
    "log_file_header": "step_count,avg_loss,avg_dir_loss,avg_dist_loss\n"
}
learning_lock = threading.Lock()

picam2 = Picamera2()
half_resolution = [dim // 2 for dim in picam2.sensor_resolution]
main_stream = {"size": half_resolution}
lores_stream = {"size": (640, 480)}
picam2.configure(picam2.create_video_configuration(main_stream, lores_stream, encode="lores"))

def start_rpi_recording():
    picam2.start_recording(H264Encoder(2000000), f'{folder_location}rpi_video.h264')
    while True: time.sleep(10)
    
threading.Thread(target=start_rpi_recording, daemon=True).start()

def save_image_async(path, image):
    # Helper to save images without blocking the main thread
    try:
        cv2.imwrite(path, image)
    except Exception as e:
        print(f"Warning: failed to save image {path}: {e}")

def wind_correct(pitch, roll, x, y, image):
    # Adjust coordinates based on drone pitch/roll to compensate for wind
    x = (5.5439) * pitch + (-135.9334) * roll + (839.4777)
    y = (136.7321) * pitch + (2.7694) * roll + (636.8401)
    return int(x), int(y)

def rectilinear(file_name, output_location, file_id, capture_data):
    mask, x, y, r = get_mask()
    image = read_image(file_name)
    if args.wind_correction == 'on':
        x, y = wind_correct(float(capture_data['pitch']), float(capture_data['roll']), x, y, image)
    
    image_polar = convert_image(image, x, y, r)
    image_final = after_process(image_polar)
    image_rotate = rotate_image(image_final)
    
    if args.crop_type == 'upper':
        height, width, _ = image_rotate.shape
        img_lower = image_rotate[height//2:, :]
        img_lower = cv2.resize(img_lower, (width, height))
        image_rotate = img_lower
        
    recti_filename = f"{file_name.split('/')[-1].split('.')[0].split('_')[0]}_rected_{file_id}.jpg"
    recti_image_path = os.path.join(output_location, recti_filename)
    return recti_image_path, recti_filename, image_rotate


def learning_worker_thread():
    print("✅ Learning worker started (Non-Blocking Mode).")
    
    best_loss = float('inf')
    loss_100 = 0.0
    direction_loss_100 = 0.0
    distance_loss_100 = 0.0
    
    while len(global_learning_state["replay_buffer"]) < BATCH_SIZE * 4:
        time.sleep(1)
    print("🧠 Buffer filled. Training loop active.")
    
    while True:
        try:
            t_loop_start = time.time()
            batch_indices = None
            
            t_lock_req = time.time()
            with learning_lock:
                t_lock_acq = time.time()
                if len(global_learning_state["replay_buffer"]) >= BATCH_SIZE:
                    batch_indices = random.sample(global_learning_state["replay_buffer"], BATCH_SIZE)
                    net = global_learning_state["net"]
                    optimizer = global_learning_state["optimizer"]
            t_lock_rel = time.time()
            
            if batch_indices is None:
                time.sleep(0.05)
                continue
                
            t_prep_start = time.time()
            batch_data = [global_learning_state["dataset_factory"][i] for i in batch_indices]
            
            inputs = torch.stack([d[0] for d in batch_data])
            labels = torch.stack([d[1] for d in batch_data])
            t_prep_end = time.time()
            
            t_train_start = time.time()
            net.train()
            
            # Use autocast for better performance on CPU
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                outputs = net(inputs)
                
                # Convert to float32 for precise loss calculation
                outputs_f32 = outputs.to(torch.float32)
                labels_f32 = labels.to(torch.float32)

                loss = F.mse_loss(outputs_f32, labels_f32[..., :2])

                pred_direction = torch.atan2(outputs_f32[:, 1], outputs_f32[:, 0])
                pred_distance = torch.sqrt(outputs_f32[:, 0]**2 + outputs_f32[:, 1]**2)

                target_direction = torch.atan2(labels_f32[:, 1], labels_f32[:, 0])
                target_distance = torch.sqrt(labels_f32[:, 0]**2 + labels_f32[:, 1]**2)

                direction_error = F.mse_loss(pred_direction, target_direction)
                distance_error = F.mse_loss(pred_distance, target_distance)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_100 += loss.item()
            direction_loss_100 += direction_error.item()
            distance_loss_100 += distance_error.item()
            
            t_train_end = time.time()
            
            with learning_lock:
                global_learning_state["step_counter"] += 1
                step = global_learning_state["step_counter"]
                
                lock_ms = (t_lock_rel - t_lock_req) * 1000
                prep_ms = (t_prep_end - t_prep_start) * 1000
                train_ms = (t_train_end - t_train_start) * 1000
                total_ms = (t_train_end - t_loop_start) * 1000
                
                if step % LOG_INTERVAL == 0:
                    avg_loss = loss_100 / LOG_INTERVAL
                    avg_dir_loss = direction_loss_100 / LOG_INTERVAL
                    avg_dist_loss = distance_loss_100 / LOG_INTERVAL
                    
                    print(f'[{step}, {step}] Direction Loss: {avg_dir_loss:.3f}, Distance Loss: {avg_dist_loss:.3f}, Total Loss: {avg_loss:.3f}')
                    
                    with open(log_file_path, 'a') as f:
                        f.write(f'{step},{avg_dir_loss},{avg_dist_loss},{avg_loss}\n')
                    
                    if avg_loss < best_loss and step > 800:
                        best_loss = avg_loss
                        torch.save(net.state_dict(), global_learning_state["model_save_path"] + "_best.pth")

                    loss_100 = 0.0
                    direction_loss_100 = 0.0
                    distance_loss_100 = 0.0
                
                if step % SAVE_EVERY_N_STEPS == 0:
                    torch.save(net.state_dict(), global_learning_state["model_save_path"] + f"_step{step}.pth")
            
        except Exception as e:
            print(f"Worker Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

@app.route('/trigger', methods=['POST'])
def combined_trigger():
    t_trig_start = time.time()
    capture_data = request.form.to_dict()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_idx = f"{capture_data['path_idx']}_{timestamp}"

    pi_request = picam2.capture_request()
    raw_path = f"{rawdata_location}{path_idx}.jpg"
    pi_request.save("main", raw_path)
    pi_request.release()
    t_cam = time.time()
    
    recti_path, recti_filename, img_arr = rectilinear(raw_path, rectilinear_location, timestamp, capture_data)
    t_proc = time.time()
    
    if args.mode == 'learning':
        global_ram_cache[recti_filename] = img_arr
        
        # Save to disk in background since we don't need the file immediately
        threading.Thread(target=save_image_async, args=(recti_path, img_arr), daemon=True).start()
        
        log_data_row = {
            'recti_path': recti_filename,
            'pos_x': capture_data['pos_x'],
            'pos_y': capture_data['pos_y'],
            'heading': capture_data['heading'],
            f'{model_name}_output': 0.0,
            f'{model_name}_distance': 0.0
        }
        
        log_values = [str(log_data_row.get(k, 0)) for k in csv_header.strip().split(',')]
        with open(csv_path, 'a') as f: f.write(','.join(log_values) + '\n')
        
        with learning_lock:
            base_idx = global_learning_state["dataset_factory"].append_data(log_data_row)
            num_aug = global_learning_state["num_augmentations"]
            new_indices = list(range(base_idx * num_aug, (base_idx + 1) * num_aug))
            global_learning_state["replay_buffer"].extend(new_indices)

    elif args.mode == 'homing':
        # We must save the file synchronously because 'predict' reads it from disk.
        cv2.imwrite(recti_path, img_arr)
        
        if args.onnx_model_path is not None:
            nav_prediction, nav_distance = predict_onnx(recti_path, args.onnx_model_path)
        else:
            nav_prediction, nav_distance = predict(recti_path, args.net_path, args.net_size)
        
        final_prediction, final_distance = nav_prediction, nav_distance
            
        log_line = f"{path_idx},{recti_filename},{capture_data['pos_x']},{capture_data['pos_y']},{capture_data['heading']},{capture_data['pitch']},{capture_data['roll']},{capture_data['pos_x_mocap']},{capture_data['pos_y_mocap']},{capture_data['heading_mocap']},{final_prediction},{final_distance}\n"
        with open(csv_path, 'a') as f: f.write(log_line)
        
        try:
            requests.post('http://localhost:5001/predict', data={'path_idx': capture_data['path_idx'], 'output': final_prediction, 'distance': final_distance})
        except Exception:
            pass  # Flight control server may not be running during data collection

    with open(csv_path_raw, 'a') as f:
        f.write(f"{path_idx},{capture_data['pos_x']},{capture_data['pos_y']},{capture_data['heading']},{capture_data['pitch']},{capture_data['roll']},{capture_data['pos_x_mocap']},{capture_data['pos_y_mocap']},{capture_data['heading_mocap']}\n")

    t_end = time.time()
    print(f"[TIMING-TRIGGER] Cam: {(t_cam-t_trig_start)*1000:.1f}ms | Proc: {(t_proc-t_cam)*1000:.1f}ms | Total: {(t_end-t_trig_start)*1000:.1f}ms")
    
    return "OK", 200

if __name__ == '__main__':
    print(f"--- Starting Server in {args.mode.upper()} Mode ---")
    
    if args.mode == 'learning':
        print("Initializing Online Learning...")
        config = learning_utils.load_config("./online_learning_config.yaml")
        net, transform, optimizer, num_aug = learning_utils.get_learning_components(config)
        dataset_factory = learning_utils.GazeDataset(csv_path, rectilinear_location, transform, global_ram_cache)
        
        global_learning_state["net"] = net
        global_learning_state["optimizer"] = optimizer
        global_learning_state["dataset_factory"] = dataset_factory
        global_learning_state["replay_buffer"] = deque(maxlen=REPLAY_BUFFER_SIZE)
        global_learning_state["num_augmentations"] = num_aug
        
        threading.Thread(target=learning_worker_thread, daemon=True).start()
    
    print("🚀 Flask Server Running.")
    app.run(host='0.0.0.0', port=5000)
