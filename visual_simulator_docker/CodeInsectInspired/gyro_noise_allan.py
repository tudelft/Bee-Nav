import numpy as np

def simulate_gyro_noise(arw_per_sqrt_hour, bi_per_hour, rrw_per_hour_3_2, rr_per_hour_2, dt, num_steps):
    """
    Simulates a gyroscope output with noise components: ARW, Bias Instability, Rate Random Walk, and Rate Ramp.
    
    Parameters:
    - arw_per_sqrt_hour: Angle Random Walk coefficient (°/sqrt(hr))
    - bi_per_hour: Bias Instability coefficient (°/hr)
    - rrw_per_hour_3_2: Rate Random Walk coefficient (°/hr^(3/2))
    - rr_per_hour_2: Rate Ramp coefficient (°/hr^2)
    - dt: Time step (s)
    - num_steps: Number of simulation steps
    
    Returns:
    - time: Time array
    - noisy_angle: Simulated gyroscope angle (rad)
    """
    # Convert input values from per-hour to per-second
    arw_per_sqrt_sec = arw_per_sqrt_hour / np.sqrt(3600)
    bi_per_sec = bi_per_hour / 3600
    rrw_per_sec = rrw_per_hour_3_2 / (3600**(3/2))
    rr_per_sec = rr_per_hour_2 / (3600**2)
    
    time = np.arange(0, num_steps * dt, dt)
    noisy_angle = 0.0  # Initial angle
    yaw_angles = []
    
    # Initialize Bias Instability as a flicker noise approximation
    alpha = 0.999  # Slow decay factor
    bi_noise = 0.0
    
    # Initialize Rate Random Walk
    rrw_bias = 0.0
    
    for t in time:
        # Angle Random Walk (ARW) - White noise scaled by sqrt(dt)
        arw_noise = arw_per_sqrt_sec * np.sqrt(dt) * np.random.randn()
        
        # Bias Instability (BI) - Modeled as a slow random drift (Ornstein-Uhlenbeck Process Approx.)
        bi_noise = alpha * bi_noise + bi_per_sec * dt * np.random.randn()
        
        # Rate Random Walk (RRW) - Randomly drifting bias term
        rrw_bias += rrw_per_sec * np.sqrt(dt) * np.random.randn()
        
        # Rate Ramp (RR) - Linear drift in rate
        rr_noise = rr_per_sec * t
        
        # Total yaw noise
        total_yaw_noise_deg = arw_noise + bi_noise + rrw_bias + rr_noise
        total_yaw_noise_rad = np.radians(total_yaw_noise_deg)
        
        # Integrate to get angle
        noisy_angle += total_yaw_noise_rad
        yaw_angles.append(noisy_angle)
    
    return time, np.array(yaw_angles)

# Example usage with realistic MEMS gyro values
# https://www.engineeringforchange.org/forums/topic/mems-gyroscope-error-compensation-by-allan-variance-method/?utm_source=chatgpt.com
time, yaw_angles = simulate_gyro_noise(
    arw_per_sqrt_hour=7.5,  # ARW in °/sqrt(hr)
    bi_per_hour=2.19,       # BI in °/hr
    rrw_per_hour_3_2=5.64,  # RRW in °/hr^(3/2)
    rr_per_hour_2=109.58,   # RR in °/hr^2
    dt=0.01,               # 100 Hz sampling rate
    num_steps=10000        # Simulate for 100 seconds
)

# Plot results
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(time, yaw_angles, label="Simulated Noisy Yaw Angle", linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Yaw Angle (rad)")
plt.title("Simulated Gyroscope Output with Noise Components")
plt.legend()
plt.grid(True)
plt.show()
