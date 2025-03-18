import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy import integrate

def process_jump_data(file_path, sampling_rate=100):
    # Load and parse data
    with open(file_path, "r") as file:
        lines = file.readlines()
    start_idx = next(i for i, line in enumerate(lines) if line.startswith("DATA")) + 1
    data_lines = lines[start_idx:]
    
    data = []
    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) == 4:
            try:
                timestamp = int(parts[0])
                hdr_x = int(parts[1]) * (9.8 / 1000)  # Convert to m/s² from mg
                data.append([timestamp, hdr_x])
            except ValueError:
                continue

    df = pd.DataFrame(data, columns=["Timestamp", "Hdr_X"])
    df["Time (s)"] = (df["Timestamp"] - df["Timestamp"].iloc[0]) / 1000.0
    df["Hdr_X"] -= np.mean(df["Hdr_X"])  # Remove offset
    
    # Apply high-pass filter to remove drift if the signal is long enough
    cutoff_freq = 0.5
    b, a = butter(1, cutoff_freq / (sampling_rate / 2), btype='high')
    
    # Apply the filter to the data
    df["Hdr_X_filtered"] = filtfilt(b, a, df["Hdr_X"])
    
    # Detect peaks corresponding to jumps (focusing on one isolated peak per jump)
    jump_threshold = 3  # Further lowered threshold for better detection (previously 10)
    peaks, _ = find_peaks(df["Hdr_X_filtered"], height=jump_threshold, distance=100)  # Ensure peaks are sufficiently separated
    
    if len(peaks) == 0:
        print("No valid peaks detected!")
        return pd.DataFrame()  # Return an empty dataframe
    
    # For now, we'll focus on the first peak and assume that each jump corresponds to one peak
    peak = peaks[0]
    print(f"Processing Jump at Peak Index {peak}")
    
    # Define window around the peak (a smaller window around the peak)
    window_size = 100  # Smaller window size to focus only on the jump around the peak
    window_start = max(0, peak - window_size // 2)
    window_end = min(len(df), peak + window_size // 2)
    
    # Extract the window for analysis
    window = df.iloc[window_start:window_end].copy()
    
    # Look for zero-crossings to determine takeoff and landing points
    zero_crossings = np.where(np.diff(np.sign(window["Hdr_X_filtered"])))
    if len(zero_crossings[0]) < 2:
        print(f"Warning: Not enough zero-crossings detected for this jump.")
        return pd.DataFrame()  # Skip this jump
    
    takeoff_idx = zero_crossings[0][0]
    landing_idx = zero_crossings[0][-1]
    
    takeoff_time = window["Time (s)"].iloc[takeoff_idx]
    landing_time = window["Time (s)"].iloc[landing_idx]
    time_of_flight = landing_time - takeoff_time
    
    # Reduce scaling for TOF-based height calculation (scaled down to 0.2)
    scaling_factor = 0.2  # Lower the scaling factor further
    g = 9.81  # Gravity acceleration in m/s²
    height_tof = scaling_factor * (g * (time_of_flight / 2) ** 2)
    
    # Integration-based height calculation
    accel = window["Hdr_X_filtered"].values
    time = window["Time (s)"].values
    velocity = integrate.cumulative_trapezoid(accel, time, initial=0)
    displacement = integrate.cumulative_trapezoid(velocity, time, initial=0)
    jump_height_integrated = max(displacement) - min(displacement)
    
    # Energy-based height calculation
    v_takeoff = max(velocity)
    jump_height_energy = (v_takeoff ** 2) / (2 * g) if v_takeoff > 0 else 0
    
    # Calculate the mean of the three height calculations
    mean_jump_height = np.mean([height_tof, jump_height_integrated, jump_height_energy])
    
    # Convert takeoff time to Unix Epoch
    epoch_timestamp = df["Timestamp"].iloc[0] / 1000.0  # Unix Epoch of the first timestamp
    takeoff_unix_epoch = epoch_timestamp + takeoff_time
    
    # Convert to ISO 8601 string
    takeoff_iso = pd.to_datetime(takeoff_unix_epoch, unit='s').isoformat()
    
    # Create the result
    result = {
        "Jump #": 1,
        "Time of Flight (s)": time_of_flight,
        "Height (m) - TOF": height_tof,
        "Height (m) - Integration": jump_height_integrated,
        "Height (m) - Energy": jump_height_energy,
        "Mean Height (m)": mean_jump_height,
        "Takeoff ISO 8601": takeoff_iso
    }
    
    results_df = pd.DataFrame([result])
    print(f"Results DataFrame: {results_df}")
    
    return results_df

# Example usage
file_path = "06-03-2025 104801.txt"
results_df = process_jump_data(file_path)
print(results_df)
