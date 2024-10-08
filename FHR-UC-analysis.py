import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# Load data
data = pd.read_csv('Simulator_readings.csv')

# Convert time from milliseconds to seconds
data['Time (s)'] = data['Time(ms)'] / 1000

# 1. Plot Time vs FHR and Time vs UC
plt.figure(figsize=(12, 6))

# Time vs FHR
plt.subplot(2, 1, 1)
plt.plot(data['Time (s)'], data['Fhr1(BPM)'])
plt.title('Time vs FHR')
plt.xlabel('Time (s)')
plt.ylabel('FHR (bpm)')

# Time vs UC
plt.subplot(2, 1, 2)
plt.plot(data['Time (s)'], data['Uc(TOCO)'], color='orange')
plt.title('Time vs UC')
plt.xlabel('Time (s)')
plt.ylabel('UC (TOCO)')

plt.tight_layout()
plt.show()

# 2. FHR Analysis (Epoch-based FHR Analysis)
epoch_duration = 3.75  # in seconds
samples_per_epoch = int(epoch_duration * 4)  # 4 data points per second

# Group data into epochs and calculate average FHR and pulse interval
num_epochs = len(data) // samples_per_epoch
epoch_fhr = []
epoch_pulse_interval = []

for i in range(num_epochs):
    start_idx = i * samples_per_epoch
    end_idx = start_idx + samples_per_epoch
    epoch_data = data['Fhr1(BPM)'].iloc[start_idx:end_idx]
    avg_fhr = epoch_data.mean()
    pulse_interval = 60000 / avg_fhr  # in milliseconds
    epoch_fhr.append(avg_fhr)
    epoch_pulse_interval.append(pulse_interval)

# Convert lists to arrays for further analysis
epoch_fhr = np.array(epoch_fhr)
epoch_pulse_interval = np.array(epoch_pulse_interval)

# 3. UC Peak Detection
uc_data = data['Uc(TOCO)']

# Detect peaks in UC data
peaks, _ = find_peaks(uc_data, height=5)  # You can adjust the height threshold as needed

# Calculate peak widths at half maximum
peak_widths_result = peak_widths(uc_data, peaks, rel_height=0.5)

# Convert widths from indices to time in seconds
peak_widths_in_seconds = peak_widths_result[0] * 0.25  # 0.25 seconds per data point

# Count peaks with width greater than 30 seconds
wide_peaks = peak_widths_in_seconds[peak_widths_in_seconds > 30]
num_wide_peaks = len(wide_peaks)
avg_wide_peak_duration = np.mean(wide_peaks) if num_wide_peaks > 0 else 0

# Output results
print(f'Number of UC peaks with width > 30 seconds: {num_wide_peaks}')
print(f'Average duration of wide UC peaks: {avg_wide_peak_duration:.2f} seconds')
