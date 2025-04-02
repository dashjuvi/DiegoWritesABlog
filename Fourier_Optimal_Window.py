import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters (same as previous)
duration_days = 30
duration_hours = duration_days * 24
sampling_rate = 3600  # in seconds (sampling every hour)
num_points = int((duration_hours * 3600) / sampling_rate)

# Generate time axis
time_index = pd.date_range(start='2023-10-01', periods=num_points, freq='h')

# Create the original signal with square-like behavior and smooth transitions
np.random.seed(0)  # For reproducibility

# Parameters for the square-like signal
low_value = 50
high_value = 200
num_sections = 20  # Number of sections for square-like behavior
signal_data = np.zeros(num_points)

# Create a mix of high and low blocks
for i in range(num_sections):
    # Determine if this section will be high or low
    if i % 2 == 0:
        signal_amp = np.random.uniform(high_value - 25, high_value + 25)  # Variability in high signals
    else:
        signal_amp = np.random.uniform(low_value - 10, low_value + 10)  # Variability in low signals
        
    section_length = np.random.randint(1, num_points // num_sections)  # Random section length

    # Make sure we don't exceed num_points
    start_index = (i * (num_points // num_sections)) % num_points
    end_index = min(start_index + section_length, num_points)

    # Fill the signal block with the chosen amplitude
    signal_data[start_index:end_index] = signal_amp

# Clip values to ensure non-negativity
signal_data = np.clip(signal_data, 0, None)

# Create a C2 beacon signal (sine wave) with jitter in amplitude and timing
beacon_amplitude = 500  # Amplitude of the sine beacon signal
jitter_amplitude = 5  # Amplitude of the jitter

# Generate the sine wave as the beacon signal
beacon_data = np.zeros(num_points)
current_time = 0
peak_times = []

while current_time < num_points:
    # Add jitter to the timing
    jittered_interval = np.random.uniform(20, 29) * 3600  # Jittered interval in seconds
    next_ping = int(current_time + jittered_interval / sampling_rate)
    
    if next_ping < num_points:
        beacon_data[next_ping] = beacon_amplitude * abs(np.sin(2 * np.pi * (1 / 24) * next_ping))
        peak_times.append(next_ping)
        current_time = next_ping
    else:
        break

# Add jitter to the beacon signal amplitude
jitter = jitter_amplitude * np.random.randn(num_points)
beacon_data_with_jitter = beacon_data + jitter

# Combine the beacon signal with the original signal
combined_signal = signal_data + beacon_data_with_jitter  # Ensuring it's a NumPy array

# Create a DataFrame for better plotting
data = pd.DataFrame(data={
    'Time': time_index,
    'Original Signal': signal_data,
    'C2 Beacon Signal': beacon_data_with_jitter,
    'Combined Signal': combined_signal
})

# Function to calculate FFT
def calculate_fft(signal):
    fft = np.fft.rfft(signal)  # Efficient FFT for real-valued signals
    fft_magnitude = np.abs(fft)  # Get the magnitude of the FFT
    return fft_magnitude

# Frequency bins for the FFT
nyquist_freq = 0.5 * (1 / (sampling_rate / 3600))  # Nyquist frequency in cycles per hour
num_window_bins = 24 * 7 // 2 + 1  # Frequency bins for each FFT window (1 week of data, 1-hour samples)
frequencies_window = np.linspace(0, nyquist_freq, num_window_bins)

# Expected frequency for the beacon
expected_frequency = 1 / 24  # cycles per hour

# Function to perform sliding window FFT
def sliding_window_fft(signal, window_size, step_size):
    num_points = len(signal)
    fft_results = []
    time_windows = []

    for start in range(0, num_points - window_size, step_size):
        end = start + window_size
        window_signal = signal[start:end]
        
        # Perform FFT for the current window
        fft = np.fft.rfft(window_signal)
        fft_magnitude = np.abs(fft)  # Get magnitude
        fft_results.append(fft_magnitude)
        
        # Capture the midpoint of the window for reference
        time_windows.append(data['Time'][start + window_size // 2])
    
    return np.array(fft_results), time_windows

# Step sizes to try
step_sizes = [1, 12, 24, 48, 70, 200]  # Different step sizes in hours (e.g., every 12 hours, 1 day, etc.)

# Plot all the results for different step sizes
plt.figure(figsize=(14, 10))

for step_size in step_sizes:
    # Apply sliding window FFT to the combined signal for each step size
    fft_sliding, _ = sliding_window_fft(combined_signal, 24 * 7, step_size)  # Use 1-week window

    # Take the average FFT for plotting
    avg_fft = np.mean(fft_sliding, axis=0)

    # Plot the FFT for this step size
    plt.plot(frequencies_window[1:], avg_fft[1:], label=f'Step Size = {step_size} hours')

plt.axvline(x=expected_frequency, color='k', linestyle='--', label='Expected Frequency')
plt.title('Comparison of Sliding Window FFTs for Different Step Sizes')
plt.xlabel('Frequency (cycles per hour)')
plt.ylabel('Magnitude')
plt.xlim(0, 0.08)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
