import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
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

# Parameters for sliding window
window_size = 24 * 7  # Number of points in the window (e.g., 1 week of data if sampling every hour)
step_size_overlap = 24 # Overlap step (e.g., move by 1 day, 24 points at a time)
step_size_no_overlap = window_size  # Non-overlapping step (move by the entire window size)

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

# Apply sliding window FFT to the combined signal with and without overlap
fft_sliding_overlap, window_times_overlap = sliding_window_fft(combined_signal, window_size, step_size_overlap)
fft_sliding_no_overlap, window_times_no_overlap = sliding_window_fft(combined_signal, window_size, step_size_no_overlap)

# Frequency bins for the FFT
nyquist_freq = 0.5 * (1 / (sampling_rate / 3600))  # Nyquist frequency in cycles per hour
num_window_bins = window_size // 2 + 1  # Frequency bins for each FFT window
frequencies_window = np.linspace(0, nyquist_freq, num_window_bins)

# Expected frequency for the beacon
expected_frequency = 1 / 24  # cycles per hour

# Plot original signal, beacon signal, combined signal, and sliding window FFTs
plt.figure(figsize=(14, 18))

# Plot: Original signal
plt.subplot(5, 1, 1)
plt.plot(data['Time'], data['Original Signal'], label='Original Signal', color='b', alpha=0.8)
plt.title('Original Signal With Randomized High-Low Blocks Over 30 Days')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.grid()
plt.legend()

# Plot: C2 Beacon signal with jitter
plt.subplot(5, 1, 2)
plt.plot(data['Time'], data['C2 Beacon Signal'], label='C2 Beacon Signal with Jitter', color='r', alpha=0.8)
plt.title('C2 Beacon Signal with Jitter Over 30 Days')
plt.ylabel('Beacon Signal Value')
plt.grid()
plt.legend()

# Plot: Combined signal
plt.subplot(5, 1, 3)
plt.plot(data['Time'], data['Combined Signal'], label='Combined Signal', color='g', alpha=0.8)
plt.title('Combined Signal Over 30 Days')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.grid()
plt.legend()

# FFT Magnitude for the combined signal (Static)
fft_combined = calculate_fft(combined_signal)

# Fixing the plotting of FFT (make sure the frequency axis matches the length of the FFT result)
frequencies_combined = np.linspace(0, nyquist_freq, len(fft_combined))

plt.subplot(5, 1, 4)
plt.plot(frequencies_combined[1:], fft_combined[1:], label='FFT of Combined Signal (Static)', color='g', alpha=0.8)
plt.axvline(x=expected_frequency, color='k', linestyle='--', label='Expected Frequency')
plt.title('FFT of Combined Signal (Static)')
plt.xlabel('Frequency (cycles per hour)')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()

# Plot: Sliding window FFT results (with overlap and no overlap)
plt.subplot(5, 1, 5)
avg_fft_overlap = np.mean(fft_sliding_overlap, axis=0)
avg_fft_no_overlap = np.mean(fft_sliding_no_overlap, axis=0)

# Ensure same length for frequencies and FFT results
# Slice frequencies_window if needed
frequencies_window_plot = frequencies_window[:len(avg_fft_overlap)]

plt.plot(frequencies_window_plot[1:], avg_fft_overlap[1:], label='Sliding Window FFT (Overlap)', color='b', alpha=0.8)
plt.plot(frequencies_window_plot[1:], avg_fft_no_overlap[1:], label='Sliding Window FFT (No Overlap)', color='r', alpha=0.8)
plt.axvline(x=expected_frequency, color='k', linestyle='--', label='Expected Frequency')
plt.title('Comparison: Sliding Window FFT (Overlap vs No Overlap)')
plt.xlabel('Frequency (cycles per hour)')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

