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
high_value = 150
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

# Create a C2 beacon signal (sine wave) with Amplitude only jitter
beacon_frequency = 1 / 24  # 1 ping per day in cycles per hour
beacon_amplitude = 100  # Amplitude of the sine beacon signal
jitter_amplitude = 10  # Amplitude of the jitter

# Generate the sine wave as the beacon signal
beacon_data = beacon_amplitude * abs(np.sin(2 * np.pi * beacon_frequency * np.arange(num_points)))

# Add jitter to the beacon signal
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

# Calculate FFT for each signal
fft_original = calculate_fft(signal_data)
fft_beacon = calculate_fft(beacon_data_with_jitter)
fft_combined = calculate_fft(combined_signal)

# Calculate Nyquist frequency properly
nyquist_freq = 0.5 * (1 / (sampling_rate / 3600))  # Nyquist frequency in cycles per hour
num_bins = len(fft_combined)
bin_width = nyquist_freq / num_bins

# Calculate frequency bins correctly
frequencies = np.arange(num_bins) * bin_width  # Gives frequencies in cycles per hour

# Create plots
plt.figure(figsize=(14, 16))

# Plot: Original signal
plt.subplot(4, 1, 1)
plt.plot(data['Time'], data['Original Signal'], label='Original Signal', color='b', alpha=0.8)
plt.title('Network Traffic Over 30 Days')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.grid()
plt.legend()

# Plot: C2 Beacon signal with jitter
plt.subplot(4, 1, 2)
plt.plot(data['Time'], data['C2 Beacon Signal'], label='C2 Beacon Signal with Jitter', color='r', alpha=0.8)
plt.title('C2 Beacon Signal with Amplitude Jitter Over 30 Days')
plt.ylabel('Beacon Signal Value')
plt.grid()
plt.legend()

# Plot: Combined signal
plt.subplot(4, 1, 3)
plt.plot(data['Time'], data['Combined Signal'], label='Combined Signal', color='g', alpha=0.8)
plt.title('Combined Signal Over 30 Days')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.grid()
plt.legend()

# Plot FFT Magnitude for each signal
plt.subplot(4, 1, 4)
plt.plot(frequencies[1:], fft_combined[1:], label='FFT of Combined Signal', color='g')  # Combined signal
plt.plot(frequencies[1:], fft_original[1:], label='FFT of Original Signal', color='b', alpha=0.7)  # Original signal
plt.plot(frequencies[1:], fft_beacon[1:], label='FFT of Beacon Signal without Jitter', color='r', alpha=0.7)  # Beacon signal

# Title and labels for the FFT plot
plt.title('FFT of Signals (Excluding 0 Hz)')
plt.xlabel('Frequency (cycles per hour)')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()

# Show plots
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# Print bin width
print(f'Bin width: {bin_width}')
