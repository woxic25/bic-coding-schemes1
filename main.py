"""
main.py

Replicates Figure 3 from:
"Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" by Guo et al.

- Top row: Input spike patterns (raster plots) of an example input digit "5" for different coding methods in a 100 ms time window.
- Bottom row: Average input spike counts over time across all the training input images (MNIST) for different coding methods in a 100 ms time window.

Coding methods: Rate, TTFS, Phase, Burst
"""

import numpy as np
import matplotlib.pyplot as plt

# Import coding scheme classes
from rate_coding import RateCoding
from time_to_spike_coding import TTFSCoding
from phase_coding import PhaseCoding
from burst_coding import BurstCoding

# Load MNIST dataset (Keras)
from tensorflow.keras.datasets import mnist

# --- Parameters ---
DURATION = 0.1  # 100 ms
DT = 0.001      # 1 ms time step
NUM_TRAIN_MAX = None  # Use the full training set or set an integer to limit

# 1) Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Normalize pixel values to [0,1]
x_train /= 255.0
x_test /= 255.0

if NUM_TRAIN_MAX is not None:
    x_train = x_train[:NUM_TRAIN_MAX]
    y_train = y_train[:NUM_TRAIN_MAX]

# 2) Initialize coders
# You can tweak parameters like thresholds or frequencies inside each coder.
rate_coder = RateCoding(duration=DURATION, dt=DT)
ttfs_coder = TTFSCoding(duration=DURATION, dt=DT)
phase_coder = PhaseCoding(duration=DURATION, dt=DT, num_cycles=10)  # 10 cycles in 100 ms, can adjust
burst_coder = BurstCoding(duration=DURATION, dt=DT)

# --- Plot Setup ---
fig, axs = plt.subplots(2, 4, figsize=(14, 6))

# Example digit "5" from the training set
example_indices = np.where(y_train == 5)[0]
example_index = example_indices[0] if len(example_indices) > 0 else 0
example_image = x_train[example_index]

# 3) Raster Plots for Example Digit
coders = [rate_coder, ttfs_coder, phase_coder, burst_coder]
titles = ["Rate coding\nExample digit '5'",
          "TTFS coding\nExample digit '5'",
          "Phase coding\nExample digit '5'",
          "Burst coding\nExample digit '5'"]

for i, coder in enumerate(coders):
    spike_times_list = coder.encode(example_image)
    ax_top = axs[0, i]

    # spike_times_list is a list of arrays, each array is the spike times for one neuron
    # Plot raster
    neuron_indices = []
    spike_times = []
    for neuron_idx, times in enumerate(spike_times_list):
        neuron_indices.extend([neuron_idx] * len(times))
        spike_times.extend(times)

    ax_top.scatter(spike_times, neuron_indices, s=2, c='black')
    ax_top.set_title(titles[i])
    ax_top.set_xlabel("Time (ms)")
    ax_top.set_ylabel("Input neuron index")
    ax_top.set_xlim([0, 100])  # 100 ms
    ax_top.set_ylim([0, 784])  # 28x28 MNIST

# 4) Average Spike Count Over Time (All Training Images)
for i, coder in enumerate(coders):
    # We'll sample all or NUM_TRAIN_MAX images
    spike_counts_over_time = np.zeros(int(DURATION / DT))  # e.g., 100 steps if DURATION=0.1, DT=0.001

    for img in x_train:
        spike_times_list = coder.encode(img)
        # Build a time histogram of spikes
        # For each neuron, increment counts at the spike times
        time_bins = np.arange(0, DURATION + DT, DT)
        hist = np.zeros_like(spike_counts_over_time)
        for times in spike_times_list:
            bin_indices = np.floor(np.array(times) / DT).astype(int)
            bin_indices = bin_indices[bin_indices < len(hist)]
            hist[bin_indices] += 1
        spike_counts_over_time += hist

    # Average across all training images
    spike_counts_over_time /= len(x_train)

    ax_bottom = axs[1, i]
    ax_bottom.plot(np.arange(0, DURATION, DT) * 1000, spike_counts_over_time)  # convert s to ms
    ax_bottom.set_xlabel("Time (ms)")
    ax_bottom.set_ylabel("Average spike counts")
    ax_bottom.set_xlim([0, 100])

axs[0, 0].set_ylabel("Input neuron index")  # top-left
axs[1, 0].set_ylabel("Average spike counts")  # bottom-left

plt.tight_layout()
plt.show()
