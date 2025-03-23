"""
phase_coding.py

PhaseCoding:
- Encodes pixel intensity into the phase of the spike within multiple cycles in a given time window.
- If num_cycles=10 over a 100 ms window, each cycle is 10 ms. The spike time depends on the pixel intensity within each cycle.
"""

import numpy as np

class PhaseCoding:
    def __init__(self, duration=0.1, dt=0.001, num_cycles=10):
        """
        duration: total simulation time in seconds (e.g., 0.1 for 100 ms)
        dt: time step in seconds
        num_cycles: how many phase cycles occur within the duration
        """
        self.duration = duration
        self.dt = dt
        self.num_cycles = num_cycles

    def encode(self, image):
        """
        image: 2D array (28x28) with values in [0,1]

        Returns:
        spike_times_list: list of arrays, each array contains spike times for one neuron.
        """
        # Flatten the image to [784]
        flat_image = image.flatten()

        # Determine cycle length
        total_time_ms = self.duration * 1000.0  # e.g., 100 ms
        cycle_length_ms = total_time_ms / self.num_cycles  # e.g., 100/10 = 10 ms per cycle
        cycle_length_s = cycle_length_ms / 1000.0

        spike_times_list = []

        # For each pixel/neuron, compute spike times across multiple cycles
        for pixel_value in flat_image:
            # If pixel_value = 0, might skip spiking
            # If pixel_value = 1, spike early in each cycle
            # Map pixel_value to a phase time offset [0, cycle_length_s]
            # e.g., offset = (1 - pixel_value) * cycle_length_s
            # (If you want bright pixels to spike earlier, invert as needed.)

            if pixel_value <= 0:
                # No spikes if intensity = 0
                spike_times_list.append(np.array([]))
                continue

            # You can invert the offset if you want bright pixels earlier or later:
            offset_in_cycle = (1.0 - pixel_value) * cycle_length_s

            # Build spike times across all cycles
            times = []
            for c in range(self.num_cycles):
                start_of_cycle = c * cycle_length_s
                spike_time = start_of_cycle + offset_in_cycle
                # Check if spike_time <= total duration
                if spike_time <= self.duration:
                    times.append(spike_time)

            spike_times_list.append(np.array(times))

        return spike_times_list
