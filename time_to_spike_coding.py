"""
time_to_spike_coding.py

TTFSCoding:
- Encodes pixel intensity into the time of a single spike within the simulation window.
- Brighter pixels (intensity near 1) fire very early.
- Darker pixels (intensity near 0) fire later or may not fire if time exceeds the set max_time window.

By default, max_time is 20 ms (0.02 s), so all spikes occur in [0, 20 ms].
"""

import numpy as np

class TTFSCoding:
    def __init__(self, duration=0.1, dt=0.001, max_time=0.02):
        """
        duration: total simulation time in seconds (e.g., 0.1 for 100 ms).
        dt: time step in seconds (e.g., 0.001 for 1 ms).
        max_time: maximum time (in seconds) when a spike can occur.
                  By default, 0.02 = 20 ms, so all spikes happen within the first 20 ms.
        """
        self.duration = duration
        self.dt = dt
        self.max_time = max_time

    def encode(self, image):
        """
        image: 2D array (28x28) with values in [0,1].

        Returns:
        spike_times_list: list of arrays, each array contains the single spike time for one neuron.
        """
        # Flatten the image to a 1D array of 784 pixels
        flat_image = image.flatten()

        spike_times_list = []

        for pixel_value in flat_image:
            if pixel_value > 0:
                # Map pixel_value (0..1) to spike_time in [0..max_time].
                # Bright pixels (near 1) spike near 0 ms, dark pixels (near 0) spike near max_time.
                # Using (1 - pixel_value) ensures brighter pixels fire earlier.
                spike_time = (1.0 - pixel_value) * self.max_time

                # Only add the spike if it occurs before the total duration
                if spike_time <= self.duration:
                    spike_times_list.append(np.array([spike_time]))
                else:
                    spike_times_list.append(np.array([]))
            else:
                # No spike for pixel_value = 0
                spike_times_list.append(np.array([]))

        return spike_times_list
