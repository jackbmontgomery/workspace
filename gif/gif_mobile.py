import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
import os

# Define the sine function range
x = np.linspace(0, 10 * np.pi, 100)  # Makes the x-axis much larger than the y
frames = []

# Create frames for the GIF
for i in tqdm(range(len(x))):
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.plot(x[:i], np.sin(x[:i]), color="black")
    ax.set_xlim(0, 10 * np.pi)
    ax.set_ylim(-1.1, 1.1)  # Keep y-axis limited to emphasize flatness
    ax.set_axis_off()
    fig.patch.set_alpha(0)

    # Save frame to a temporary file
    frame_path = f"frame_{i}.png"
    plt.savefig(
        frame_path, bbox_inches="tight", pad_inches=0.1, transparent=True, dpi=500
    )
    plt.close()
    frames.append(imageio.imread(frame_path))
    os.remove(frame_path)

# Save as a GIF
imageio.mimsave("sine_wave_mobile.gif", frames, duration=0.05)

print("GIF saved as sine_wave.gif")
