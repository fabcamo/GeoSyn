import numpy as np
import matplotlib.pyplot as plt

# Function parameters
amplitude = 10            # Amplitude (A)
period = 500              # User-defined period (T)
horizontal_shift = 0     # Horizontal phase shift (h)
vertical_shift = 0       # Vertical shift (v)
function_type = "cos"    # Choose between "sin" or "cos"

# Calculate frequency from period
frequency = 2 * np.pi / period

# Generate x values covering a larger range (e.g., 10 periods)
x = np.linspace(0, 2 * period, 1000)  # 10 full periods

# Define the function
if function_type == "cos":
    y = amplitude * np.cos(frequency * (x - horizontal_shift)) + vertical_shift
else:
    y = amplitude * np.sin(frequency * (x - horizontal_shift)) + vertical_shift

# Original function for comparison
y_original = amplitude * np.cos(frequency * x) if function_type == "cos" else amplitude * np.sin(frequency * x)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f"{function_type}(x) with shifts", color="blue", linewidth=2)
plt.plot(x, y_original, label=f"Original {function_type}(x)", linestyle="--", color="gray")

# Highlight the period visually on the plot
plt.axvline(period, color="green", linestyle="--", label=f"Period = {period}")

# Axis labels and legend
plt.title(f"Visualization of {function_type}(x) with Amplitude, Period, and Phase Shifts", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
# limit the x axis to 512 distance
plt.xlim(0, 512)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.grid(alpha=0.4)
plt.show()
