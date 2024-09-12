import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import InterpolatedUnivariateSpline  # Import spline interpolation
from scipy.stats import norm  # Import normal distribution

# Constants
wavelength = 0.5  # Wavelength
k = 2 * np.pi / wavelength  # Wave number
num_points = 500  # Number of points for the grid
slit1_position = np.array([-0.5, -10])  # Position of the first slit
slit2_position = np.array([0.5, -10])   # Position of the second slit
frames = 1000  # Number of frames in the animation

# Create a grid of points
x = np.linspace(-10, 10, num_points)  # X coordinates
y = np.linspace(-10, 10, num_points)  # Y coordinates
X, Y = np.meshgrid(x, y)  # Create a 2D grid

# Calculate distances from each point on the grid to the slits
r1 = np.sqrt((X - slit1_position[0]) ** 2 + (Y - slit1_position[1]) ** 2)
r2 = np.sqrt((X - slit2_position[0]) ** 2 + (Y - slit2_position[1]) ** 2)

# Initialize the figure for the animation
fig, ax = plt.subplots(figsize=(10, 10))
cax_res = ax.imshow(np.zeros_like(r1), extent=(-10, 10, -10, 10), cmap='viridis', vmin=-2, vmax=2, alpha=0.5, origin='lower', interpolation='bilinear')
plt.colorbar(cax_res, ax=ax, label='Wave Amplitude')
plt.title('Wave Interference Pattern with elementary particles')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')

# Ball properties
num_balls_to_add = 1000
balls = []  # List to keep track of balls
hit_positions = []  # List to log hit x-positions

speed = 0.5  # Speed of the balls
start_delay = 0.1  # Number of frames to wait before starting the next ball

# Initialize balls with random angles and staggered start
for i in range(num_balls_to_add):
    angle = np.random.uniform(-np.pi / 8, np.pi / 8)  # Random angle for offset
    ball = {
        'x': 0,       # Starting x position
        'y': -10,     # Start at y = -10
        'angle': angle,  # Store angle for movement direction
        'line': ax.plot(0, -10, 'bo', markersize=0.5)[0],  # Initial position of the ball
        'start_frame': i * start_delay  # Delay start based on ball index
    }
    balls.append(ball)

# Function to update the frame
def update(frame):
    global ani  # Reference to the animation object to stop it later
    # Calculate the wave functions with varying phase
    wave1 = np.sin(k * r1 - frame * 0.1)  # Moving wave from slit 1
    wave2 = np.sin(k * r2 - frame * 0.1)  # Moving wave from slit 2
    
    # Calculate the resultant wave
    resultant_wave = wave1 + wave2

    # Update wave images
    cax_res.set_array(resultant_wave)

    # Update balls' positions
    for ball in balls[:]:  # Use a copy of the list for safe iteration
        if frame >= ball['start_frame']:  # Only update if it's time for this ball to start
            x_idx = int(np.clip((ball['x'] + 10) / 20 * (num_points - 1), 0, num_points - 1))
            y_idx = int(np.clip((ball['y'] + 10) / 20 * (num_points - 1), 0, num_points - 1))

            # Check the wave amplitude at the ball's current position
            current_amplitude = resultant_wave[y_idx, x_idx]

            if np.abs(current_amplitude) < 0.1:  # Ball is in a low amplitude area (0 amplitude)
                region = resultant_wave[max(0, y_idx - 5):min(num_points, y_idx + 5), 
                                         max(0, x_idx - 5):min(num_points, x_idx + 5)]
                max_amplitude_index = np.unravel_index(np.argmax(region.flatten()), region.shape)
                max_amplitude_y = max_amplitude_index[0] + (y_idx - 5)
                max_amplitude_x = max_amplitude_index[1] + (x_idx - 5)

                ball['x'] += 5 * (x[max(0, min(num_points - 1, max_amplitude_x))] - ball['x'])
                ball['y'] += 1 * (y[max(0, min(num_points - 1, max_amplitude_y))] - ball['y'])
            else:
                ball['y'] += speed * np.cos(ball['angle'])  # Update vertical position
                ball['x'] += speed * np.sin(ball['angle'])  # Slight horizontal movement

            if ball['y'] >= 10:
                hit_positions.append(ball['x'])  # Log the hit position
                balls.remove(ball)  # Remove the ball from the list
            else:
                ball['line'].set_data([ball['x']], [ball['y']])  # Update ball's position on the plot

    if not balls:
        ani.event_source.stop()  # Stop the animation
    
    return cax_res, *[ball['line'] for ball in balls]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, repeat=False)

# Show the animation
plt.show()

# After the animation, plot the hitting probabilities
plt.figure(figsize=(10, 5))
hist_data, bins = np.histogram(hit_positions, bins=np.linspace(-10, 10, 200), density=True)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Plot histogram
plt.hist(hit_positions, bins=bins, density=True, alpha=0.25, color='g', edgecolor='black', label='Hit Distribution')

# Fit a spline to the histogram data
spline = InterpolatedUnivariateSpline(bin_centers, hist_data, k=3)  # k=3 for cubic spline
x_smooth = np.linspace(-10, 10, 500)
y_smooth = spline(x_smooth)


# Fit a normal distribution curve to the hit positions
mean = np.mean(hit_positions)
std_dev = np.std(hit_positions)
norm_dist = norm.pdf(x_smooth, mean, std_dev)  # PDF of the normal distribution


plt.title('Hit Distribution')
plt.xlabel("Position")
plt.ylabel('Density')
plt.xlim(-10, 10)
plt.ylim(0, 1)
plt.grid(True)
plt.axhline(0, color='black', linewidth=1)  # Line at y=0 for visibility
plt.legend()


plt.show()
