import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Polygon
from IPython.display import HTML, display
import matplotlib.animation as animation
from google.colab import files
import io
from PIL import Image

# Upload drone image
print("Please upload a drone image (PNG or JPG):")
uploaded = files.upload()

# Get the path of the uploaded image
drone_image_path = list(uploaded.keys())[0]
drone_img = plt.imread(drone_image_path)

def create_drone_visualization_with_image(true_states, drone_img, dt=0.05):
    """
    Create drone visualization using the uploaded drone image
    """
    time = np.arange(len(true_states)) * dt

    # Set up the figure and axis
    plt.ioff()  # Turn off interactive mode for Colab
    fig, ax = plt.subplots(figsize=(15, 9))

    # Compute more precise axis limits based on trajectory
    max_x = np.max(true_states[:, 0]) + 0.2
    min_x = np.min(true_states[:, 0]) - 0.2
    max_y = np.max(true_states[:, 1]) + 0.2
    min_y = np.min(true_states[:, 1]) - 0.2

    # Set axis limits with some padding
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.set_title('Drone Simulation with Package Pickup and Delivery')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Z position (m)')
    ax.grid(True)

    # Goal position marker
    goal = plt.Circle((1.0, 0.0), 0.07, color='green', fill=True, label='Goal')
    ax.add_patch(goal)
    ax.plot([1.0], [0.0], 'go', markersize=10)

    # Calculate drone image size in data coordinates
    # Adjust these values to change drone size
    drone_width = 1.2
    drone_height = 1.2

    # Create a "blank" image for the drone that we'll update
    drone_plot = ax.imshow(np.zeros((10, 10)),
                         extent=[0, 0, 0, 0],  # Will be updated in animation
                         zorder=10)  # High zorder to keep drone on top

    # Create package - IMPROVED: more visible package with border
    package_width, package_height = 0.25, 0.25  # Slightly larger package
    package = plt.Rectangle((-0.03, 0.0), package_width, package_height,
                           color='black', fill=True,
                           linewidth=2, edgecolor='black',
                           zorder=9)  # High zorder but below drone
    ax.add_patch(package)

    # Add ground
    ground = plt.Line2D([min_x, max_x], [0, 0], color='black', linestyle='-', linewidth=2)
    ground_level = ax.text(max_x - 0.5, -0.05, 'Ground', color='black')
    ax.add_line(ground)

    # Time and event indicators
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    event_text = ax.text(0.3, 0.95, '', transform=ax.transAxes, color='g', fontweight='bold', fontsize=12)

    # Package pickup and drop locations - IMPROVED: more visible markers
    pickup_x, pickup_y = 0.0, 0.0
    dropoff_x, dropoff_y = 1.5, 0.0

    # Create distinctive markers for pickup and drop locations
    pickup_location = plt.Circle((pickup_x, pickup_y), 0.06, color='red', alpha=0.3, fill=True)
    dropoff_location = plt.Circle((dropoff_x, dropoff_y), 0.06, color='blue', alpha=0.3, fill=True)

    # Add flag markers to make locations more visible
    flag_height = 0.15
    # Pickup flag (red)
    ax.plot([pickup_x, pickup_x], [pickup_y, pickup_y+flag_height], 'r-', linewidth=3)
    ax.plot([pickup_x, pickup_x+0.1], [pickup_y+flag_height, pickup_y+flag_height-0.05], 'r-', linewidth=3)
    # Dropoff flag (blue)
    ax.plot([dropoff_x, dropoff_x], [dropoff_y, dropoff_y+flag_height], 'b-', linewidth=3)
    ax.plot([dropoff_x, dropoff_x+0.1], [dropoff_y+flag_height, dropoff_y+flag_height-0.05], 'b-', linewidth=3)

    # Better labels with background for readability
    pickup_text = ax.text(-0.5, -0.1, 'PICKUP', color='white', fontsize=10,
                         fontweight='bold', bbox=dict(facecolor='red', alpha=0.7))
    dropoff_text = ax.text(dropoff_x-0.15, dropoff_y-0.1, 'DROPOFF', color='white', fontsize=10,
                          fontweight='bold', bbox=dict(facecolor='blue', alpha=0.7))

    ax.add_patch(pickup_location)
    ax.add_patch(dropoff_location)

    # Add a legend with more elements
    ax.legend([goal, pickup_location, dropoff_location, package],
             ['Goal position', 'Pickup location', 'Dropoff location', 'Package'])

    # Package attached to drone flag and package dropped flag
    package_attached = False
    package_dropped = False
    package_drop_position = (dropoff_x-0.004, 0)  # Aligned with dropoff marker

    # Debug: print some trajectory info
    print(f"Trajectory start: ({true_states[0, 0]}, {true_states[0, 1]})")
    print(f"Trajectory end: ({true_states[-1, 0]}, {true_states[-1, 1]})")

    def init():
        time_text.set_text('')
        event_text.set_text('')

        # Initialize drone position
        x = true_states[0, 0]
        y = true_states[0, 1]
        # Set the drone image extent
        drone_plot.set_extent([x - drone_width/2, x + drone_width/2,
                            y - drone_height/2, y + drone_height/2])
        drone_plot.set_data(drone_img)  # Set the actual image data

        # Initialize package on ground at pickup location
        package.set_xy((pickup_x - package_width/2, 0))

        return time_text, event_text, drone_plot, package

    def animate(i):
        nonlocal package_attached, package_dropped, package_drop_position

        # Get current state for drone visualization
        current_idx = min(i, len(time)-1)
        x = true_states[current_idx, 0]
        y = true_states[current_idx, 1]
        theta = true_states[current_idx, 2]

        # Update drone position by updating the extent
        drone_plot.set_extent([x - drone_width/2, x + drone_width/2,
                            y - drone_height/2, y + drone_height/2])

        # Handle package pickup at step 2
        current_time = time[current_idx]

        # Package pickup - visual notification when near pickup point
        if np.abs(x - pickup_x) < 0.1 and not package_attached and not package_dropped:
            package_attached = True
            event_text.set_text('PACKAGE PICKED UP!')

        # Package delivery - visual notification when near dropoff point
        if np.abs(x - dropoff_x) < 0.1 and package_attached and not package_dropped:
            package_attached = False
            package_dropped = True
            # Store the drop position
            package_drop_position = (dropoff_x - package_width/2, 0)
            package.set_xy(package_drop_position)
            event_text.set_text('PACKAGE DROPPED OFF!')

        # Update package position
        if package_attached:
            # Center package below drone with cable effect
            package.set_xy((x - package_width/2, y - package_height - 0.1))
            # Draw a "cable" connecting drone to package
            if hasattr(animate, 'cable'):
                animate.cable.remove()
            animate.cable = ax.plot([x, x],
                                   [y - drone_height/2, y - package_height - 0.1],
                                   'k-', linewidth=1.5, alpha=0.7)[0]

        elif package_dropped:
            # Keep the package at the drop position
            package.set_xy(package_drop_position)
            # Remove cable when dropped
            if hasattr(animate, 'cable'):
                animate.cable.remove()
                delattr(animate, 'cable')
        else:
            # Keep at pickup position before pickup
            package.set_xy((pickup_x - package_width/2, 0))
            # No cable needed
            if hasattr(animate, 'cable'):
                animate.cable.remove()
                delattr(animate, 'cable')

        # Update time indicator
        time_text.set_text(f'Time: {current_time:.2f}s')

        # Add cable to the return objects only if it exists
        returns = [time_text, event_text, drone_plot, package]
        if hasattr(animate, 'cable'):
            returns.append(animate.cable)
        return returns

    # Create animation with slower speed
    frames = len(time)
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames,
                         interval=150, blit=True)

    # Display animation in the notebook
    plt.close()  # Prevent duplicate display
    return HTML(anim.to_jshtml()), anim

# Then update how you call and save:
html_animation, anim_obj = create_drone_visualization_with_image(true_states, drone_img)
display(html_animation)  # Display in the notebook

# Fix the save function to use the parameter name correctly
def save_animation_as_html(html_animation, filename='drone_simulation.html'):
    from google.colab import files

    # Write HTML content to file
    with open(filename, 'w') as f:
        f.write(html_animation.data)

    # Download the file
    files.download(filename)

    return f"Animation saved as {filename} and downloaded"

save_animation_as_html(html_animation, 'drone_simulation.html')
