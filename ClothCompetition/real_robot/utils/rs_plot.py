import numpy as np
import matplotlib.pyplot as plt

def plot_pc(file_name, dir):

    point_cloud = np.load(dir+'/{}.npy'.format(file_name))
    # First, convert your point cloud to a numpy array for easier manipulation
    point_cloud_np = np.array(point_cloud)

    # Split your NumPy array into positions (x, y, z) and colors (r, g, b)
    positions = point_cloud_np[:, :3]

    # Create a new matplotlib figure and axis.
    fig = plt.figure()
    # window size to be square
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the x, y, and z coordinates and the color information
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=[0,0,1], s=1)  # s is the size of the points

    ax.set_xlim3d(-0, 1.0)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0, 1.0)

    # Set labels for axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()

def plot_depth(file_name):

    depth = np.load('../log/{}.npy'.format(file_name))
    plt.imshow(depth)
    plt.show()



if __name__ == '__main__':
    # plot_depth('masked_depth')
    plot_pc('cloth_pc', dir='../log')

