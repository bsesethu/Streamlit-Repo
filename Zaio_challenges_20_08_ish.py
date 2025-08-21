# 21/08
#### STUDENT CODE CELL
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def plot_3d_visualizations():
    """
    Create 3D line plot, scatter plot, and contour plot.

    Args:
    None

    Returns:
    matplotlib.figure.Figure: The generated Matplotlib figure for validation
    """
    theta = np.linspace(-12, 12, 100)
    r = 2
    c = 3
    x_line = r * np.sin(theta)
    y_line = r * np.cos(theta)
    z_line = c * theta

    x = np.random.randn(100)
    y = np.random.randn(100)
    z = np.random.randn(100)

    x_grid = np.linspace(-6, 6, 30)
    y_grid = np.linspace(-6, 6, 30)
    
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.sin(np.sqrt(X ** 2 + Y **2))
    
    fig = plt.figure()
    
    ax = fig.add_subplot(131, projection= '3d')
    ax.plot3D(x_line, y_line, z_line, label= '3D Line (Helix)')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # plt.show()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(132, projection= '3d')
    ax1.scatter3D(x, y, z, c=z, cmap='viridis', label="3D Scatter")
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(133, projection= '3d')
    ax2.contour3D(X, Y, Z, 50, cmap= 'viridis')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    
    fig.suptitle('3D Visualizations')

    plt.show()
    return fig, fig1, fig2
plot_3d_visualizations()


# Tayob Version
def plot_3d_visualizations():
    """
    Create 3D line plot, scatter plot, and contour plot.
    
    Args:
        None
    
    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure for validation
    """
    # YOUR CODE HERE
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    theta = np.linspace(-12, 12, 100)
    r = 2
    c = 3
    x_line = r * np.sin(theta)
    y_line = r * np.cos(theta)
    z_line = c * theta
    ax.plot3D(x_line, y_line, z_line, label="3D Line (Helix)")

    x_scatter = np.random.randn(100)
    y_scatter = np.random.randn(100)
    z_scatter = np.random.randn(100)
    ax.scatter3D(x_scatter, y_scatter, z_scatter, c=z_scatter, cmap='viridis', label="3D Scatter")

    def f(x, y):
        return np.sin(np.sqrt(x**2 + y**2))
    
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    ax.contour3D(X, Y, Z, 50, cmap='viridis')

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("3D Visualizations")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

print(plot_3d_visualizations())
    