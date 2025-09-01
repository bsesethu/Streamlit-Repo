# 21/08
#### STUDENT CODE CELL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    

# 25/08
def plot_area_charts():
    """
    Create and return stacked and overlapped area charts for Covid data.

    Args:
    None

    Returns:
    tuple: Two Matplotlib figures for the stacked and overlapped area charts
    """
    Total_Daily_Infection= [1326, 1456, 1794, 1712, 1634, 1565]
    Vaccine_Jab_1 =  [651, 670, 710, 736, 722, 768]
    Vaccine_Jab_2 =  [322, 341, 361, 383, 399, 404]
    date_range = pd.date_range(start= '2021-04-01', end= '2021-04-06', freq= 'D')

    df = pd.DataFrame({'Total Daily Infection': Total_Daily_Infection, 'Vaccine_Jab_1': Vaccine_Jab_1, 'Vaccine_Jab_2': Vaccine_Jab_2, 'date_range': date_range})

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    plt.stackplot(df['date_range'], df['Total Daily Infection'], df['Vaccine_Jab_1'], df['Vaccine_Jab_2'], labels= ['Total Daily Infection', 'Vaccine_Jab_1', 'Vaccine_Jab_2'])
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Stacked Area Chart: Covid Data')
    plt.legend()
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(122)
    plt.fill_between(df['date_range'], 0, df['Total Daily Infection'], alpha= 0.5, label= 'Total Daily Infection')
    plt.fill_between(df['date_range'], 0, df['Vaccine_Jab_1'], alpha= 0.5, label= 'Vaccine_Jab_1')
    plt.fill_between(df['date_range'], 0, df['Vaccine_Jab_2'], alpha= 0.5, label= 'Vaccine_Jab_1')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Overlapped Area Chart: Covid Data')
    plt.legend()
    plt.show()

    return fig1, fig2

plot_area_charts()


# 26/08
def plot_sales_pie_chart(sales, labels):
    """
    Create and return a pie chart for sales proportions.

    Args:
    sales (list): List of integers representing sales data.
    labels (list): List of strings representing labels for the sales.

    Returns:
    matplotlib.figure.Figure: The created Matplotlib figure for validation.
    """
    # Find the index of the largest slice
    largest_slice_index = np.argmax(sales)
    
    # Create the explode tuple
    explode = [0] * len(sales)  # Initialize with zeros
    explode[largest_slice_index] = 0.1  # Explode the largest slice by 0.1
    
    fig = plt.figure()
    colours = ['limegreen', 'lavender', 'orange', 'yellow', 'seagreen']
    plt.pie(sales, labels= labels, autopct='%1.1f%%', colors= colours, explode = explode)
    plt.show()
    return fig

sales = [12000, 17000, 22300, 20500, 15600]
labels = ['Kansas', 'Montgomery', 'Marysville', 'Georgetown', 'Dearborn']

plot_sales_pie_chart(sales, labels)


# 26/08
#### STUDENT CODE CELL
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical_visualizations():
    """
    Create Boxplot, Violinplot, and Stripplot for categorical data.

    Args:
    None

    Returns:
    matplotlib.figure.Figure: The figure object containing the plots.
    """
    cont = ['Asia', 'Europe', 'Africa', 'Americas', 'Oceania']
    cont_list = []
    life_exp = []
    pop = []
    for i in range(100):
        cont_list.append(random.choice(cont))
        life_exp.append(random.randint(50, 85))
        pop.append(random.randint(1000000, 1000000000))
    df = pd.DataFrame({'continent': cont_list, 'lifeExp': life_exp, 'population': pop})
    # print(df.info())
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(311)
    sns.boxplot(x= 'continent', y= 'lifeExp', data= df[['continent', 'lifeExp']])
    plt.title('Categorical Data Visualization')
    plt.xlabel('Continents')
    plt.ylabel('Life Expectancy')

    ax3 = fig.add_subplot(312)
    sns.violinplot(x= 'continent', y= 'lifeExp', data= df[['continent', 'lifeExp']])
    plt.title('Categorical Data Visualization')
    plt.xlabel('Continents')
    plt.ylabel('Life Expectancy')

    ax2 = fig.add_subplot(313)
    sns.stripplot(x= 'continent', y= 'lifeExp', data= df[['continent', 'lifeExp']], jitter= True)
    plt.title('Categorical Data Visualization')
    plt.xlabel('Continents')
    plt.ylabel('Life Expectancy')

    plt.show()
    return fig
plot_categorical_visualizations()


