import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_metrics(Y_ts, y_ts):
   
    tf_ts = len(Y_ts)  # Total number of samples (assuming it's the length of Y_ts)

    # NMSE
    NMSE = np.mean((Y_ts - y_ts) ** 2) / np.mean(np.var(Y_ts, axis=0))

    # MSE
    MSE = mean_squared_error(Y_ts, y_ts)

    # R2
    R2 = 1 - (np.sum((Y_ts - y_ts) ** 2) / (np.sum(Y_ts ** 2) - (1 / tf_ts) * np.sum(Y_ts ** 2)))

    # RMSE
    RMSE = np.sqrt((1 / tf_ts) * np.sum((Y_ts - y_ts) ** 2))

    # MAPE
    MAPE = (1 / tf_ts) * np.sum(np.abs((Y_ts - y_ts) / Y_ts))

    # AE
    AE = (1 / tf_ts) * np.sum(np.abs(Y_ts - y_ts) / Y_ts)

    # VAF
    VAF = 1 - (np.var(y_ts - Y_ts) / np.var(y_ts))

    # NRMSE
    NRMSE = np.sqrt((1 / tf_ts) * np.sum((Y_ts - y_ts) ** 2)) / ((1 / tf_ts) * np.sum(Y_ts))

    return {
        'R2': R2,
        'RMSE': RMSE,
    }

def pinn_heatmap(nn_result):

    # Pivot the dataframe to create a grid for plotting
    pivot_table = nn_result.pivot(index='t', columns='x', values='u')

    # Create the heat map
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_table, extent=[nn_result['x'].min(), nn_result['x'].max(), nn_result['t'].min(), nn_result['t'].max()], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Heat Map of u at Different t and x')
    plt.savefig('heatmap.png', format='png')


def pinn_time_intervals(nn_result):

    nn_result_03 = nn_result[(nn_result['t'] > 0.2) & (nn_result['t'] < 0.3)]
    nn_result_05 = nn_result[nn_result['t']==0.5]
    nn_result_1 = nn_result[nn_result['t']==1]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot for T = 0.3
    axs[0].plot(nn_result_03['x'], nn_result_03['u'], label="T = 0.3")
    axs[0].set_title("T = 0.3")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("u")
    axs[0].legend()

    # Plot for T = 0.5
    axs[1].plot(nn_result_05['x'], nn_result_05['u'], label="T = 0.5")
    axs[1].set_title("T = 0.5")
    axs[1].set_xlabel("x")
    axs[1].legend()

    # Plot for T = 1
    axs[2].plot(nn_result_1['x'], nn_result_1['u'], label="T = 1")
    axs[2].set_title("T = 1")
    axs[2].set_xlabel("x")
    axs[2].legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.savefig('pinn_time_intervals.png', format='png')

def fem_vs_pinn(fem_result, nn_result):

    fem_result = fem_result[fem_result['t'] == 1]
    
    fem_result = fem_result.reset_index(drop=True)
    
    print(fem_result)
    

    plt.plot(fem_result['x'], fem_result['u'], label='FEM')
    plt.plot(nn_result['x'], nn_result['u'], label='NN')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("PINN vs. FEM For 1D Burgers Equation")
    plt.grid(True) 
    plt.savefig('fem_vs_pinns.png', format='png')

def relu_vs_tanh():

    # Load trial data from Excel file
    file_name = 'trials.xlsx'
    df = pd.read_excel(file_name)

        # Filter data for 'relu' and 'tanh' activation functions
    df_relu = df[df['Activation'] == 'relu']
    df_tanh = df[df['Activation'] == 'tanh']

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot ReLU data
    ax.scatter(df_relu['n_layers'], df_relu['n_units'], df_relu['Learning Rate'], color='blue', label='ReLU')
    # Plot Tanh data
    ax.scatter(df_tanh['n_layers'], df_tanh['n_units'], df_tanh['Learning Rate'], color='red', label='Tanh')

    ax.set_xlabel('Hidden Layers')
    ax.set_ylabel('Hidden Units')
    ax.set_zlabel('Learning Rate')
    ax.set_title('Learning Rate vs Hidden Layers and Units for ReLU and Tanh')
    ax.legend()

    # Annotate points with loss values
    for i in range(len(df_relu)):
        ax.text(df_relu.iloc[i]['n_layers'], df_relu.iloc[i]['n_units'], df_relu.iloc[i]['Learning Rate'], 
                f"{df_relu.iloc[i]['Loss after 200 epochs']:.4f}", fontsize=6, color='blue')
    for i in range(len(df_tanh)):
        ax.text(df_tanh.iloc[i]['n_layers'], df_tanh.iloc[i]['n_units'], df_tanh.iloc[i]['Learning Rate'], 
                f"{df_tanh.iloc[i]['Loss after 200 epochs']:.4f}", fontsize=6, color='red')

    # Save the plot in EPS format
    plt.savefig('learning_rate_3d_plot.png', format='eps')

def sigmoid_vs_relu_vs_tanh():
    # Load trial data from Excel file
    file_name = 'trials.xlsx'
    df = pd.read_excel(file_name)

    learning_rates = df['Learning Rate']
    values = df['Loss after 200 epochs']
    activation_functions = df['Activation']

    activation_colors = {'relu': 'blue', 'sigmoid': 'green', 'tanh': 'red'}
    colors = [activation_colors[act] for act in activation_functions]

    # Create scatter plot
    plt.scatter(learning_rates, values, c=colors, s=50, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss after 200 epochs')
    plt.title('Scatter Plot of Learning Rate vs. Loss after 200 epochs with Activation Functions')
    plt.grid(True)

    # Add legend for color key
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markersize=10, markerfacecolor=color) 
            for key, color in activation_colors.items()]
    plt.legend(title='Activation Function', handles=handles, loc='upper right', bbox_to_anchor=(1, 1))

    # Save the plot in PNG format
    plt.savefig('scatter_plot_trials.png', format='png')


def main():


    fem_result = pd.read_csv('./results/results_fdm.csv')
    nn_result = pd.read_csv('./results/results_1d_burgers.csv', sep=',')

    fem_vs_pinn(fem_result, nn_result)
    pinn_heatmap( nn_result)
    pinn_time_intervals(nn_result)
    relu_vs_tanh()
    sigmoid_vs_relu_vs_tanh()


if __name__ == "__main__":
    main()