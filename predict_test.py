import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NN_Structure_48 as NN
import joblib
from mpl_toolkits.mplot3d import Axes3D
import os

test_filepath_in = r'C:\Users\Admin\Desktop\WHK CS NN code\testdata_npy\inputs_48.npy'
test_filepath_out = r'C:\Users\Admin\Desktop\WHK CS NN code\testdata_npy\outputs_48.npy'

pos_filepath = r'C:\Users\Admin\Desktop\WHK CS NN code\traindata_npy\coordinate_index_48.npy'

test_x = np.load(test_filepath_in)
test_y = np.load(test_filepath_out)

test_x = np.array(test_x, dtype=np.float32)
test_y = np.array(test_y, dtype=np.float32)

scaler_x = joblib.load(r'C:\Users\Admin\Desktop\WHK CS NN code\scaler_x.pkl')
scaler_y = joblib.load(r'C:\Users\Admin\Desktop\WHK CS NN code\scaler_y.pkl')

test_x_n = scaler_x.transform(test_x).reshape(-1, 3)

test_y_n = scaler_y.transform(test_y).reshape(-1, 48)

test_x_n = torch.tensor(test_x_n, dtype=torch.float).view(-1, 3)
test_y_n = torch.tensor(test_y_n, dtype=torch.float).view(-1, 48)

net = NN.Net(af_output_flag = 0)

def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true + 1e-12))) 
criterion = log_cosh_loss
# criterion = nn.MSELoss()
net.load_state_dict(torch.load(r'C:\Users\Admin\Desktop\WHK CS NN code\best_model_48_nw.pth', weights_only=True))

net.eval()

with torch.no_grad():
    test_outputs = net(test_x_n)
    test_outputs_zero = torch.clamp(test_outputs, min=0)

    test_loss = criterion(test_outputs, test_y_n)
    test_loss_zero = criterion(test_outputs_zero, test_y_n)

print(f'Test Loss: {test_loss.item():.16f}')
print(f'Test Loss processed: {test_loss_zero.item():.16f}')


test_outputs = scaler_y.inverse_transform(test_outputs.numpy())

test_outputs = np.clip(test_outputs, 0, None)



pos = np.load(pos_filepath)
df_pos = pd.DataFrame(pos)
df_actual = pd.DataFrame(test_y.T)
df_predict = pd.DataFrame(test_outputs.T)



ind = df_pos.sort_values(by=[1, 0], ascending=[False, True]).index

save_folder = '48matrix_values_PE'
save_plot_folder = 'plot'
for i in range(39):
    T, A, YM = test_x[i]
    actual_filename = f'T{T:.2f} A{A:.3f} YM{int(YM)} actual values.xlsx'
    predicted_filename = f'T{T:.2f} A{A:.3f} YM{int(YM)} predicted values.xlsx'
    percentage_error_filename = f'T{T:.2f} A{A:.3f} YM{int(YM)} percentage error.xlsx'
    fig_title_actual = f'XZ Plane T{T:.2f} A{A:.3f} YM{int(YM)} for Actual Values'
    fig_title_predicted = f'XZ Plane T{T:.2f} A{A:.3f} YM{int(YM)} for Predicted Values (Size = Error)'
    save_plot_title = f'XZ_Plane_T{T:.2f}_A{A:.3f}_YM{int(YM)}_for_Comparison.png'



    test_i_actual = df_actual[i].iloc[ind].values
    matrix_1 = test_i_actual.reshape(6,8)


    test_i_pre = df_predict[i].iloc[ind].values
    matrix_2 = test_i_pre.reshape(6,8)

    df_matrix_1 = pd.DataFrame(matrix_1)
    df_matrix_2 = pd.DataFrame(matrix_2)

    percentage_error = np.abs(matrix_2 - matrix_1) / np.abs(matrix_1) * 100

    df_percentage_error = pd.DataFrame(percentage_error)

    x = df_pos[0].iloc[ind].values
    z = df_pos[1].iloc[ind].values
    values_actual = pd.DataFrame(test_i_actual)
    values_predict = pd.DataFrame(test_i_pre)
    error_values = percentage_error.reshape(-1,1)
    error_scaled = (error_values - error_values.min()) / (error_values.max() - error_values.min()) * 100


    vmin = min(values_actual.values.min(), values_predict.values.min())
    vmax = max(values_actual.values.max(), values_predict.values.max())

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    sc = plt.scatter(x, z, c=values_actual, cmap='rainbow', vmin=vmin, vmax=vmax)
    color_bar = plt.colorbar(sc)
    color_bar.set_label('Value')  
    plt.title(fig_title_actual)
    plt.xlabel('X')
    plt.ylabel('Z')

    max_error_idx = np.argmax(error_scaled)
    min_error_idx = np.argmin(error_scaled)

    # Get the x and z coordinates for these max and min error points
    max_error_x = x[max_error_idx]
    max_error_z = z[max_error_idx]
    min_error_x = x[min_error_idx]
    min_error_z = z[min_error_idx]

    # Get the actual error values at these points
    max_error_value = error_values[max_error_idx].item()
    min_error_value = error_values[min_error_idx].item()

    plt.subplot(1, 2, 2)
    sc = plt.scatter(x, z, c=values_predict, cmap='rainbow', s=error_scaled+20, alpha=1, vmin=vmin, vmax=vmax)
    color_bar = plt.colorbar(sc)
    color_bar.set_label('Value')  
    plt.title(fig_title_predicted)
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.annotate(f'Max Error: {max_error_value:.2f}%', 
             (max_error_x, max_error_z), 
             textcoords="offset points", 
             xytext=(0, 10), 
             ha='center', 
             fontsize=10, 
             color='black', 
             weight='bold')

    plt.annotate(f'Min Error: {min_error_value:.2f}%', 
             (min_error_x, min_error_z), 
             textcoords="offset points", 
             xytext=(0, 10), 
             ha='center', 
             fontsize=10, 
             color='black', 
             weight='bold')

    
    
    plt.tight_layout()


    df_matrix_1.to_excel(os.path.join(save_folder, actual_filename), header=False, index=False)
    df_matrix_2.to_excel(os.path.join(save_folder, predicted_filename), header=False, index=False)
    df_percentage_error.to_excel(os.path.join(save_folder, percentage_error_filename), header=False, index=False)
    save_path = os.path.join(save_plot_folder, save_plot_title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 

    print(f'Test {i+1} completly saved.')



