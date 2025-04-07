import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import NN_Structure_48 as NN

# ---------------------- Load the training data
filepath_in = r'C:\Users\Admin\Desktop\WHK CS NN code\traindata_npy\inputs_48.npy'
filepath_out = r'C:\Users\Admin\Desktop\WHK CS NN code\traindata_npy\outputs_48.npy'

filepath_val_in = r'C:\Users\Admin\Desktop\WHK CS NN code\valdata_npy\inputs_48.npy'
filepath_val_out = r'C:\Users\Admin\Desktop\WHK CS NN code\valdata_npy\outputs_48.npy'

x = np.load(filepath_in).astype(np.float32)
y = np.load(filepath_out).astype(np.float32)

# ----------------------------- Scale the data 
scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

x = scaler_x.fit_transform(x).reshape(-1, 3)
y = scaler_y.fit_transform(y).reshape(-1, 48)

joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

x_val = np.load(filepath_val_in).astype(np.float32)
y_val = np.load(filepath_val_out).astype(np.float32)

x_val = scaler_x.transform(x_val).reshape(-1, 3)
y_val = scaler_y.transform(y_val).reshape(-1, 48)

features = torch.tensor(x, dtype=torch.float)
labels = torch.tensor(y, dtype=torch.float).view(-1, 48)

features_val = torch.tensor(x_val, dtype=torch.float)
labels_val = torch.tensor(y_val, dtype=torch.float).view(-1, 48)

# --------------------- Create the model
torch.manual_seed(0)
np.random.seed(0)

net = NN.Net(af_output_flag=0) 
net.init_weights()

def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true + 1e-12))) 
criterion = log_cosh_loss

# criterion = nn.MSELoss()

optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.95, 0.999), weight_decay=5e-6)


loss_values = []
val_loss_values = []
best_val_loss = float('inf')
patience = 15
patience_counter = 0  
early_stopping_triggered = False
negative_values_log = {}  

for epoch in range(10000):
    net.train()
    optimizer.zero_grad()
    outputs_train = net(features)
    train_loss = criterion(outputs_train, labels)
    train_loss.backward()


    optimizer.step()

    #evalution
    net.eval()
    with torch.no_grad():
        outputs_val = net(features_val)
        val_loss = criterion(outputs_val, labels_val)


    negative_positions = torch.where(outputs_train < 0)
    num_negative_values = negative_positions[0].shape[0]
    negative_values_log[epoch + 1] = {
        "count": num_negative_values,
        "positions": negative_positions
    }

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0  
        torch.save(net.state_dict(), 'best_model_48_nw.pth')
    else:
        patience_counter += 1

    
    if (patience_counter >= patience) and (train_loss.item() < 8e-6):
        print(f'Early stopping triggered at epoch {epoch + 1}')
        print(f'Best Loss at epoch {epoch + 1 - patience}')
        print(f'Best Loss: {loss_values[-patience]}')
        early_stopping_triggered = True
        break

    loss_values.append(train_loss.item())
    val_loss_values.append(val_loss.item())

    if epoch % 100 == 0:
        print(f'Epoch {epoch + 1} - Loss: {train_loss.item()}')

if not early_stopping_triggered:
    print('Training completed without early stopping')

joblib.dump(loss_values, 'loss_values_48.pkl')
joblib.dump(negative_values_log, 'negative_values_log.pkl')

plt.plot(np.log10(loss_values), label='train_Loss', color='blue')
plt.plot(np.log10(val_loss_values), label='val_Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Log10(Loss)')
plt.title('48 output Training Loss')
plt.legend()
plt.show()

epochs = list(negative_values_log.keys())  
negative_counts = [negative_values_log[epoch]["count"] for epoch in epochs]  

plt.figure(figsize=(10, 5))
plt.plot(epochs, negative_counts, linestyle='-', color='b', label='Negative Value Count')

plt.title("Negative Value Count per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Number of Negative Values")
plt.legend()
plt.grid(True)
plt.show()

a = 1
