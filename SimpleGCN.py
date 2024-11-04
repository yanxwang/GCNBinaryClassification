import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# GCN Layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, input, adj):
        support = input @ self.weight  # Matrix multiplication
        output = torch.einsum('bij,bjd->bid', [adj, support])  # Graph convolution operation
        return output

# GCN Model
class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = self.gc2(x, adj)
        return x

def load_connectivity_matrix(path):
    return torch.tensor(np.loadtxt(path, dtype=np.float32))

def normalize_adj(adj):
    if adj.dim() == 2:
        rowsum = adj.sum(dim=1)
    elif adj.dim() == 3:
        rowsum = adj.sum(dim=2)
    else:
        raise ValueError("Adjacency matrix must be 2D or 3D")

    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    
    if adj.dim() == 2:
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    elif adj.dim() == 3:
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)

    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def load_all_data(base_path):
    data = []
    labels = []
    adj = []

    # Loop through the directories for AD and CN
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            fc_path = os.path.join(folder_path, 'FunctionalConnectivity.txt')
            sc_path = os.path.join(folder_path, 'StructuralConnectivity.txt')

            if os.path.exists(fc_path) and os.path.exists(sc_path):
                fc_matrix = load_connectivity_matrix(fc_path)
                sc_matrix = load_connectivity_matrix(sc_path)

                input_matrix = fc_matrix.unsqueeze(0)  # Add a batch dimension
                adj_matrix = normalize_adj(sc_matrix.unsqueeze(0))  # Add a batch dimension

                print(f"Folder: {folder}, FC shape: {fc_matrix.shape}, SC shape: {sc_matrix.shape}")
                data.append(input_matrix)
                adj.append(adj_matrix)

                # Labels: AD (1) or CN (0)
                labels.append(1 if 'AD' in folder else 0)

    # Concatenate all data and labels
    data = torch.cat(data, dim=0)  # Shape: (N, 1, 150, 150)
    adj = torch.cat(adj, dim=0)  # Shape: (N, 1, 150, 150)
    labels = torch.tensor(labels)  # Shape: (N,)

    print(f"Concatenated Data shape: {data.shape}")
    print(f"Concatenated Adj shape: {adj.shape}")
    print(f"Labels shape: {labels.shape}")

    return data, adj, labels

def train(model, data, labels, adj, num_epochs=100, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    train_losses = []
    y_true = []
    y_pred = []
    y_scores = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data, adj)

        # Output shape debugging
        # print(f"Output shape: {output.shape}")

        # Ensure output and labels have compatible shapes
        loss = F.cross_entropy(output.mean(dim=1), labels)

        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Predictions
        preds = torch.argmax(output.mean(dim=1), dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())
        y_scores.extend(output.mean(dim=1)[:, 1].detach().numpy())

    return train_losses, y_true, y_pred, y_scores

def plot_loss_curve(train_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    base_path = './'  # Set to your base directory containing AD and CN folders
    data, adj, labels = load_all_data(base_path)
    
    model = GCN(in_features=150, hidden_features=64, out_features=2)  # Adjust in_features based on your data
    train_losses, y_true, y_pred, y_scores = train(model, data, labels, adj, num_epochs=100, learning_rate=0.01)

    # Calculate and print metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # Plot results
    plot_loss_curve(train_losses)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_scores)