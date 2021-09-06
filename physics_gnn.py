import numpy as np
import torch
from torch.nn import Linear, Dropout
from torch import tensor
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GatedGraphConv, LEConv, APPNP
from torch_geometric.data.data import Data
from torch_geometric.data import DataLoader
import random
import copy


# Setting up network
class GCN(torch.nn.Module):
    def __init__(self, n_features, n_outputs, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GraphConv(n_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.linear1 = Linear(hidden_channels, n_outputs)
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.linear1(x)
        return x


def k_fold_train(train_set, k, batch_size, min_val_loss, model, trained_model, optimizer, criterion):
    random.shuffle(train_set)
    train_size, val_size = int(len(train_set)*(k-1)/k), int(len(train_set)/k)

    train_loss = 0
    val_loss = 0
    for fold in range(k):
        train_lower, val_lower = int(fold * train_size/k), int(train_size + fold * val_size/k)
        train_upper, val_upper = int((fold+1) * train_size/k), int(train_size + (fold+1) * val_size/k)
        train_loader = DataLoader(train_set[train_lower:train_upper], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(train_set[val_lower:val_upper], batch_size=batch_size, shuffle=False)

        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            x, y, edge_index = data.x, \
                data.y, \
                data.edge_index
            out = model(x, edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss

        model.eval()
        for data in val_loader:
            x, y, edge_index = data.x, \
                               data.y, \
                               data.edge_index
            out = model(x, edge_index)
            val_loss += criterion(out, y)
    train_loss /= (k * len(train_loader))
    val_loss /= (k * len(val_loader))

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        trained_model.load_state_dict(model.state_dict())

    return train_loss, val_loss, min_val_loss, trained_model


def main():
    # Initialising parameters
    n_train = 5000
    n_features = 4
    hidden_channels = 32
    n_outputs = 2
    batch_size = 100

    # Checking CUDA
    print('CUDA available:', torch.cuda.is_available())
    device = torch.device("cuda:0")

    # Data setup
    x_train_files = np.load('x_train.npz')
    y_train_files = np.load('y_train.npz')
    edge_index_files = np.load('edge_index.npz')

    x_train = [np.array([]) for _ in range(n_train)]
    y_train = [np.array([]) for _ in range(n_train)]
    edge_indices = [np.array([]) for _ in range(n_train)]
    for i in range(n_train):
        x_train[i] = x_train_files['arr_{}'.format(i)]
        y_train[i] = y_train_files['arr_{}'.format(i)]
        edge_indices[i] = edge_index_files['arr_{}'.format(i)]

    cutoff = int(n_train * 0.8)
    random.Random(4).shuffle(x_train)
    random.Random(4).shuffle(y_train)
    random.Random(4).shuffle(edge_indices)
    x_train, x_val = x_train[:cutoff], x_train[cutoff:]
    y_train, y_val = y_train[:cutoff], y_train[cutoff:]
    edge_train, edge_val = edge_indices[:cutoff], edge_indices[cutoff:]

    train_set = [Data() for _ in range(len(x_train))]
    for i in range(len(x_train)):
        train_set[i] = Data(tensor(x_train[i]), tensor(edge_train[i], dtype=torch.long), y=tensor(y_train[i])).to(
            device)

    val_set = [Data() for _ in range(len(x_val))]
    for i in range(len(y_val)):
        val_set[i] = Data(tensor(x_val[i]), tensor(edge_val[i], dtype=torch.long), y=tensor(y_val[i])).to(device)

    model = GCN(n_features, n_outputs, hidden_channels)
    model = model.double()
    model = model.to(device)
    trained_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss().to(device)

    min_val_loss = np.inf
    for epoch in range(1, 10000):
        if epoch == 5000:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_loss, val_loss, min_val_loss, trained_model = k_fold_train(train_set, 4, batch_size, min_val_loss, model,
                                                                         trained_model, optimizer, criterion)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.6f}, '
              f'Validation loss: {val_loss:.6f}, Min validation loss: {min_val_loss:.6f}')

    model = copy.deepcopy(trained_model)
    model.eval()
    for i in range(5):
        index = np.random.choice(np.arange(len(val_set)))
        print(model(val_set[index].x, val_set[index].edge_index))
        print(val_set[index].y, end='\n\n')
    torch.save(model.state_dict(), 'collision_gnn_0.pt')


if __name__ == '__main__':
    main()