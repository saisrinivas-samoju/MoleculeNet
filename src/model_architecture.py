import torch
from torch.nn import Module
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch.nn import BatchNorm1d, Dropout, Linear

class MoleculeNetRegressor(Module):
    def __init__(self, num_features, hidden_dim=64, layer_type='gcn', dropout_rate=0.2):
        super(MoleculeNetRegressor, self).__init__()
        torch.manual_seed(42)
        
        # Layer type selection
        if layer_type == 'gcn':
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        elif layer_type == 'gat':
            self.conv1 = GATConv(num_features, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
            self.conv3 = GATConv(hidden_dim, hidden_dim)
        elif layer_type == 'sage':
            self.conv1 = SAGEConv(num_features, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
        
        # Output layer
        self.out = Linear(hidden_dim * 2, 1)
    
    def forward(self):
        pass
    
class MoleculeNetClassifier(Module):
    def __init__(self, num_features, hidden_dim=64, layer_type='gcn', dropout_rate=0.2, num_classes=2):
        super(MoleculeNetClassifier, self).__init__()
        torch.manual_seed(42)
        
        # Layer type selection
        if layer_type == 'gcn':
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        elif layer_type == 'gat':
            self.conv1 = GATConv(num_features, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
            self.conv3 = GATConv(hidden_dim, hidden_dim)
        elif layer_type == 'sage':
            self.conv1 = SAGEConv(num_features, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
        
        # Output layer
        self.out = Linear(hidden_dim * 2, num_classes)
        self.num_classes = num_classes
    
    def forward(self):
        pass