"""
GCN Model Architecture for Toxicity Prediction
This module implements the Graph Convolutional Network (GCN) model 
consisting of an encoder and classifier for molecular toxicity prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network Encoder
    
    Extracts and updates node representations through several graph 
    convolutional layers with dropout for regularization.
    
    The last layer merges all node features into a tensor using 
    max-pooling and weighted sum operations.
    """
    
    def __init__(self, 
                 in_feats, 
                 hidden_feats, 
                 num_layers=3, 
                 dropout=0.2,
                 activation=F.relu):
        """
        Initialize the GCN Encoder
        
        Args:
            in_feats (int): Input feature dimension
            hidden_feats (list): List of hidden feature dimensions for each layer
            num_layers (int): Number of graph convolutional layers
            dropout (float): Dropout rate after each graph conv layer
            activation: Activation function (default: ReLU)
        """
        super(GCNEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        
        # Build graph convolutional layers
        self.conv_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(GraphConv(in_feats, hidden_feats[0]))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(1, num_layers):
            in_dim = hidden_feats[i-1]
            out_dim = hidden_feats[i] if i < len(hidden_feats) else hidden_feats[-1]
            self.conv_layers.append(GraphConv(in_dim, out_dim))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Pooling layers for graph-level representation
        self.max_pool = MaxPooling()
        self.avg_pool = AvgPooling()
        
        # Weighted sum parameters for combining pooling operations
        self.pool_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(self, g, features):
        """
        Forward pass through the encoder
        
        Args:
            g (DGLGraph): Input molecular graph
            features (Tensor): Node features
            
        Returns:
            Tensor: Graph-level representation
        """
        h = features
        
        # Apply graph convolutional layers with dropout
        for i in range(self.num_layers):
            h = self.conv_layers[i](g, h)
            h = self.activation(h)
            h = self.dropout_layers[i](h)
        
        # Graph-level pooling: combine max-pooling and weighted sum
        g.ndata['h'] = h
        
        # Max pooling
        max_pooled = self.max_pool(g, h)
        
        # Average pooling (weighted sum)
        avg_pooled = self.avg_pool(g, h)
        
        # Combine pooling operations with learnable weights
        weights = F.softmax(self.pool_weight, dim=0)
        graph_representation = weights[0] * max_pooled + weights[1] * avg_pooled
        
        return graph_representation


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron Classifier
    
    Two-layer perceptron with dropout and batch normalization
    for final toxicity prediction.
    """
    
    def __init__(self, 
                 in_feats, 
                 hidden_feats, 
                 num_classes=2, 
                 dropout=0.2):
        """
        Initialize the MLP Classifier
        
        Args:
            in_feats (int): Input feature dimension
            hidden_feats (int): Hidden layer dimension
            num_classes (int): Number of output classes (default: 2 for binary)
            dropout (float): Dropout rate
        """
        super(MLPClassifier, self).__init__()
        
        # First layer
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_feats, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the classifier
        
        Args:
            x (Tensor): Input features
            
        Returns:
            Tensor: Class predictions (logits)
        """
        # First layer with batch norm and dropout
        x = self.fc1(x)
        # Only apply batch norm during training and with batch size > 1
        if self.training and x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x


class GCNModel(nn.Module):
    """
    Complete GCN Model for Toxicity Prediction
    
    Combines the GCN Encoder and MLP Classifier for end-to-end
    molecular toxicity prediction.
    """
    
    def __init__(self, 
                 in_feats,
                 hidden_feats,
                 num_layers=3,
                 classifier_hidden=128,
                 num_classes=2,
                 dropout=0.2):
        """
        Initialize the complete GCN model
        
        Args:
            in_feats (int): Input node feature dimension
            hidden_feats (list): List of hidden dimensions for encoder
            num_layers (int): Number of graph conv layers
            classifier_hidden (int): Hidden dimension for classifier
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
        """
        super(GCNModel, self).__init__()
        
        # Encoder
        self.encoder = GCNEncoder(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Classifier
        # Input to classifier is the output dimension of encoder
        encoder_output_dim = hidden_feats[-1]
        self.classifier = MLPClassifier(
            in_feats=encoder_output_dim,
            hidden_feats=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout
        )
        
    def forward(self, g, features):
        """
        Forward pass through the complete model
        
        Args:
            g (DGLGraph): Input molecular graph
            features (Tensor): Node features
            
        Returns:
            Tensor: Class predictions (logits)
        """
        # Encode graph to graph-level representation
        graph_representation = self.encoder(g, features)
        
        # Classify
        predictions = self.classifier(graph_representation)
        
        return predictions
    
    def get_embeddings(self, g, features):
        """
        Get graph-level embeddings without classification
        
        Args:
            g (DGLGraph): Input molecular graph
            features (Tensor): Node features
            
        Returns:
            Tensor: Graph-level embeddings
        """
        return self.encoder(g, features)


def create_gcn_model(in_feats, 
                     hidden_dims=[64, 128, 256],
                     num_layers=3,
                     classifier_hidden=128,
                     num_classes=2,
                     dropout=0.2):
    """
    Factory function to create a GCN model with default hyperparameters
    
    Args:
        in_feats (int): Input node feature dimension
        hidden_dims (list): Hidden dimensions for encoder layers
        num_layers (int): Number of graph conv layers
        classifier_hidden (int): Hidden dimension for classifier
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
        
    Returns:
        GCNModel: Initialized GCN model
    """
    return GCNModel(
        in_feats=in_feats,
        hidden_feats=hidden_dims,
        num_layers=num_layers,
        classifier_hidden=classifier_hidden,
        num_classes=num_classes,
        dropout=dropout
    )


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 70)
    print("GCN Model Architecture Test")
    print("=" * 70)
    
    # Create a sample graph
    import dgl
    import torch
    
    # Create a simple graph: 5 nodes, 8 edges
    edges = ([0, 0, 1, 1, 2, 2, 3, 4],
             [1, 2, 2, 3, 3, 4, 4, 0])
    g = dgl.graph(edges)
    
    # Node features (5 nodes, 74 features - typical for molecular graphs)
    in_feats = 74
    node_features = torch.randn(5, in_feats)
    
    # Create model
    model = create_gcn_model(
        in_feats=in_feats,
        hidden_dims=[64, 128, 256],
        num_layers=3,
        classifier_hidden=128,
        num_classes=2,
        dropout=0.2
    )
    
    print(f"\nModel created successfully!")
    print(f"Input features: {in_feats}")
    print(f"Output classes: 2 (toxic/non-toxic)")
    
    # Forward pass
    with torch.no_grad():
        predictions = model(g, node_features)
        embeddings = model.get_embeddings(g, node_features)
    
    print(f"\nOutput predictions shape: {predictions.shape}")
    print(f"Graph embeddings shape: {embeddings.shape}")
    
    # Print model summary
    print("\n" + "=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nEncoder:")
    print(model.encoder)
    print("\nClassifier:")
    print(model.classifier)
    print("=" * 70)
