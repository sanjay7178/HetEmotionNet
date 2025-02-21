import torch
from model.prototype import Net
from torch_geometric.data import Data

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sample model configuration
model_config = {
    'num_node': 32,
    'num_edge': 3,
    'final_out_node': 16,
    'dropout': 0.2,
    'sample_feature_num': 128
}

# Create sample input tensors
batch_size = 2

# Frequency-domain features (batch_size, num_nodes, 4)
x_frequency = torch.randn(batch_size, 32, 4)

# Time-domain features (batch_size, num_nodes, 128)
x_time = torch.randn(batch_size, 32, 128)

# Adjacency matrices (batch_size, num_nodes, num_nodes, num_edge_types)
adj_matrices = torch.randn(batch_size, 32, 32, 3)

# Create a sample batch
data = Data(
    FS=x_frequency.to(device),
    TS=x_time.to(device),
    A=adj_matrices.to(device)
)

def test_model():
    # Initialize model
    model = Net(model_config).to(device)
    model.eval()

    print("Input shapes:")
    print(f"Frequency domain features: {data.FS.shape}")
    print(f"Time domain features: {data.TS.shape}")
    print(f"Adjacency matrices: {data.A.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(data)
        print("\nOutput shape:", output.shape)

    # Expected output shape: (batch_size, 2) for binary classification
    assert output.shape == (batch_size, 2), f"Expected output shape (2, 2), got {output.shape}"
    print("\nModel test passed successfully!")

if __name__ == "__main__":
    test_model()
