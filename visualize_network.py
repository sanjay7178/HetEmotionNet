from graphviz import Digraph
import os

def create_network_visualization():
    # Create a new directed graph
    dot = Digraph(comment='HetEmotionNet Architecture')
    dot.attr(rankdir='TB')

    # Add nodes for input
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input Streams')
        c.node('freq_input', 'Frequency Domain\nFeatures')
        c.node('time_input', 'Time Domain\nFeatures')
        c.node('adj_matrix', 'Adjacency\nMatrices')

    # Add GTN blocks
    dot.node('gtn_f', 'GTN\n(Frequency)')
    dot.node('gtn_t', 'GTN\n(Time)')

    # Add STDCN blocks
    dot.node('stdcn_f', 'STDCN-GRU\n(Frequency)')
    dot.node('stdcn_t', 'STDCN-GRU\n(Time)')

    # Add fusion and classification layers
    dot.node('fusion', 'Feature Fusion')
    dot.node('fc1', 'FC Layer (64)')
    dot.node('output', 'Classification\nOutput (2)')

    # Add edges
    dot.edge('freq_input', 'gtn_f')
    dot.edge('time_input', 'gtn_t')
    dot.edge('adj_matrix', 'gtn_f')
    dot.edge('adj_matrix', 'gtn_t')
    
    dot.edge('gtn_f', 'stdcn_f')
    dot.edge('gtn_t', 'stdcn_t')
    
    dot.edge('freq_input', 'stdcn_f')
    dot.edge('time_input', 'stdcn_t')
    
    dot.edge('stdcn_f', 'fusion')
    dot.edge('stdcn_t', 'fusion')
    
    dot.edge('fusion', 'fc1')
    dot.edge('fc1', 'output')

    # Save the visualization
    dot.render('model_architecture', format='png', cleanup=True)
    print("Network visualization saved as 'model_architecture.png'")

if __name__ == "__main__":
    create_network_visualization()
