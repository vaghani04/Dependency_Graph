import json
import networkx as nx
import matplotlib.pyplot as plt
import os.path

try:
    from networkx.drawing.nx_agraph import graphviz_layout
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# ----------------------------------------------------
# Updated color mappings to match the figure's style
# ----------------------------------------------------
NODE_TYPE_COLORS = {
    "file": "#ffd966",      # light yellow
    "class": "#9ecae1",     # light blue
    "function": "#a1d99b",  # light green
    "variable": "#fdae6b",  # orange-ish
    # Fallback if type not in dictionary
}

# These relationship names come from the figure:
# "call", "classin", "methodin", "var", "instantiate", "parentof".
RELATIONSHIP_COLORS = {
    "call": "#9467bd",          # purple
    "classin": "#2ca02c",       # green
    "methodin": "#ffa500",      # orange
    "var": "#ff7f0e",           # orange-ish
    "instantiate": "#d62728",   # red
    "parentof": "#1f77b4",      # blue
    # If your JSON has old names like "calls" or "contains",
    # you can add them here or rename them in your JSON.
}

def visualize_code_graph(json_path, output_path=None):
    """
    Load a code knowledge graph from the specified JSON file and visualize it
    in a style similar to the provided figure.
    The JSON should have the format:
    {
      "nodes": [
        { "id": <int>, "name": <str>, "type": <str>, "file": <str>, "line_no": <int or None> },
        ...
      ],
      "edges": [
        { "source": <int>, "target": <int>, "relationship": <str>, "line_no": <int or None> },
        ...
      ]
    }
    
    Args:
        json_path (str): Path to the JSON file containing the graph data
        output_path (str, optional): Path where the visualization image will be saved
    """
    # 1. Check if file exists
    if not os.path.isfile(json_path):
        print(f"Error: File '{json_path}' not found.")
        return
    
    # 2. Check if file has .json extension
    if not json_path.lower().endswith('.json'):
        print(f"Warning: File '{json_path}' doesn't have a .json extension.")
    
    # 3. Load JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file '{json_path}'")
        print(f"JSON Error: {str(e)}")
        return
    except Exception as e:
        print(f"Error: Failed to read file '{json_path}': {str(e)}")
        return
    
    # 4. Create a directed graph
    G = nx.DiGraph()
    
    # 5. Add nodes
    for node in data.get("nodes", []):
        node_id = node["id"]
        node_type = node.get("type", "unknown").lower()  # 'file', 'class', 'function', etc.
        node_name = node.get("name", f"Node{node_id}")
        
        G.add_node(
            node_id,
            label=node_name,
            ntype=node_type,
            file=node.get("file"),
            line_no=node.get("line_no")
        )
    
    # 6. Add edges
    for edge in data.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        relationship = edge.get("relationship", "unknown").lower()
        line_no = edge.get("line_no")
        
        G.add_edge(
            source,
            target,
            relationship=relationship,
            line_no=line_no
        )
    
    # Check if graph is empty
    if len(G.nodes()) == 0:
        print("Warning: Graph has no nodes. Check if your JSON file has the right structure.")
        return
    
    # 7. Assign colors to nodes
    node_colors = []
    for n in G.nodes():
        ntype = G.nodes[n].get("ntype", "unknown")
        color = NODE_TYPE_COLORS.get(ntype, "#cccccc")  # default gray
        node_colors.append(color)
    
    # 8. Assign colors to edges
    edge_colors = []
    for (u, v) in G.edges():
        rel = G[u][v].get("relationship", "unknown")
        color = RELATIONSHIP_COLORS.get(rel, "#999999")  # default gray
        edge_colors.append(color)
    
    # 9. Choose a layout - try to use spring_layout directly without scipy
    try:
        # First try spring_layout with custom parameters for better distribution
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    except ImportError as e:
        if 'scipy' in str(e):
            print("Warning: scipy package is missing for optimal layout.")
            print("Installing scipy is recommended: pip install scipy")
            # Fallback to shell layout which doesn't require scipy
            pos = nx.shell_layout(G)
        else:
            # For other import errors, just use a circular layout
            print(f"Layout error: {str(e)}")
            pos = nx.circular_layout(G)
    except Exception as e:
        print(f"Layout calculation failed: {str(e)}")
        print("Using circular layout as fallback.")
        pos = nx.circular_layout(G)
    
    # 10. Draw the graph
    plt.figure(figsize=(12, 10))
    
    # Draw edges (behind nodes)
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        edge_color=edge_colors,
        arrowstyle="-|>",
        arrowsize=15
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=1200,
        edgecolors="#333333"  # dark border
    )
    
    # Draw node labels
    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels,
        font_size=10,
        font_color="black"
    )
    
    # 11. Draw edge labels: "relationship:line_no" if line_no is present
    edge_labels = {}
    for (u, v) in G.edges():
        rel = G[u][v].get("relationship", "unknown")
        line_no = G[u][v].get("line_no")
        if line_no is not None:
            edge_labels[(u, v)] = f"{rel}:{line_no}"
        else:
            edge_labels[(u, v)] = rel
    
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=9,
        font_color="black",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
        label_pos=0.5
    )
    
    # Check if the data contains statistics and display them
    if "statistics" in data:
        stats = data["statistics"]
        stat_text = f"Graph Statistics:\n"
        stat_text += f"Total Nodes: {stats.get('total_nodes', 'N/A')}\n"
        stat_text += f"Total Edges: {stats.get('total_edges', 'N/A')}\n"
        
        # Node types breakdown
        if "node_types" in stats:
            node_types = stats["node_types"]
            stat_text += f"Node Types: "
            stat_text += ", ".join([f"{k}: {v}" for k, v in node_types.items() if v > 0])
            stat_text += "\n"
        
        # Add more stats as needed
        plt.figtext(0.02, 0.02, stat_text, fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    plt.title("Code Knowledge Graph Visualization")
    plt.axis("off")
    plt.tight_layout()
    
    # Save the image if output path is provided
    if output_path:
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    # Show plot
    plt.show()

def print_statistics(json_path):
    """
    Print the statistics from a knowledge graph JSON file.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "statistics" in data:
            stats = data["statistics"]
            print("\n======= KNOWLEDGE GRAPH STATISTICS =======")
            print(f"File: {os.path.basename(json_path)}")
            print(f"Total Nodes: {stats.get('total_nodes', 'N/A')}")
            print(f"Total Edges: {stats.get('total_edges', 'N/A')}")
            
            # Node types breakdown
            if "node_types" in stats:
                print("\nNode Types:")
                for node_type, count in stats["node_types"].items():
                    print(f"  - {node_type}: {count}")
            
            # Relationship types
            if "relationship_types" in stats:
                print("\nRelationship Types:")
                for rel_type, count in stats["relationship_types"].items():
                    print(f"  - {rel_type}: {count}")
            
            # Most connected nodes
            if "most_connected_nodes" in stats and stats["most_connected_nodes"]:
                print("\nMost Connected Nodes:")
                for node in stats["most_connected_nodes"]:
                    print(f"  - {node['name']} ({node['type']}): {node['total_connections']} connections "
                          f"(in: {node['in_degree']}, out: {node['out_degree']})")
            
            # Additional statistics
            if "nodes_density" in stats:
                print(f"\nNode Density: {stats['nodes_density']:.2f} edges per node")
            if "average_connections_per_node" in stats:
                print(f"Average Connections: {stats['average_connections_per_node']:.2f} per node")
            
            print("=========================================\n")
        else:
            # If statistics not in the file, generate basic stats from nodes and edges
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            
            node_types = {}
            for node in nodes:
                node_type = node.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            relationship_types = {}
            for edge in edges:
                rel_type = edge.get("relationship", "unknown")
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            print("\n======= BASIC KNOWLEDGE GRAPH STATISTICS =======")
            print(f"File: {os.path.basename(json_path)}")
            print(f"Total Nodes: {len(nodes)}")
            print(f"Total Edges: {len(edges)}")
            
            print("\nNode Types:")
            for node_type, count in node_types.items():
                print(f"  - {node_type}: {count}")
            
            print("\nRelationship Types:")
            for rel_type, count in relationship_types.items():
                print(f"  - {rel_type}: {count}")
            
            print("=================================================\n")
    except Exception as e:
        print(f"Error reading statistics from {json_path}: {str(e)}")

# Default usage
if __name__ == "__main__":
    # Using default files
    default_files = "outputs/Testing_folder.json"
    # default_files = "RAG_Playground.json"
    json_file_path = None

    if os.path.isfile(default_files):
        print(f"Using default file: {default_files}")
        json_file_path = default_files
    
    if not json_file_path:
        print("No default files found.")
    else:
        # Determine output path
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_path = os.path.join("outputs", f"{base_name}_graph.png")
        
        # Generate the visualization
        visualize_code_graph(json_file_path, output_path)
        
        # Print the statistics
        print_statistics(json_file_path)
