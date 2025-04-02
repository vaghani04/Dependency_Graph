import os
import json
import argparse
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

class GraphQuery:
    def __init__(self, graph_file):
        """
        Initialize the graph query tool with the path to a knowledge graph JSON file.
        
        Args:
            graph_file (str): Path to the knowledge graph JSON file
        """
        self.graph_file = graph_file
        self.nodes = []
        self.edges = []
        self.node_index = {}  # Maps node IDs to their positions in the nodes list
        
        # Load the graph
        self.load_graph()
    
    def load_graph(self):
        """Load the knowledge graph from the JSON file."""
        try:
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.nodes = data.get('nodes', [])
            self.edges = data.get('edges', [])
            
            # Build node index for faster lookups
            self.node_index = {node['id']: i for i, node in enumerate(self.nodes)}
            
            print(f"Loaded knowledge graph with {len(self.nodes)} nodes and {len(self.edges)} edges.")
        except Exception as e:
            print(f"Error loading graph file: {e}")
    
    def find_function(self, function_name, file_path=None):
        """
        Find a function in the knowledge graph by name and optional file path.
        
        Args:
            function_name (str): Name of the function to find
            file_path (str, optional): Path to the file containing the function
            
        Returns:
            dict: Information about the function found, or None if not found
        """
        # Step 1: Try to find exact match with file path
        if file_path:
            for node in self.nodes:
                if (node.get('type') == 'function' and 
                    node.get('name') == function_name and 
                    node.get('file') == file_path):
                    return node
        
        # Step 2: Try to find by name only
        candidates = []
        for node in self.nodes:
            if node.get('type') == 'function' and node.get('name') == function_name:
                candidates.append(node)
        
        if candidates:
            if len(candidates) > 1:
                print(f"Found {len(candidates)} functions named '{function_name}'")
                if file_path:
                    # Try partial path match
                    for node in candidates:
                        if node.get('file') and file_path in node.get('file'):
                            return node
                
                # If still ambiguous, return the first one but warn
                print(f"Returning the first match. Specify file_path for more precision.")
            return candidates[0]
        
        return None
    
    def get_function_definition(self, function_node):
        """
        Get detailed information about where a function is defined.
        
        Args:
            function_node (dict): The function node from the knowledge graph
            
        Returns:
            list: List of definition locations with file path and line number
        """
        if not function_node:
            return []
        
        definitions = []
        
        # Check if we have locations field (better)
        if 'locations' in function_node:
            for loc in function_node['locations']:
                definitions.append({
                    'file': loc.get('file', 'unknown'),
                    'line_no': loc.get('line_no', 'unknown')
                })
        else:
            # Fallback to basic information
            definitions.append({
                'file': function_node.get('file', 'unknown'),
                'line_no': function_node.get('line_no', 'unknown')
            })
        
        return definitions
    
    def get_function_callers(self, function_id):
        """
        Find all functions that call the specified function.
        
        Args:
            function_id (int): ID of the function node
            
        Returns:
            list: List of caller information (caller node and call site)
        """
        callers = []
        
        for edge in self.edges:
            if edge.get('relationship') == 'call' and edge.get('target') == function_id:
                caller_id = edge.get('source')
                if caller_id in self.node_index:
                    caller_node = self.nodes[self.node_index[caller_id]]
                    
                    caller_info = {
                        'caller': caller_node,
                        'line_no': edge.get('line_no', 'unknown')
                    }
                    callers.append(caller_info)
        
        return callers
    
    def get_recursive_callers(self, function_id, max_depth=10):
        """
        Find all functions that call the specified function, recursively up to max_depth.
        
        Args:
            function_id (int): ID of the function node
            max_depth (int): Maximum recursion depth
            
        Returns:
            dict: Nested dictionary representing the call tree
        """
        if max_depth <= 0:
            return {"id": function_id, "truncated": True}
        
        # Get direct callers
        direct_callers = self.get_function_callers(function_id)
        
        result = {
            "id": function_id,
            "callers": []
        }
        
        # Add node information
        if function_id in self.node_index:
            function_node = self.nodes[self.node_index[function_id]]
            result["name"] = function_node.get("name", "unknown")
            result["file"] = function_node.get("file", "unknown")
            result["type"] = function_node.get("type", "unknown")
        
        # Process each caller recursively
        seen_callers = set([function_id])  # Avoid cycles
        
        for caller_info in direct_callers:
            caller_id = caller_info['caller'].get('id')
            if caller_id not in seen_callers:
                seen_callers.add(caller_id)
                caller_tree = self.get_recursive_callers(caller_id, max_depth - 1)
                caller_tree["line_no"] = caller_info.get("line_no", "unknown")
                result["callers"].append(caller_tree)
        
        return result
    
    def trace_function_usage(self, function_name, file_path=None, max_depth=5):
        """
        Trace how a function is used throughout the codebase.
        
        Args:
            function_name (str): Name of the function to trace
            file_path (str, optional): Path to the file containing the function
            max_depth (int): Maximum recursion depth for caller tracing
            
        Returns:
            dict: Complete information about function usage
        """
        # Find the function
        function_node = self.find_function(function_name, file_path)
        
        if not function_node:
            return {
                "error": f"Function '{function_name}' not found in the knowledge graph"
            }
        
        # Get function ID
        function_id = function_node.get('id')
        
        # Get definition information
        definitions = self.get_function_definition(function_node)
        
        # Get direct callers
        direct_callers = self.get_function_callers(function_id)
        
        callers_info = []
        for caller in direct_callers:
            caller_node = caller.get('caller')
            callers_info.append({
                "name": caller_node.get('name', 'unknown'),
                "file": caller_node.get('file', 'unknown'),
                "line_no": caller.get('line_no', 'unknown'),
                "type": caller_node.get('type', 'unknown'),
                "id": caller_node.get('id')
            })
        
        # Get recursive call tree
        call_tree = self.get_recursive_callers(function_id, max_depth)
        
        # Prepare result
        result = {
            "function": {
                "name": function_node.get('name'),
                "qualified_name": function_node.get('qualified_name', function_node.get('name')),
                "fully_qualified_name": function_node.get('fully_qualified_name', ''),
                "signature": function_node.get('signature', ''),
                "id": function_id,
                "docstring": function_node.get('docstring', '')
            },
            "defined_in": definitions,
            "direct_callers": callers_info,
            "call_tree": call_tree,
            "total_direct_callers": len(callers_info)
        }
        
        return result
    
    def visualize_call_tree(self, call_tree, output_file=None):
        """
        Visualize a call tree as a directed graph.
        
        Args:
            call_tree (dict): Call tree from get_recursive_callers
            output_file (str, optional): Path to save the visualization
        """
        G = nx.DiGraph()
        
        # Helper function to recursively add nodes and edges
        def add_to_graph(node):
            node_id = node.get("id")
            node_name = node.get("name", f"Node-{node_id}")
            
            # Add node with attributes
            G.add_node(node_id, name=node_name, file=node.get("file", ""), 
                      type=node.get("type", ""), label=node_name)
            
            # Add edges to all callers
            for caller in node.get("callers", []):
                caller_id = caller.get("id")
                G.add_node(caller_id, name=caller.get("name", f"Node-{caller_id}"),
                          file=caller.get("file", ""), type=caller.get("type", ""),
                          label=caller.get("name", f"Node-{caller_id}"))
                
                # Add edge from caller to node
                G.add_edge(caller_id, node_id, line_no=caller.get("line_no", ""))
                
                # Recurse
                add_to_graph(caller)
        
        # Build graph
        add_to_graph(call_tree)
        
        # Visualize
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes()}
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, arrows=True)
        
        # Title
        root_name = call_tree.get("name", f"Function-{call_tree.get('id')}")
        plt.title(f"Call Tree for '{root_name}'", size=15)
        
        # Save or show
        if output_file:
            plt.savefig(output_file, format="png", bbox_inches="tight")
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()
        
        return G

    def print_function_usage_report(self, function_name, file_path=None, max_depth=5):
        """
        Print a human-readable report of a function's usage in the codebase.
        
        Args:
            function_name (str): Name of the function to analyze
            file_path (str, optional): Path to the file containing the function
            max_depth (int): Maximum depth for caller tracing
        """
        result = self.trace_function_usage(function_name, file_path, max_depth)
        
        if "error" in result:
            print(result["error"])
            return
        
        function = result["function"]
        print(f"\n{'='*80}")
        print(f"FUNCTION USAGE REPORT: {function['name']}")
        print(f"{'='*80}")
        
        # Print qualified names if available
        if function.get("qualified_name") and function["qualified_name"] != function["name"]:
            print(f"Qualified name: {function['qualified_name']}")
        if function.get("fully_qualified_name"):
            print(f"Fully qualified name: {function['fully_qualified_name']}")
        
        # Print signature if available
        if function.get("signature"):
            print(f"\nSignature: {function['signature']}")
        
        # Print docstring if available
        if function.get("docstring"):
            print(f"\nDocumentation:\n{function['docstring']}")
        
        # Print definition locations
        print(f"\nDEFINED IN ({len(result['defined_in'])} locations):")
        for loc in result["defined_in"]:
            print(f"  - {loc['file']}:{loc['line_no']}")
        
        # Print direct callers
        callers = result["direct_callers"]
        print(f"\nDIRECT CALLERS ({len(callers)}):")
        
        if callers:
            # Group callers by file for easier reading
            callers_by_file = defaultdict(list)
            for caller in callers:
                callers_by_file[caller["file"]].append(caller)
            
            for file, file_callers in callers_by_file.items():
                print(f"\n  In {file}:")
                for caller in file_callers:
                    print(f"    - Line {caller['line_no']}: {caller['name']} ({caller['type']})")
        else:
            print("  No direct callers found in the codebase.")
        
        print(f"\n{'='*80}\n")
        
        # Visualize if available
        return result["call_tree"]


def main():

    # graph_file = "Testing_folder.json"
    graph_file = "RAG_Playground.json"

    function_name = "get_password_hash"
    # function_name = "submit"
    # function_name = "add_user"
    # function_name = "Timeout"
    # function_name = "_make_anthropic_request"

    file = "/Users/maunikvaghani/Developer/Hackathons/H1/Testing_folder/utils/security.py"
    # file = "/Users/maunikvaghani/Developer/Hackathons/H1/Testing_folder/services/auth_service.py"
    # file = "/Users/maunikvaghani/Developer/Hackathons/H1/Testing_folder/controllers/auth_controller.py"
    # file = "/Users/maunikvaghani/Developer/Hackathons/H1/Testing_folder/usecases/auth/submit_usecase_temp.py"
    # file = "RAG_Playground/app/utils/llm_utils.py"
    # file = "RAG_Playground/app/utils/llm_utils.py"

    depth = 5
    visualize = True
    output_to_store = f"{function_name}_call_tree.png"

    # Initialize and run query
    query = GraphQuery(graph_file)
    call_tree = query.print_function_usage_report(function_name, file, depth)
    
    # Visualize if requested
    if visualize and call_tree:
        output_file = output_to_store
        query.visualize_call_tree(call_tree, output_file)

if __name__ == "__main__":
    main() 