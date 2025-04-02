import os
import ast
import json
import igraph as ig
import numpy as np
from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional, Any
import asyncio
import aiofiles
import matplotlib.pyplot as plt

class MergedDependencyAnalyzer(ast.NodeVisitor):
    def __init__(self):
        # Core graph data structures
        self.nodes = []  # List of dicts with node information
        self.edges = []  # List of dicts with edge information
        self.node_counter = 0
        self.node_index = {}  # (file, name, type) -> node_id mapping
        
        # Context stacks for hierarchical analysis
        self.file_stack = []
        self.class_stack = []
        self.function_stack = []
        
        # Additional metadata for enhanced analysis
        self.imports = defaultdict(set)  # file -> imported modules
        self.function_calls = defaultdict(set)  # function -> called functions
        self.class_inheritance = defaultdict(set)  # class -> parent classes
        self.method_map = defaultdict(set)  # class -> methods
        
        # Risk scoring parameters
        self.edge_weights = {
            "import": 0.7,
            "inheritance": 0.9,
            "call": 0.8,
            "methodin": 0.6,
            "classin": 0.5,
            "attribute": 0.4,
            "variable": 0.5,
        }

    def _get_or_create_node(self, name: str, node_type: str, file_path: str, line_no: Optional[int] = None, 
                          complexity: float = 1.0, level: int = 0) -> int:
        """Create or retrieve a node with enhanced metadata."""
        key = (file_path, name, node_type)
        if key in self.node_index:
            return self.node_index[key]
        
        self.node_counter += 1
        node_id = self.node_counter
        
        node_info = {
            "id": node_id,
            "name": name,
            "type": node_type,
            "file": file_path,
            "line_no": line_no,
            "complexity": complexity,
            "level": level,
            "cluster": -1  # Will be set during semantic clustering
        }
        self.nodes.append(node_info)
        self.node_index[key] = node_id
        return node_id
    
    def _resolve_function_or_class(self, name: str, file_path: str, node_type: str) -> Optional[int]:
        """
        Resolve a function or class by its name and file path across the codebase.
        """
        key = (name, file_path)
        return self.node_index.get(key)

    def _add_edge(self, source_id: int, target_id: int, relationship: str, 
                 line_no: Optional[int] = None, weight: Optional[float] = None):
        """Add an edge with enhanced metadata and risk scoring."""
        if weight is None:
            weight = self.edge_weights.get(relationship, 0.5)
        
        # Calculate risk score based on source and target complexity
        source_node = next(n for n in self.nodes if n["id"] == source_id)
        target_node = next(n for n in self.nodes if n["id"] == target_id)
        risk_score = min(1.0, source_node["complexity"] * target_node["complexity"] * weight)
        
        edge_info = {
            "source": source_id,
            "target": target_id,
            "relationship": relationship,
            "line_no": line_no,
            "weight": weight,
            "risk_score": risk_score
        }
        self.edges.append(edge_info)

    def _calculate_node_complexity(self, node_type: str, name: str, file_path: str) -> float:
        """Calculate complexity score for a node based on its relationships."""
        complexity = 1.0  # Base complexity
        
        if node_type == "class":
            # Classes with more methods or inheritance are more complex
            num_methods = len(self.method_map.get(name, set()))
            num_parents = len(self.class_inheritance.get(name, set()))
            complexity += 0.2 * num_methods + 0.3 * num_parents
        elif node_type == "function":
            # Functions that call more other functions are more complex
            num_calls = len(self.function_calls.get(name, set()))
            complexity += 0.2 * num_calls
        
        return complexity

    def visit_Import(self, node):
        """Handle import statements with line numbers."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self.imports[self._current_file_path()].add(module_name)
            
            # Create import edge from file to imported module
            if self.file_stack:
                current_file_id = self.file_stack[-1]
                imported_node_id = self._get_or_create_node(
                    name=module_name,
                    node_type="module",
                    file_path=f"external:{module_name}",
                    line_no=node.lineno
                )
                self._add_edge(current_file_id, imported_node_id, "import", line_no=node.lineno)
        
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle from-import statements with line numbers."""
        if node.module:
            module_name = node.module.split('.')[0]
            self.imports[self._current_file_path()].add(module_name)
            
            # Create import edge from file to imported module
            if self.file_stack:
                current_file_id = self.file_stack[-1]
                imported_node_id = self._get_or_create_node(
                    name=module_name,
                    node_type="module",
                    file_path=f"external:{module_name}",
                    line_no=node.lineno
                )
                self._add_edge(current_file_id, imported_node_id, "import", line_no=node.lineno)
        
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Enhanced class definition handling with inheritance and line numbers."""
        current_file_id = self.file_stack[-1] if self.file_stack else None
        
        # Create class node with complexity calculation
        complexity = self._calculate_node_complexity("class", node.name, self._current_file_path())
        class_node_id = self._get_or_create_node(
            name=node.name,
            node_type="class",
            file_path=self._current_file_path(),
            line_no=node.lineno,
            complexity=complexity,
            level=1
        )
        
        # Add inheritance relationships
        for base in node.bases:
            if isinstance(base, ast.Name):
                parent_name = base.id
                self.class_inheritance[node.name].add(parent_name)
                
                # Create parent class node and inheritance edge
                parent_node_id = self._get_or_create_node(
                    name=parent_name,
                    node_type="class",
                    file_path=f"external:{parent_name}",
                    line_no=node.lineno
                )
                self._add_edge(class_node_id, parent_node_id, "inheritance", line_no=node.lineno)
        
        # Connect class to file
        if current_file_id is not None:
            self._add_edge(current_file_id, class_node_id, "classin", line_no=node.lineno)
        
        self.class_stack.append(class_node_id)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        """Enhanced function definition handling with line numbers."""
        # Determine if this is a method or standalone function
        is_method = bool(self.class_stack)
        node_type = "method" if is_method else "function"
        
        # Calculate complexity
        complexity = self._calculate_node_complexity(node_type, node.name, self._current_file_path())
        
        # Create function node
        func_node_id = self._get_or_create_node(
            name=node.name,
            node_type=node_type,
            file_path=self._current_file_path(),
            line_no=node.lineno,
            complexity=complexity,
            level=2
        )
        
        # Connect to parent (class or file)
        if is_method:
            current_class_id = self.class_stack[-1]
            self._add_edge(current_class_id, func_node_id, "methodin", line_no=node.lineno)
            self.method_map[self.nodes[current_class_id-1]["name"]].add(node.name)
        elif self.file_stack:
            current_file_id = self.file_stack[-1]
            self._add_edge(current_file_id, func_node_id, "methodin", line_no=node.lineno)
        
        self.function_stack.append(func_node_id)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Call(self, node):
        """Enhanced function call handling with line numbers."""
        # Determine the name of the called function
        if isinstance(node.func, ast.Name):
            called_func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called_func_name = node.func.attr
        else:
            called_func_name = "<unknown>"

        current_file_path = self._current_file_path()
        called_node_id = self._resolve_function_or_class(called_func_name, current_file_path, "function")
        
        # Create or get the called function node
        if called_node_id is None:
            called_node_id = self._get_or_create_node(
                name=called_func_name,
                node_type="function",
                file_path="unknown (external or attribute call)",
                line_no=node.lineno,
            )
        
        # Record the function call
        if self.function_stack:
            current_func_id = self.function_stack[-1]
            self._add_edge(current_func_id, called_node_id, "call", line_no=node.lineno)
        elif self.file_stack:
            current_file_id = self.file_stack[-1]
            self._add_edge(current_file_id, called_node_id, "call", line_no=node.lineno)
        
        self.generic_visit(node)

    def visit_Assign(self, node):
        """
        Detect variables as dependencies by analyzing assignment statements.
        """
        if isinstance(node.targets[0], ast.Name):
            variable_name = node.targets[0].id
            current_file_id = self.file_stack[-1] if self.file_stack else None

            # Create a node for the variable
            variable_node_id = self._get_or_create_node(
                name=variable_name,
                node_type="variable",
                file_path=self._current_file_path(),
                line_no=node.lineno,
                level=2
            )

            # Connect the variable to the current file or function
            if self.function_stack:
                current_func_id = self.function_stack[-1]
                self._add_edge(current_func_id, variable_node_id, "variable", line_no=node.lineno)
            elif current_file_id:
                self._add_edge(current_file_id, variable_node_id, "variable", line_no=node.lineno)

        self.generic_visit(node)

    def _current_file_path(self) -> Optional[str]:
        """Get the current file path from the file stack."""
        if not self.file_stack:
            return None
        file_node_id = self.file_stack[-1]
        for nd in self.nodes:
            if nd["id"] == file_node_id:
                return nd["file"]
        return None

    async def parse_python_file(self, file_path: str):
        """Parse a single Python file with enhanced error handling."""
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                file_contents = await f.read()
            
            tree = ast.parse(file_contents, filename=file_path)
            
            # Create file node
            file_name = os.path.basename(file_path)
            file_node_id = self._get_or_create_node(
                name=file_name,
                node_type="file",
                file_path=file_path,
                level=0
            )
            
            self.file_stack.append(file_node_id)
            self.visit(tree)
            self.file_stack.pop()
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    async def build_graph_from_directory(self, directory: str):
        """Recursively analyze all Python files in the directory."""
        tasks = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".py"):
                    file_path = os.path.join(root, filename)
                    tasks.append(self.parse_python_file(file_path))
        
        await asyncio.gather(*tasks)

    def _perform_semantic_clustering(self):
        """Group nodes into semantic clusters based on their relationships."""
        # Create an igraph instance for clustering
        g = ig.Graph(directed=True)
        
        # Add vertices
        g.add_vertices(len(self.nodes))
        
        # Add edges
        for edge in self.edges:
            g.add_edge(edge["source"]-1, edge["target"]-1, weight=edge["weight"])
        
        # Perform community detection
        if g.ecount() > 0:
            try:
                # Convert to undirected graph for community detection
                g_undirected = g.as_undirected()
                communities = g_undirected.community_fastgreedy(weights="weight").as_clustering()
                
                # Assign cluster IDs to nodes
                for i, community in enumerate(communities):
                    for vertex_id in community:
                        self.nodes[vertex_id]["cluster"] = i
            except Exception as e:
                print(f"Warning: Community detection failed: {e}")
                # Fallback to type-based clustering
                type_clusters = {}
                for node in self.nodes:
                    if node["type"] not in type_clusters:
                        type_clusters[node["type"]] = len(type_clusters)
                    node["cluster"] = type_clusters[node["type"]]

    def get_recursive_dependencies(self, node_id: int) -> Dict[str, Any]:
        """Get all dependencies recursively up to specified depth."""
        visited = set()
        dependencies = []
        
        def traverse(current_id, current_depth):
            if current_id in visited:
                return
            visited.add(current_id)

            # Get all edges where this node is the source
            for edge in self.edges:
                if edge["source"] == current_id:
                    target_id = edge["target"]

                    # Get target node info
                    target_node = next(n for n in self.nodes if n["id"] == target_id)

                    # Record the dependency
                    dependencies.append({
                        "source": current_id,  # Use ID instead of name
                        "target": target_id,  # Use ID instead of name
                        "relationship": edge["relationship"],
                        "depth": current_depth + 1,
                        "risk_score": edge["risk_score"],
                        "line_no": edge["line_no"]
                    })

                    # Recursively traverse the target node
                    traverse(target_id, current_depth + 1)

        # Start traversal from the given node
        traverse(node_id, 0)

        return {
            "node": node_id,  # Use ID instead of name
            "dependencies": sorted(dependencies, key=lambda x: x["depth"])
        }

    def save_to_json(self, output_path: str):
        """Save the enhanced graph to JSON with recursive dependencies."""
        # Perform semantic clustering
        self._perform_semantic_clustering()
        
        # Calculate recursive dependencies for each node
        recursive_deps = {}
        for node in self.nodes:
            deps = self.get_recursive_dependencies(node["id"])
            recursive_deps[node["id"]] = deps
        
        # Prepare the final JSON structure
        data = {
            "metadata": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "cluster_count": max(n["cluster"] for n in self.nodes) + 1
            },
            "nodes": self.nodes,
            "edges": self.edges,
            "recursive_dependencies": recursive_deps
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def visualize(self, output_file: Optional[str] = None, show: bool = False, 
                 layout: str = "auto", figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize the dependency graph with enhanced styling.
        
        Args:
            output_file: Path to save the visualization (None to skip saving)
            show: Whether to display the graph interactively
            layout: Graph layout algorithm ("auto", "fruchterman_reingold", "drl", "kamada_kawai")
            figsize: Figure size (width, height)
        """
        if not self.nodes:
            print("Graph is empty, nothing to visualize.")
            return

        # Create igraph instance for visualization
        g = ig.Graph(directed=True)
        
        # Add vertices
        g.add_vertices(len(self.nodes))
        
        # Add edges
        for edge in self.edges:
            g.add_edge(edge["source"]-1, edge["target"]-1, 
                      weight=edge["weight"],
                      relationship=edge["relationship"])
        
        # Define visual properties
        visual_style = {}
        
        # Define node colors based on type
        color_map = {
            "file": "lightblue",
            "class": "lightgreen",
            "function": "orange",
            "method": "pink",
            "module": "gray"
        }
        
        # Define node shapes based on type
        shape_map = {
            "file": "square",
            "class": "circle",
            "function": "triangle-up",
            "method": "diamond",
            "module": "hexagon"
        }
        
        # Define edge colors based on relationship type
        edge_color_map = {
            "import": "gray",
            "inheritance": "red",
            "call": "blue",
            "methodin": "green",
            "classin": "purple",
            "attribute": "orange"
        }
        
        # Set node colors, sizes, and labels
        visual_style["vertex_color"] = [color_map.get(node["type"], "white") for node in self.nodes]
        visual_style["vertex_size"] = [20 + 10 * min(5, node["complexity"]) for node in self.nodes]
        visual_style["vertex_shape"] = [shape_map.get(node["type"], "circle") for node in self.nodes]
        visual_style["vertex_label"] = [node["name"] for node in self.nodes]
        visual_style["vertex_label_size"] = 8
        
        # Set edge colors and widths
        visual_style["edge_color"] = [edge_color_map.get(edge["relationship"], "gray") 
                                    for edge in self.edges]
        visual_style["edge_width"] = [1 + 3 * edge["weight"] for edge in self.edges]
        visual_style["edge_curved"] = True
        
        # Choose layout based on graph size and user preference
        if layout == "auto":
            if len(self.nodes) < 100:
                layout = "fruchterman_reingold"
            else:
                layout = "drl"
        
        # Apply selected layout
        if layout == "fruchterman_reingold":
            layout = g.layout_fruchterman_reingold(weights="weight")
        elif layout == "drl":
            layout = g.layout_drl(weights="weight")
        elif layout == "kamada_kawai":
            layout = g.layout_kamada_kawai(weights="weight")
        else:
            layout = g.layout_fruchterman_reingold(weights="weight")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the graph with simplified vertex drawing
        ig.plot(
            g,
            target=ax,
            layout=layout,
            vertex_color=visual_style["vertex_color"],
            vertex_size=visual_style["vertex_size"],
            vertex_label=visual_style["vertex_label"],
            vertex_label_size=visual_style["vertex_label_size"],
            edge_color=visual_style["edge_color"],
            edge_width=visual_style["edge_width"],
            edge_curved=visual_style["edge_curved"]
        )
        
        # Add title
        plt.title("Dependency Graph", pad=20)
        
        # Add legend for node types
        node_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10) 
                       for color in color_map.values()]
        node_labels = list(color_map.keys())
        
        # Add legend for edge types
        edge_handles = [plt.Line2D([0], [0], color=color, linewidth=2) 
                       for color in edge_color_map.values()]
        edge_labels = list(edge_color_map.keys())
        
        # Add legend
        plt.legend(handles=node_handles + edge_handles, 
                  labels=node_labels + edge_labels, 
                  loc='upper right', bbox_to_anchor=(1.1, 1))
        
        # Add statistics
        stats_text = f"Nodes: {len(self.nodes)} | Edges: {len(self.edges)} | Clusters: {max(n['cluster'] for n in self.nodes) + 1}"
        plt.figtext(0.02, 0.02, stats_text, fontsize=8)
        
        # Save if requested
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {output_file}")
        
        # Show if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        plt.close()

async def main():
    # Example usage
    directory_to_parse = "/Users/vishwasbheda/Developer/Maunik_dependency/Testing_folder"
    
    analyzer = MergedDependencyAnalyzer()
    await analyzer.build_graph_from_directory(directory_to_parse)
    
    # Save JSON output
    output_file = "Testing_folder_merged2.json"
    analyzer.save_to_json(output_file)
    
    # Generate visualization
    analyzer.visualize(
        output_file="Testing_folder_merged2.png",
        show=True,
        layout="auto",
        figsize=(15, 10)
    )
    
    print(f"Enhanced dependency graph has been saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main()) 