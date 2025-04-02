# Code Knowledge Graph

A Python tool that builds and analyzes knowledge graphs from Python codebases, helping developers understand code dependencies and relationships between functions, classes, and files.

## Approach

The tool uses Python's Abstract Syntax Tree (AST) to parse Python files and build a comprehensive knowledge graph. It tracks:
- Nodes
  - Files
  - Classes
  - Functions
  - Varaibles
- Edges
  - classin
  - methodin
  - call
  - var
  - instantiate
  - parentof

## Steps to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Build knowledge graph:
   Provide the path to store the .json file in the .py file
```bash
python knowledge_graph.py
```
This will create a JSON file containing the knowledge graph.

3. Query the graph:
   Provide the path of .json to give the graph's context
   Provide the path to store the .png file in the .py file
```bash
python graph_query.py graph.json function_name --file /path/to/file.py --visualize
```

4. Visualize the entire graph:
   Provide the path of .json to give the graph's context
   Provide the path to store the .png file in the .py file
```bash
python visualizer.py
```

## Features

- Build a comprehensive knowledge graph of Python code dependencies
- Track function calls across files
- Identify where functions are defined and where they are called
- Query and visualize function usage throughout the codebase

## Components

- `knowledge_graph.py`: Builds the knowledge graph by parsing Python files
- `graph_query.py`: Queries the knowledge graph to find function references and visualize call chains
- `visualizer.py`: Additional visualization tools for the knowledge graph
