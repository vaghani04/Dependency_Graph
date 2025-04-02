import os
import ast
import json
import collections
import argparse

class CodeGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        # We keep separate lists for nodes and edges.
        self.nodes = []  # List of dicts: { "id": int, "name": str, "type": str, "file": str, ... }
        self.edges = []  # List of dicts: { "source": int, "target": int, "relationship": str, ... }
        
        self.node_counter = 0
        
        self.file_index = {}
        self.class_index = {}
        self.function_index = {}
        self.variable_index = {}
        
        # Track imports to resolve cross-file function calls
        # Maps: (current_file, imported_name) -> (source_module, source_name)
        self.imports = {}
        # Maps: module_name -> file_path (to resolve import locations)
        self.module_files = {}
        
        # Context stacks to know if we are inside a file, class, or function
        self.file_stack = []     # track current file node
        self.class_stack = []    # track current class node
        self.function_stack = [] # track current function node
        self.variable_stack = [] # track current variable node
        
        # Enhanced context tracking for recursion management
        self.current_scope = []  # Tracks the full path of the current scope
        
        # A map to track paths from module names to file paths
        self.module_to_file_map = {}
        
        # Function signature tracking to distinguish functions with same name
        self.function_signatures = {}  # Maps function_id -> signature hash
        
        # Current file being processed
        self.current_file = None
        
        # Track call chains for deeper analysis
        self.call_graph = {}  # Maps caller_id -> list of called function ids

    def _get_or_create_node(self, name, node_type, file_path, line_no=None):
        """
        Create nodes with a smarter indexing system to avoid duplicates
        across files for functions and classes.
        """
        # Handle each type differently for better unique identification
        if node_type == "file":
            if file_path in self.file_index:
                return self.file_index[file_path]
            
            # Create new node
            self.node_counter += 1
            node_id = self.node_counter
            
            node_info = {
                "id": node_id,
                "name": name,
                "type": node_type,
                "file": file_path,
                "line_no": line_no,
                "locations": [{"file": file_path, "line_no": line_no}]
            }
            self.nodes.append(node_info)
            self.file_index[file_path] = node_id
            return node_id
            
        elif node_type == "class":
            if name in self.class_index:
                # Update locations where this class is found
                for node in self.nodes:
                    if node["id"] == self.class_index[name]:
                        if "locations" not in node:
                            node["locations"] = []
                        
                        # Check if this location is already recorded
                        location_exists = False
                        for loc in node["locations"]:
                            if loc["file"] == file_path and loc["line_no"] == line_no:
                                location_exists = True
                                break
                        
                        if not location_exists:
                            node["locations"].append({"file": file_path, "line_no": line_no})
                        break
                return self.class_index[name]
            
            # Create new node
            self.node_counter += 1
            node_id = self.node_counter
            
            node_info = {
                "id": node_id,
                "name": name,
                "type": node_type,
                "file": file_path,  # First occurrence 
                "line_no": line_no, # First occurrence
                "locations": [{"file": file_path, "line_no": line_no}]
            }
            self.nodes.append(node_info)
            self.class_index[name] = node_id
            return node_id
            
        elif node_type == "function":
            # For methods in classes, we might want to distinguish them by both class and name
            # to avoid conflicts between different classes with methods of the same name
            if self.class_stack:
                current_class_id = self.class_stack[-1]
                # Find the class name
                class_name = None
                for node in self.nodes:
                    if node["id"] == current_class_id:
                        class_name = node["name"]
                        break
                
                if class_name:
                    function_key = f"{class_name}.{name}"
                else:
                    function_key = name
            else:
                function_key = name
            
            if function_key in self.function_index:
                # Update locations where this function is found
                for node in self.nodes:
                    if node["id"] == self.function_index[function_key]:
                        if "locations" not in node:
                            node["locations"] = []
                        
                        # Check if this location is already recorded
                        location_exists = False
                        for loc in node["locations"]:
                            if loc["file"] == file_path and loc["line_no"] == line_no:
                                location_exists = True
                                break
                        
                        if not location_exists:
                            node["locations"].append({"file": file_path, "line_no": line_no})
                        break
                return self.function_index[function_key]
            
            # Create new node
            self.node_counter += 1
            node_id = self.node_counter
            
            # Store the fully qualified name (if in a class) in the node
            if self.class_stack and class_name:
                qualified_name = f"{class_name}.{name}"
            else:
                qualified_name = name
                
            node_info = {
                "id": node_id,
                "name": name,
                "qualified_name": qualified_name,
                "type": node_type,
                "file": file_path,  # First occurrence
                "line_no": line_no, # First occurrence
                "locations": [{"file": file_path, "line_no": line_no}]
            }
            self.nodes.append(node_info)
            self.function_index[function_key] = node_id
            return node_id
            
        elif node_type == "variable":
            # Variables are scoped to their containing entity (function, class, or file)
            scope_id = None
            if self.function_stack:
                scope_id = self.function_stack[-1]
            elif self.class_stack:
                scope_id = self.class_stack[-1]
            elif self.file_stack:
                scope_id = self.file_stack[-1]
            
            # Create a composite key: (scope_id, variable_name)
            var_key = (scope_id, name)
            
            if var_key in self.variable_index:
                # Update locations where this variable is found
                for node in self.nodes:
                    if node["id"] == self.variable_index[var_key]:
                        if "locations" not in node:
                            node["locations"] = []
                        
                        # Check if this location is already recorded
                        location_exists = False
                        for loc in node["locations"]:
                            if loc["file"] == file_path and loc["line_no"] == line_no:
                                location_exists = True
                                break
                        
                        if not location_exists:
                            node["locations"].append({"file": file_path, "line_no": line_no})
                        break
                return self.variable_index[var_key]
            
            # Create new node
            self.node_counter += 1
            node_id = self.node_counter
            
            node_info = {
                "id": node_id,
                "name": name,
                "type": node_type,
                "file": file_path,
                "line_no": line_no,
                "scope_id": scope_id,
                "locations": [{"file": file_path, "line_no": line_no}]
            }
            self.nodes.append(node_info)
            self.variable_index[var_key] = node_id
            return node_id
        
        # Fallback for any other node types
        self.node_counter += 1
        node_id = self.node_counter
        
        node_info = {
            "id": node_id,
            "name": name,
            "type": node_type,
            "file": file_path,
            "line_no": line_no,
            "locations": [{"file": file_path, "line_no": line_no}]
        }
        self.nodes.append(node_info)
        return node_id

    def _add_edge(self, source_id, target_id, relationship, line_no=None):
        """
        Add an edge of type `relationship` from source to target.
        """
        # Check if this exact edge already exists
        for edge in self.edges:
            if (edge["source"] == source_id and 
                edge["target"] == target_id and 
                edge["relationship"] == relationship):
                # Edge already exists, might want to update line_no or other attributes
                if line_no is not None and "line_no" in edge and edge["line_no"] != line_no:
                    # Could store multiple line numbers if needed
                    if isinstance(edge["line_no"], list):
                        if line_no not in edge["line_no"]:
                            edge["line_no"].append(line_no)
                    else:
                        edge["line_no"] = [edge["line_no"], line_no]
                return
        
        # If edge doesn't exist, create it
        edge_info = {
            "source": source_id,
            "target": target_id,
            "relationship": relationship,  # "classin", "methodin", "call", "var", "instantiate", "parentof", etc.
            "line_no": line_no
        }
        self.edges.append(edge_info)

    # ----------------------------------------------------------------------
    # AST Parsing Logic
    # ----------------------------------------------------------------------
    def parse_python_file(self, file_path):
        """
        Parse a single Python file to extract:
          - class definitions
          - function definitions
          - function calls
          - variable assignments
          - class instantiations
          - inheritance relationships
          - import statements
        and build corresponding nodes and edges in our graph.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            file_contents = f.read()
        
        try:
            tree = ast.parse(file_contents, filename=file_path)
        except SyntaxError:
            # If there's a syntax error, skip this file
            return
        
        # Set current file
        self.current_file = file_path
        
        # Register this file as a module
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        self.module_files[module_name] = file_path
        
        # If the file is in a directory with an __init__.py, it might be a package
        dir_path = os.path.dirname(file_path)
        if os.path.exists(os.path.join(dir_path, "__init__.py")):
            # The module is part of a package
            package_parts = []
            current_dir = dir_path
            
            # Traverse upwards through directories to find all package components
            while os.path.exists(os.path.join(current_dir, "__init__.py")):
                package_parts.insert(0, os.path.basename(current_dir))
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Avoid infinite loop
                    break
                current_dir = parent_dir
            
            if package_parts:
                # Register the fully qualified package name
                full_package = ".".join(package_parts)
                full_module = f"{full_package}.{module_name}"
                self.module_files[full_module] = file_path
                
                # Register subpackages too
                for i in range(1, len(package_parts) + 1):
                    subpackage = ".".join(package_parts[:i])
                    self.module_files[subpackage] = os.path.join(
                        os.path.dirname(dir_path), 
                        *package_parts[:i], 
                        "__init__.py"
                    )
        
        # We'll create a node for the file itself so we know everything belongs to it.
        file_name = os.path.basename(file_path)
        file_node_id = self._get_or_create_node(
            name=file_name,
            node_type="file",
            file_path=file_path
        )
        
        # Push onto the file stack
        self.file_stack.append(file_node_id)
        
        # Use self.visit() on the tree to call our overridden visit_* methods
        self.visit(tree)
        
        # Pop file stack
        self.file_stack.pop()
        
        # Clear current file reference
        self.current_file = None

    def visit_ClassDef(self, node):
        """
        Called when we find a 'class MyClass:' definition in Python code.
        """
        current_file_id = self.file_stack[-1] if self.file_stack else None
        
        class_node_id = self._get_or_create_node(
            name=node.name,
            node_type="class",
            file_path=self._current_file_path(),
            line_no=node.lineno
        )
        
        # We create an edge from the file to this class with relationship "classin"
        # (or from the containing class if you wanted nested classes, but typically file->class).
        if current_file_id is not None:
            self._add_edge(current_file_id, class_node_id, "classin", line_no=node.lineno)
        
        # Handle inheritance (parentof relationship)
        for base in node.bases:
            if isinstance(base, ast.Name):
                parent_class_name = base.id
                parent_class_id = self._get_or_create_node(
                    name=parent_class_name,
                    node_type="class",
                    file_path=self._current_file_path(),
                    line_no=node.lineno
                )
                # Add "parentof" relationship: parent -> child
                self._add_edge(parent_class_id, class_node_id, "parentof", line_no=node.lineno)
        
        # Push onto class stack
        self.class_stack.append(class_node_id)
        
        # Visit the body (function defs, etc.)
        self.generic_visit(node)
        
        # Pop class stack
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        """
        Called when we find 'def my_function(...)' in Python code.
        """
        # Generate a function signature to distinguish between functions with the same name
        func_signature = self._generate_function_signature(node)
        
        # Build a fully qualified name for this function
        qualified_name = node.name
        module_path = self._get_module_path_from_file(self._current_file_path())
        
        # If we're in a class, prefix with the class name
        if self.class_stack:
            current_class_id = self.class_stack[-1]
            class_name = None
            for n in self.nodes:
                if n["id"] == current_class_id:
                    class_name = n["name"]
                    break
            if class_name:
                qualified_name = f"{class_name}.{qualified_name}"
        
        # Add module path to create a globally unique identifier
        fully_qualified_name = f"{module_path}.{qualified_name}" if module_path else qualified_name
        
        # Create a composite key that includes the signature
        func_key = f"{fully_qualified_name}#{func_signature}"
        
        # Check if we already have this function in our index
        func_node_id = None
        if func_key in self.function_index:
            func_node_id = self.function_index[func_key]
            
            # Update the locations for this function
            for node_item in self.nodes:
                if node_item["id"] == func_node_id:
                    if "locations" not in node_item:
                        node_item["locations"] = []
                    
                    new_location = {
                        "file": self._current_file_path(),
                        "line_no": node.lineno
                    }
                    
                    # Check if this location is already recorded
                    if not any(loc["file"] == new_location["file"] and 
                              loc["line_no"] == new_location["line_no"] 
                              for loc in node_item["locations"]):
                        node_item["locations"].append(new_location)
                    break
        
        # If we don't have this function yet, create a new node
        if func_node_id is None:
            self.node_counter += 1
            func_node_id = self.node_counter
            
            func_node = {
                "id": func_node_id,
                "name": node.name,
                "type": "function",
                "qualified_name": qualified_name,
                "fully_qualified_name": fully_qualified_name,
                "signature": func_signature,
                "file": self._current_file_path(),
                "line_no": node.lineno,
                "locations": [{
                    "file": self._current_file_path(),
                    "line_no": node.lineno
                }],
                "parameters": [arg.arg for arg in node.args.args],
                "module_path": module_path
            }
            
            # Add docstring if available
            if (len(node.body) > 0 and 
                isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Str)):
                func_node["docstring"] = node.body[0].value.s
            
            self.nodes.append(func_node)
            self.function_index[func_key] = func_node_id
            
            # Also index by simpler keys for easier lookup
            # This allows finding a function even if we don't know its exact signature
            simple_key = fully_qualified_name
            if simple_key not in self.function_index:
                self.function_index[simple_key] = func_node_id
        
        # Decide if we're in a class or at the file level and add appropriate edge
        if self.class_stack:
            # We are inside a class, so add an edge: class -> function with "methodin"
            current_class_id = self.class_stack[-1]
            self._add_edge(current_class_id, func_node_id, "methodin", line_no=node.lineno)
        else:
            # We are at the file level, so add an edge: file -> function with "methodin"
            if self.file_stack:
                current_file_id = self.file_stack[-1]
                self._add_edge(current_file_id, func_node_id, "methodin", line_no=node.lineno)
        
        # Push onto function stack
        self.function_stack.append(func_node_id)
        
        # Update current scope for deeper context tracking
        self.current_scope.append(("function", func_node_id, node.name))
        
        # Visit the function body (look for calls, etc.)
        self.generic_visit(node)
        
        # Pop function stack
        self.function_stack.pop()
        
        # Pop from current scope
        self.current_scope.pop()

    def visit_Call(self, node):
        """
        Called when we find a function call, e.g. foo(), bar.baz().
        We'll track function calls across files by resolving imports.
        """
        # Get the current caller context
        caller_id = None
        if self.function_stack:
            caller_id = self.function_stack[-1]
        elif self.class_stack:
            caller_id = self.class_stack[-1]
        elif self.file_stack:
            caller_id = self.file_stack[-1]
        
        # Initialize variables to track the called function
        called_func_name = None
        called_func_module = None
        is_method_call = False
        object_var_name = None
        
        # Handle different types of function calls
        if isinstance(node.func, ast.Name):
            # A simple call like foo()
            called_func_name = node.func.id
            
            # Check if this is an imported function from another module
            import_key = (self.current_file, called_func_name)
            if import_key in self.imports:
                source_module, source_name = self.imports[import_key]
                called_func_module = source_module
                
                if source_name is not None:
                    # This was a direct import like 'from module import function'
                    called_func_name = source_name
        
        elif isinstance(node.func, ast.Attribute):
            # A call like object.method() or module.function()
            called_func_name = node.func.attr
            is_method_call = True
            
            if isinstance(node.func.value, ast.Name):
                object_var_name = node.func.value.id
                
                # Check if this is a module
                import_key = (self.current_file, object_var_name)
                if import_key in self.imports:
                    # This is a module.function() call
                    source_module, _ = self.imports[import_key]
                    called_func_module = source_module
                    is_method_call = False
                else:
                    # This might be an object.method() call
                    # Try to determine the object's type
                    var_type = self._get_variable_type(object_var_name)
                    if var_type:
                        called_func_module = var_type
        
        # If we couldn't identify the function, skip
        if not called_func_name:
            self.generic_visit(node)
            return
        
        # Now try to find the target function node
        target_func_id = self._find_function_node(
            func_name=called_func_name,
            module_name=called_func_module,
            is_method=is_method_call,
            line_no=node.lineno
        )
        
        # If we found a target function, add a call edge
        if target_func_id and caller_id:
            self._add_edge(caller_id, target_func_id, "call", line_no=node.lineno)
            
            # Track in call graph for deeper analysis
            if caller_id not in self.call_graph:
                self.call_graph[caller_id] = []
            if target_func_id not in self.call_graph[caller_id]:
                self.call_graph[caller_id].append(target_func_id)
        
        # Continue to process arguments
        self.generic_visit(node)

    def _find_function_node(self, func_name, module_name=None, is_method=False, line_no=None):
        """
        Find a function node in the codebase by name and optional module.
        
        Args:
            func_name (str): The name of the function to find
            module_name (str, optional): The module where the function is defined
            is_method (bool): Whether this is likely a method call
            line_no (int, optional): The line number where the call occurs
            
        Returns:
            int: The node ID of the found function, or None if not found
        """
        # Case 1: We know the fully qualified name
        if module_name:
            fully_qualified_name = f"{module_name}.{func_name}"
            if fully_qualified_name in self.function_index:
                return self.function_index[fully_qualified_name]
        
        # Case 2: Look for methods if this is a method call
        if is_method:
            # Try to find methods with this name in any class
            for node in self.nodes:
                if (node["type"] == "function" and 
                    node["name"] == func_name and 
                    "qualified_name" in node and 
                    "." in node["qualified_name"]):
                    return node["id"]
        
        # Case 3: Simple function lookup by name (less accurate but better than nothing)
        candidates = []
        for node in self.nodes:
            if node["type"] == "function" and node["name"] == func_name:
                candidates.append(node["id"])
        
        if candidates:
            # If we have multiple candidates, prefer ones defined in the current file
            current_file = self._current_file_path()
            for candidate_id in candidates:
                for node in self.nodes:
                    if node["id"] == candidate_id and node.get("file") == current_file:
                        return candidate_id
            
            # If no match in current file, just return the first candidate
            return candidates[0]
        
        # Case 4: We couldn't find the function, so create a placeholder node
        self.node_counter += 1
        placeholder_id = self.node_counter
        
        placeholder_node = {
            "id": placeholder_id,
            "name": func_name,
            "type": "function",
            "status": "placeholder",  # Mark as a placeholder for future resolution
            "line_no": line_no,
            "file": self._current_file_path(),
            "reference_count": 1
        }
        
        if module_name:
            placeholder_node["module"] = module_name
            placeholder_node["fully_qualified_name"] = f"{module_name}.{func_name}"
        
        self.nodes.append(placeholder_node)
        
        # Index the placeholder for future reference
        placeholder_key = f"{module_name}.{func_name}" if module_name else func_name
        self.function_index[placeholder_key] = placeholder_id
        
        return placeholder_id

    def _get_variable_type(self, var_name):
        """
        Helper method to try to determine the type of a variable.
        This is a simple implementation and could be improved.
        """
        # Find the variable node
        var_node_id = None
        
        # Try in current function scope first
        if self.function_stack:
            scope_id = self.function_stack[-1]
            var_key = (scope_id, var_name)
            if var_key in self.variable_index:
                var_node_id = self.variable_index[var_key]
        
        # Try in class scope if not found
        if var_node_id is None and self.class_stack:
            scope_id = self.class_stack[-1]
            var_key = (scope_id, var_name)
            if var_key in self.variable_index:
                var_node_id = self.variable_index[var_key]
        
        # Try in file scope if not found
        if var_node_id is None and self.file_stack:
            scope_id = self.file_stack[-1]
            var_key = (scope_id, var_name)
            if var_key in self.variable_index:
                var_node_id = self.variable_index[var_key]
        
        if var_node_id is None:
            return None
        
        # Look for "instantiate" edges from this variable to a class
        for edge in self.edges:
            if edge["source"] == var_node_id and edge["relationship"] == "instantiate":
                class_id = edge["target"]
                for node in self.nodes:
                    if node["id"] == class_id:
                        return node["name"]
        
        return None

    def visit_Assign(self, node):
        """
        Called when we find variable assignments like x = 5 or x = foo()
        """
        # Process all targets (left side of assignment)
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Simple variable assignment like x = ...
                var_name = target.id
                var_node_id = self._get_or_create_node(
                    name=var_name,
                    node_type="variable",
                    file_path=self._current_file_path(),
                    line_no=node.lineno
                )
                
                # Determine the container scope and add the "var" relationship
                if self.function_stack:
                    # Variable inside a function
                    current_func_id = self.function_stack[-1]
                    self._add_edge(current_func_id, var_node_id, "var", line_no=node.lineno)
                elif self.class_stack:
                    # Class variable
                    current_class_id = self.class_stack[-1]
                    self._add_edge(current_class_id, var_node_id, "var", line_no=node.lineno)
                elif self.file_stack:
                    # Module-level variable
                    current_file_id = self.file_stack[-1]
                    self._add_edge(current_file_id, var_node_id, "var", line_no=node.lineno)
                
                # Check if right side is a class instantiation
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    class_name = node.value.func.id
                    # Check if this is a known class
                    for n in self.nodes:
                        if n["type"] == "class" and n["name"] == class_name:
                            class_node_id = n["id"]
                            # Add instantiate relationship: var -> class
                            self._add_edge(var_node_id, class_node_id, "instantiate", line_no=node.lineno)
        
        # Continue to process the right-hand side of the assignment
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """
        Called when we find annotated variable assignments like x: int = 5
        """
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            var_node_id = self._get_or_create_node(
                name=var_name,
                node_type="variable",
                file_path=self._current_file_path(),
                line_no=node.lineno
            )
            
            # Determine the container scope and add the "var" relationship
            if self.function_stack:
                # Variable inside a function
                current_func_id = self.function_stack[-1]
                self._add_edge(current_func_id, var_node_id, "var", line_no=node.lineno)
            elif self.class_stack:
                # Class variable
                current_class_id = self.class_stack[-1]
                self._add_edge(current_class_id, var_node_id, "var", line_no=node.lineno)
            elif self.file_stack:
                # Module-level variable
                current_file_id = self.file_stack[-1]
                self._add_edge(current_file_id, var_node_id, "var", line_no=node.lineno)
            
            # Check for class instantiation in value (if it exists)
            if node.value and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                class_name = node.value.func.id
                # Check if this is a known class
                for n in self.nodes:
                    if n["type"] == "class" and n["name"] == class_name:
                        class_node_id = n["id"]
                        # Add instantiate relationship: var -> class
                        self._add_edge(var_node_id, class_node_id, "instantiate", line_no=node.lineno)
        
        # Continue processing
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """
        Process 'import foo' or 'import foo as bar' statements.
        """
        for name in node.names:
            module_name = name.name
            alias = name.asname or module_name
            
            # Register this import in our imports dictionary
            import_key = (self.current_file, alias)
            self.imports[import_key] = (module_name, None)  # No specific attribute imported
            
        # Continue with the rest of the AST
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """
        Process 'from foo import bar' or 'from foo import bar as baz' statements.
        """
        module_name = node.module
        for name in node.names:
            imported_name = name.name
            alias = name.asname or imported_name
            
            # Register this import in our imports dictionary
            import_key = (self.current_file, alias)
            self.imports[import_key] = (module_name, imported_name)
            
        # Continue with the rest of the AST
        self.generic_visit(node)
    
    # ----------------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------------
    def _current_file_path(self):
        """
        Return the file path string from the top file node, if available.
        """
        if not self.file_stack:
            return None
        # We can look up the node's file attribute from self.nodes
        file_node_id = self.file_stack[-1]
        for nd in self.nodes:
            if nd["id"] == file_node_id:
                return nd["file"]
        return None

    def _generate_function_signature(self, func_node):
        """
        Generate a signature hash for a function to distinguish between functions with the same name
        but different parameter lists. This helps identify unique functions across the codebase.
        
        Args:
            func_node (ast.FunctionDef): AST node for the function definition
            
        Returns:
            str: A signature string that represents the function's parameter structure
        """
        # Start with base name
        signature_parts = [func_node.name]
        
        # Add parameters
        for arg in func_node.args.args:
            arg_name = arg.arg
            
            # Try to get type annotation if available
            arg_type = "unknown"
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Attribute):
                    # Handle cases like module.Type
                    arg_type = f"{self._get_attribute_full_name(arg.annotation)}"
                elif isinstance(arg.annotation, ast.Subscript):
                    # Handle generic types like List[str]
                    if isinstance(arg.annotation.value, ast.Name):
                        container_type = arg.annotation.value.id
                        if isinstance(arg.annotation.slice, ast.Index):  # Python 3.8 and below
                            if isinstance(arg.annotation.slice.value, ast.Name):
                                inner_type = arg.annotation.slice.value.id
                            else:
                                inner_type = "any"
                        else:  # Python 3.9+
                            if isinstance(arg.annotation.slice, ast.Name):
                                inner_type = arg.annotation.slice.id
                            else:
                                inner_type = "any"
                        arg_type = f"{container_type}[{inner_type}]"
                    else:
                        arg_type = "complex"
            
            signature_parts.append(f"{arg_name}:{arg_type}")
        
        # Add *args if present
        if func_node.args.vararg:
            signature_parts.append(f"*{func_node.args.vararg.arg}")
        
        # Add **kwargs if present
        if func_node.args.kwarg:
            signature_parts.append(f"**{func_node.args.kwarg.arg}")
        
        # Add return type if available
        if hasattr(func_node, 'returns') and func_node.returns is not None:
            if isinstance(func_node.returns, ast.Name):
                return_type = func_node.returns.id
            elif isinstance(func_node.returns, ast.Attribute):
                return_type = f"{self._get_attribute_full_name(func_node.returns)}"
            else:
                return_type = "complex"
            signature_parts.append(f"-> {return_type}")
        
        # Join all parts to create the signature
        return "|".join(signature_parts)
    
    def _get_attribute_full_name(self, node):
        """
        Get the full name of an attribute node (e.g., 'module.submodule.Type')
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_full_name(node.value)}.{node.attr}"
        return "unknown"

    def _get_module_path_from_file(self, file_path):
        """
        Convert a file path to a Python module path.
        For example: '/path/to/project/module/submodule/file.py' -> 'module.submodule.file'
        
        This helps create fully qualified names for classes and functions.
        """
        if not file_path:
            return None
            
        # Get the filename without extension
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Initialize the module path with just the filename
        module_path_parts = [module_name]
        
        # Get the directory containing this file
        dir_path = os.path.dirname(file_path)
        
        # Check if this directory has an __init__.py (making it a package)
        if os.path.exists(os.path.join(dir_path, "__init__.py")):
            # Find all parent packages by traveling up the directory tree
            current_dir = dir_path
            package_parts = []
            
            while os.path.exists(os.path.join(current_dir, "__init__.py")):
                package_name = os.path.basename(current_dir)
                package_parts.insert(0, package_name)
                
                # Move up one directory
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Prevent infinite loop
                    break
                current_dir = parent_dir
            
            # Add package parts to the module path
            module_path_parts = package_parts + module_path_parts
        
        # Join all parts with dots to form the Python module path
        return ".".join(module_path_parts)

    def build_graph_from_directory(self, directory):
        """
        Recursively walk the given directory,
        parse every .py file found.
        """
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".py"):
                    file_path = os.path.join(root, filename)
                    self.parse_python_file(file_path)
    
    def save_to_json(self, output_path):
        """
        Save the entire graph (nodes + edges) into a JSON file.
        """
        data = {
            "nodes": self.nodes,
            "edges": self.edges
        }
        statistical_data = {
            "statistics": self.generate_statistics()
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        with open(output_path.split(".json")[0]+"_stats.json", "w", encoding="utf-8") as f:
            json.dump(statistical_data, f, indent=4)
    
    def generate_statistics(self):
        """
        Generate comprehensive statistics about the knowledge graph.
        Returns a dictionary with various metrics.
        """
        # Count node types
        node_types = collections.Counter([node.get('type', 'unknown') for node in self.nodes])
        
        # Count relationship types
        relationship_types = collections.Counter([edge.get('relationship', 'unknown') for edge in self.edges])
        
        # Count files by extension
        file_extensions = collections.Counter([
            os.path.splitext(node.get('file', ''))[1] 
            for node in self.nodes 
            if node.get('type') == 'file' and node.get('file')
        ])
        
        # Count nodes per file
        nodes_per_file = {}
        for node in self.nodes:
            if 'file' in node and node['file']:
                file_path = node['file']
                if file_path not in nodes_per_file:
                    nodes_per_file[file_path] = collections.Counter()
                nodes_per_file[file_path][node.get('type', 'unknown')] += 1
        
        # Find most connected nodes (highest degree)
        node_connections = {}
        for edge in self.edges:
            source = edge.get('source')
            target = edge.get('target')
            
            if source not in node_connections:
                node_connections[source] = {'in': 0, 'out': 0}
            if target not in node_connections:
                node_connections[target] = {'in': 0, 'out': 0}
                
            node_connections[source]['out'] += 1
            node_connections[target]['in'] += 1
        
        # Find top 5 most connected nodes
        most_connected = []
        if node_connections:
            # Sort by total connections (in + out)
            sorted_connections = sorted(
                node_connections.items(), 
                key=lambda x: x[1]['in'] + x[1]['out'],
                reverse=True
            )
            
            # Get top 5
            for node_id, counts in sorted_connections[:5]:
                node_info = None
                for node in self.nodes:
                    if node['id'] == node_id:
                        node_info = node
                        break
                
                if node_info:
                    most_connected.append({
                        'id': node_id,
                        'name': node_info.get('name', f'Node-{node_id}'),
                        'type': node_info.get('type', 'unknown'),
                        'in_degree': counts['in'],
                        'out_degree': counts['out'],
                        'total_connections': counts['in'] + counts['out']
                    })
        
        # Build statistics object
        stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": dict(node_types),
            "relationship_types": dict(relationship_types),
            "files_count": len([n for n in self.nodes if n.get('type') == 'file']),
            "classes_count": len([n for n in self.nodes if n.get('type') == 'class']),
            "functions_count": len([n for n in self.nodes if n.get('type') == 'function']),
            "variables_count": len([n for n in self.nodes if n.get('type') == 'variable']),
            "file_extensions": dict(file_extensions),
            "nodes_density": len(self.edges) / max(1, len(self.nodes)),  # edges to nodes ratio
            "most_connected_nodes": most_connected,
            "average_connections_per_node": sum(c['in'] + c['out'] for c in node_connections.values()) / max(1, len(node_connections)),
        }
        
        return stats
    
    def find_function_references(self, function_name, file_path=None):
        """
        Find all references to a function across the codebase with enhanced accuracy.
        This includes both the function definitions and all calls to that function,
        with complete traversal of the call chain.
        
        Args:
            function_name (str): The name of the function to find
            file_path (str, optional): The file path where the function is defined,
                                     to disambiguate functions with the same name
            
        Returns:
            dict: A dictionary with the following information:
                - 'function_node': The node representing the function
                - 'defined_in': List of files and line numbers where function is defined
                - 'called_from': List of calling locations with file, line number and calling function
                - 'call_chain': List of call paths leading to this function (deep traversal)
        """
        result = {
            "function_node": None,
            "defined_in": [],
            "called_from": [],
            "call_chain": []
        }
        
        # First, try to find the exact function node with the given file path
        target_func_id = None
        
        if file_path:
            # If file path is provided, try to find the exact match
            module_path = self._get_module_path_from_file(file_path)
            
            # Try with module path + function name
            fully_qualified_name = f"{module_path}.{function_name}" if module_path else function_name
            
            # Look for exact match first
            for node in self.nodes:
                if (node["type"] == "function" and
                    node["name"] == function_name and
                    node.get("file") == file_path):
                    target_func_id = node["id"]
                    result["function_node"] = node
                    break
            
            # If not found, try using the fully qualified name
            if not target_func_id and fully_qualified_name in self.function_index:
                target_func_id = self.function_index[fully_qualified_name]
                for node in self.nodes:
                    if node["id"] == target_func_id:
                        result["function_node"] = node
                        break
        
        # If still not found or no file path given, look for any function with this name
        if not target_func_id:
            candidates = []
            for node in self.nodes:
                if node["type"] == "function" and node["name"] == function_name:
                    candidates.append(node)
            
            if candidates:
                # If multiple functions with same name, pick the one that seems most relevant
                # Or provide all in the response
                result["function_node"] = candidates[0]
                target_func_id = candidates[0]["id"]
                
                # Also record other candidate functions with the same name
                result["other_candidates"] = candidates[1:] if len(candidates) > 1 else []
        
        if not target_func_id:
            print(f"Function '{function_name}' not found in the codebase.")
            return result
        
        # Now that we have the function node, gather all definition locations
        for node in self.nodes:
            if node["id"] == target_func_id:
                if "locations" in node:
                    for loc in node["locations"]:
                        result["defined_in"].append({
                            "file": loc["file"],
                            "line_no": loc["line_no"]
                        })
                else:
                    # Fallback
                    result["defined_in"].append({
                        "file": node.get("file"),
                        "line_no": node.get("line_no")
                    })
        
        # Find direct calls to this function
        direct_callers = []
        for edge in self.edges:
            if edge["relationship"] == "call" and edge["target"] == target_func_id:
                caller_id = edge["source"]
                direct_callers.append(caller_id)
                
                caller_node = None
                for node in self.nodes:
                    if node["id"] == caller_id:
                        caller_node = node
                        break
                
                if caller_node:
                    call_info = {
                        "caller": caller_node["name"],
                        "caller_type": caller_node["type"],
                        "file": caller_node.get("file", "unknown"),
                        "line_no": edge.get("line_no", "unknown"),
                        "caller_id": caller_id
                    }
                    
                    # If caller is a function, we want the full path to the caller
                    if caller_node["type"] == "function":
                        if "qualified_name" in caller_node:
                            call_info["caller"] = caller_node["qualified_name"]
                        if "fully_qualified_name" in caller_node:
                            call_info["fully_qualified_caller"] = caller_node["fully_qualified_name"]
                    
                    result["called_from"].append(call_info)
        
        # Find additional call chains (deep traversal)
        # This finds all paths in the call graph that lead to our target function
        def find_call_paths(node_id, current_path=None, visited=None):
            if current_path is None:
                current_path = []
            if visited is None:
                visited = set()
            
            # Avoid cycles
            if node_id in visited:
                return []
            
            visited.add(node_id)
            current_path.append(node_id)
            
            if node_id == target_func_id:
                return [current_path[:]]
            
            paths = []
            
            # Find all edges where this node is the target (i.e., callers of this node)
            callers = []
            for edge in self.edges:
                if edge["relationship"] == "call" and edge["target"] == node_id:
                    callers.append(edge["source"])
            
            for caller in callers:
                new_paths = find_call_paths(caller, current_path[:], visited.copy())
                paths.extend(new_paths)
            
            return paths
        
        # Start from all nodes and find paths to our target
        all_paths = []
        for node in self.nodes:
            if node["type"] == "function" and node["id"] != target_func_id:
                paths = find_call_paths(node["id"])
                if paths:
                    all_paths.extend(paths)
        
        # Convert paths of node IDs to readable paths with function names and locations
        for path in all_paths:
            readable_path = []
            for node_id in path:
                for node in self.nodes:
                    if node["id"] == node_id:
                        node_info = {
                            "id": node_id,
                            "name": node["name"],
                            "type": node["type"],
                            "file": node.get("file", "unknown")
                        }
                        if "qualified_name" in node:
                            node_info["qualified_name"] = node["qualified_name"]
                        readable_path.append(node_info)
                        break
            
            if readable_path:
                result["call_chain"].append(readable_path)
        
        return result

if __name__ == "__main__":

    directory_to_parse = "/Users/maunikvaghani/Developer/Hackathons/H1/Testing_folder"
    # directory_to_parse = "RAG_Playground"
    
    print(f"Building knowledge graph for codebase in: {directory_to_parse}")
    builder = CodeGraphBuilder()
    builder.build_graph_from_directory(directory_to_parse)

    output_path = "outputs/Testing_folder.json"
    # output_path = "RAG_Playground.json"
    builder.save_to_json(output_path)
    print(f"Knowledge graph has been saved to {output_path}.")
    
    print("\nDone!")