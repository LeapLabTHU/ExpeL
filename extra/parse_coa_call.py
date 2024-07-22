import ast

def parse_coa_call(function_call_str):
    try:
        # Parse the string into an AST node
        node = ast.parse(function_call_str, mode='eval')

        # Ensure the node is an expression and the body is a function call
        if isinstance(node, ast.Expression) and isinstance(node.body, ast.Call):

            # Extract the function name, then extract and evaluate the args
            func_name = node.body.func.id
            args = [ast.literal_eval(arg) for arg in node.body.args]
            
            # Return the function name and arguments
            print(f"func_name: {func_name}")
            print(f"args: {args}")
            return func_name, args
        else:
            raise ValueError("Invalid function call string")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

print(parse_coa_call("create_string(\"abc\", 123)"))