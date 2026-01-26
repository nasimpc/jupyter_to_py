#!/usr/bin/env python3
import os
import sys
import argparse
import nbformat
from nbconvert import PythonExporter

def convert_notebook(notebook_path, output_dir=None):
    """
    Converts a single .ipynb file to a .py script.
    """
    if not os.path.exists(notebook_path):
        print(f"Error: File not found - {notebook_path}")
        return

    print(f"Converting: {notebook_path}...")

    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_node = nbformat.read(f, as_version=4)

        # Initialize the Python exporter
        exporter = PythonExporter()
        
        # Convert the notebook to Python code
        (python_code, resources) = exporter.from_notebook_node(notebook_node)

        # Determine output path
        base_name = os.path.splitext(os.path.basename(notebook_path))[0]
        output_filename = f"{base_name}.py"
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_path = os.path.join(os.path.dirname(notebook_path), output_filename)

        # Write the python code to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(python_code)
        
        print(f"Successfully created: {output_path}")

    except Exception as e:
        print(f"Failed to convert {notebook_path}: {e}")

def process_input(input_path, output_dir=None):
    """
    Process input path (file or directory).
    """
    if os.path.isfile(input_path):
        if input_path.endswith('.ipynb'):
            convert_notebook(input_path, output_dir)
        else:
            print(f"Skipping non-notebook file: {input_path}")
    elif os.path.isdir(input_path):
        print(f"Scanning directory: {input_path}")
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.ipynb'):
                    full_path = os.path.join(root, file)
                    convert_notebook(full_path, output_dir)
    else:
        print(f"Invalid input path: {input_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Jupyter Notebooks (.ipynb) to Python scripts (.py) for easier LLM reading.")
    
    parser.add_argument("input_path", help="Path to a specific .ipynb file or a directory containing notebooks.")
    parser.add_argument("-o", "--output", help="Directory to save the converted Python scripts. Defaults to the same directory as the source.")

    args = parser.parse_args()

    process_input(args.input_path, args.output)

if __name__ == "__main__":
    main()
