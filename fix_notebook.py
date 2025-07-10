#!/usr/bin/env python3
import json
import sys

def fix_notebook(filepath):
    """Add missing 'id' fields to notebook cells."""
    with open(filepath, 'r') as f:
        notebook = json.load(f)
    
    # Add id to each cell if missing
    for i, cell in enumerate(notebook.get('cells', [])):
        if 'id' not in cell:
            # Generate a unique ID for each cell
            cell['id'] = f'cell-{i}'
    
    # Write the fixed notebook back
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Fixed notebook: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fix_notebook(sys.argv[1])
    else:
        print("Usage: python fix_notebook.py <notebook_path>")
        print("Example: python fix_notebook.py notebooks/06_temporal_ml_integration_demo.ipynb")
        sys.exit(1)