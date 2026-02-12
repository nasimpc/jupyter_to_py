import sys
import json
import re
import os

def parse_py_to_ipynb(py_content):
    lines = py_content.splitlines()
    
    # Split by "# In[ ]:" or similar markers
    # We'll group lines into segments
    segments = []
    current_segment = []
    
    # Python-export marker regex
    # Matches "# In[ ]:" or "# In[5]:" etc
    marker_pattern = re.compile(r'^#\s*In\[\s*\d*\s*\]:$')

    for line in lines:
        if marker_pattern.match(line.strip()):
            if current_segment:
                segments.append(current_segment)
            current_segment = []
        else:
            current_segment.append(line)
    
    if current_segment:
        segments.append(current_segment)

    # Now convert segments into cells
    cells = []
    
    for segment in segments:
        parsed_cells = process_segment(segment)
        cells.extend(parsed_cells)
        
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def process_segment(lines):
    # Analyze a segment of lines to determine if it's Code, Markdown, or Split
    
    # 1. Identify where method code ends
    last_code_index = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            last_code_index = i
            
    # If no code found, it's all comments/empty -> Markdown
    if last_code_index == -1:
        # Check if it's completely empty
        if not any(line.strip() for line in lines):
            return [] # Skip empty segments
        return [create_markdown_cell(lines)]

    # We have code up to last_code_index.
    # Check lines AFTER last_code_index.
    code_lines = lines[:last_code_index+1]
    tail_lines = lines[last_code_index+1:]
    
    # Analyze tail logic
    # Heuristic: If tail starts with empty lines, and then has comments, it's a separate Markdown cell.
    # OR if tail comments look like headers (# #).
    
    markdown_lines = []
    
    # Strip leading empty lines from tail to check for content
    tail_start_idx = 0
    while tail_start_idx < len(tail_lines) and not tail_lines[tail_start_idx].strip():
        tail_start_idx += 1
        
    active_tail = tail_lines[tail_start_idx:]
    
    has_gap = (tail_start_idx > 0)
    
    if active_tail:
        # Check if it looks like markdown
        # Typical markdown headers in these exports start with "# # " or "# ## "
        is_header = any(line.strip().startswith('# #') for line in active_tail)
        
        if has_gap or is_header:
            # Spilt!
            markdown_lines = active_tail
            # code_lines is already set
        else:
            # Append tail to code lines (it's just comments attached to code)
            code_lines.extend(tail_lines)
    else:
        # Just whitespace in tail, ignore
        pass

    result_cells = [create_code_cell(code_lines)]
    if markdown_lines:
        result_cells.append(create_markdown_cell(markdown_lines))
        
    return result_cells

def create_code_cell(lines):
    # Trim trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()
    
    source = [line + '\n' for line in lines]
    if source:
        source[-1] = source[-1].rstrip('\n') # Last line no newline
        
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def create_markdown_cell(lines):
    # Convert python comments to markdown
    # Strip leading '# '
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            # Find the index of the first '#' to handle indentation
            raw_stripped = line.lstrip()
            # If it looks like "#  Content", we want " Content"
            # If it looks like "#Content", we want "Content"
            if raw_stripped.startswith('# '):
                clean_lines.append(raw_stripped[2:])
            elif raw_stripped.startswith('#'):
                 clean_lines.append(raw_stripped[1:])
            else:
                 clean_lines.append(line)
        elif not stripped:
            clean_lines.append('')
        else:
            # Should not happen if logic is correct
            clean_lines.append(line)
            
    # Trim leading/trailing empty lines
    while clean_lines and not clean_lines[0].strip():
        clean_lines.pop(0)
    while clean_lines and not clean_lines[-1].strip():
        clean_lines.pop()

    source = [line + '\n' for line in clean_lines]
    if source:
        source[-1] = source[-1].rstrip('\n')
        
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python py_to_ipynb.py <input_file.py>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = os.path.splitext(input_file)[0] + ".ipynb"
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    
    print(f"Converting {input_file} to {output_file}...")
    
    with open(input_file, 'r') as f:
        content = f.read()
        
    nb = parse_py_to_ipynb(content)
    
    with open(output_file, 'w') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Done. Saved to {output_file}")
