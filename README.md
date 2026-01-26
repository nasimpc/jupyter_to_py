# Jupyter to Python Converter

This project provides a simple script to batch convert Jupyter Notebooks (`.ipynb`) into Python scripts (`.py`). This is useful for making notebook code easily readable by LLMs or for version control.

## Setup

1.  **Prerequisites**: Python 3 installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can convert a single file or an entire directory of notebooks.

### Run the script

**Convert a single file:**
```bash
python ipynb_converter.py /path/to/notebook.ipynb
```

**Convert all notebooks in a directory:**
```bash
python ipynb_converter.py /path/to/folder
```

### Options

*   `-o` or `--output`: Specify a separate output directory for the `.py` files.

**Example:**
```bash
python ipynb_converter.py . -o ./converted_scripts
```
