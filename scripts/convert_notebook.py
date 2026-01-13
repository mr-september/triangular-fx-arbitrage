import nbformat
import os

def convert_to_markdown(ipynb_path):
    print(f"Reading notebook from: {ipynb_path}")
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    output_text = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            output_text.append(f"### MARKDOWN BLOCK\n{cell.source}\n")
        elif cell.cell_type == 'code':
            output_text.append(f"### CODE BLOCK\n```python\n{cell.source}\n```")
            # Wrap prints/errors in tags
            if cell.outputs:
                output_text.append("<execution_output>")
                for out in cell.outputs:
                    if 'text' in out:
                        output_text.append(out['text'])
                    elif 'ename' in out: # For Errors
                        output_text.append(f"{out['ename']}: {out['evalue']}")
                    elif 'data' in out and 'text/plain' in out['data']: # Capture text/plain output if 'text' isn't top-level
                         output_text.append(out['data']['text/plain'])
                output_text.append("</execution_output>\n")
    
    return "\n".join(output_text)

# Save the result
if __name__ == "__main__":
    input_file = os.path.join("notebooks", "research_report_v2.ipynb")
    output_file = "research_report_v2.md"
    
    if not os.path.exists(input_file):
        # Fallback for running from root or scripts dir
        if os.path.exists(os.path.join("..", input_file)):
             input_file = os.path.join("..", input_file)
        # Fallback to local if running in same dir
        elif os.path.exists("research_report_v2.ipynb"):
             input_file = "research_report_v2.ipynb"
        else:
             print(f"Error: Could not find {input_file}")
             exit(1)

    print(f"Converting {input_file}...")
    markdown_content = convert_to_markdown(input_file)
    
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"Saved conversion to {output_file}")
