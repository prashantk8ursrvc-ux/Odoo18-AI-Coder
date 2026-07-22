# -*- coding: utf-8 -*-
"""
Odoo 18 Developer Documentation Dataset Compiler
Automates the retrieval and extraction of Odoo 18 dev docs to compile an SFT dataset.
"""

import os
import json
import subprocess
import sys

def run_sparse_checkout():
    """Perform a shallow clone and sparse checkout of the Odoo documentation source."""
    target_dir = "odoo_docs_src"
    
    if os.path.exists(target_dir):
        print(f"[*] Directory '{target_dir}' already exists. Updating sparse checkout target...")
        try:
            subprocess.run([
                "git", "-C", target_dir, 
                "sparse-checkout", "set", "content"
            ], check=True)
            return target_dir
        except subprocess.CalledProcessError as e:
            print(f"[-] Failed to update sparse checkout: {e}", file=sys.stderr)
            sys.exit(1)
        
    print("[*] Starting sparse git clone of odoo/documentation (branch 18.0)...")
    try:
        # Clone with sparse and shallow filter
        subprocess.run([
            "git", "clone", 
            "--depth", "1", 
            "--branch", "18.0", 
            "--filter=blob:none", 
            "--sparse", 
            "https://github.com/odoo/documentation.git", 
            target_dir
        ], check=True)
        
        # Pull the entire content directory (including applications/ and developer/)
        print("[*] Setting sparse-checkout target to 'content'...")
        subprocess.run([
            "git", "-C", target_dir, 
            "sparse-checkout", "set", "content"
        ], check=True)
        
        print("[+] Sparse checkout completed successfully.")
        return target_dir
    except subprocess.CalledProcessError as e:
        print(f"[-] Git command failed: {e}", file=sys.stderr)
        sys.exit(1)

def parse_rst_file(filepath: str):
    """Parses RST files to pair explanatory paragraphs with code blocks."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    examples = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if not stripped:
            i += 1
            continue
            
        # Match reStructuredText code blocks
        if stripped.startswith(".. code-block::") or stripped.startswith("::"):
            # Backtrack to find the preceding description paragraph (ignoring comments/directives)
            context = []
            k = i - 1
            while k >= 0:
                prev_line = lines[k].strip()
                if not prev_line:
                    if context:  # Stop at paragraph boundary if we already collected context
                        break
                elif prev_line.startswith("..") or prev_line.startswith(":") or (len(prev_line) > 2 and all(c == prev_line[0] for c in prev_line) and prev_line[0] in "=-~`':\"^_*+#"):
                    # Ignore Sphinx directives, metadata, or header border lines
                    k -= 1
                    continue
                else:
                    context.insert(0, prev_line)
                k -= 1
            
            instruction = " ".join(context).strip()
            # Clean instruction sentences from common rst tags (e.g. :ref:`label`)
            instruction = re_clean_rst_refs(instruction)
            
            # If no good context could be parsed, fallback to filename reference
            if not instruction or len(instruction) < 10:
                topic = os.path.splitext(os.path.basename(filepath))[0].replace("_", " ").title()
                instruction = f"Write an Odoo 18 code snippet illustrating concepts from '{topic}'."
            
            # Read the indented block of code
            i += 1
            code_lines = []
            indent = None
            while i < len(lines):
                c_line = lines[i]
                c_stripped = c_line.strip()
                
                if not c_stripped:
                    code_lines.append("")
                    i += 1
                    continue
                    
                line_indent = len(c_line) - len(c_line.lstrip())
                if indent is None:
                    indent = line_indent
                    if indent == 0:  # Code block body must be indented
                        break
                        
                if line_indent < indent:
                    # Dedented line matches end of code block
                    break
                    
                code_lines.append(c_line[indent:].rstrip())
                i += 1
                
            code_block = "\n".join(code_lines).strip()
            if code_block:
                examples.append({
                    "instruction": f"In Odoo 18, how do you accomplish the following? {instruction}",
                    "input": "",
                    "output": code_block,
                    "source_file": os.path.relpath(filepath, "odoo_docs_src")
                })
            continue
            
        i += 1
    return examples

def re_clean_rst_refs(text: str) -> str:
    """Helper to remove sphinx/rst tags like :ref:`link` or :doc:`doc` from parsed instruction."""
    import re
    # Remove link references e.g. :ref:`something <label>` -> 'something'
    text = re.sub(r':[a-z]+:`([^`]*?)\s*<.*?>`', r'\1', text)
    # Remove simple refs e.g. :ref:`label` -> 'label'
    text = re.sub(r':[a-z]+:`([^`]+?)`', r'\1', text)
    # Remove bold/italic inline asterisks
    text = text.replace("**", "").replace("*", "")
    return text

def compile_dataset(docs_dir: str):
    """Walk directories, parse all .rst files, and dump to jsonl."""
    search_path = os.path.join(docs_dir, "content")
    if not os.path.exists(search_path):
        print(f"[-] Source directory not found at: {search_path}", file=sys.stderr)
        sys.exit(1)
        
    dataset = []
    print(f"[*] Parsing .rst files in '{search_path}'...")
    
    for root, _, files in os.walk(search_path):
        for file in files:
            if file.endswith(".rst"):
                full_path = os.path.join(root, file)
                parsed_examples = parse_rst_file(full_path)
                dataset.extend(parsed_examples)
                
    output_file = "odoo_18_developer_dataset.jsonl"
    print(f"[*] Extracted {len(dataset)} instruction-response pairs.")
    print(f"[*] Writing dataset to '{output_file}'...")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for item in dataset:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"[+] Complete! Dataset generated containing {len(dataset)} examples.")

if __name__ == "__main__":
    src_dir = run_sparse_checkout()
    compile_dataset(src_dir)
