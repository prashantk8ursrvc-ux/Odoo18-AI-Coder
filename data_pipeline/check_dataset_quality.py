# -*- coding: utf-8 -*-
"""
Dataset Quality Checker for Odoo SFT JSONL files.
Checks count, duplicates, formatting issues, and Sphinx leakages.
"""

import json
import os

def check_file(filename):
    if not os.path.exists(filename):
        print(f"[-] File {filename} not found.")
        return
        
    print(f"\n==================================================")
    print(f"Auditing Dataset: {filename}")
    print(f"==================================================")
    
    total = 0
    empty_instruction = 0
    empty_output = 0
    too_short_instruction = 0
    too_short_output = 0
    duplicate_outputs = set()
    total_duplicates = 0
    rst_leaks = 0
    
    lengths_inst = []
    lengths_out = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            total += 1
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"[!] Row {idx+1}: Failed to parse JSON: {e}")
                continue
                
            inst = data.get("instruction", "").strip()
            out = data.get("output", "").strip()
            
            # Check length
            lengths_inst.append(len(inst))
            lengths_out.append(len(out))
            
            if not inst:
                empty_instruction += 1
            elif len(inst) < 15:
                too_short_instruction += 1
                
            if not out:
                empty_output += 1
            elif len(out) < 10:
                too_short_output += 1
                
            # Check duplicates
            if out in duplicate_outputs:
                total_duplicates += 1
            else:
                duplicate_outputs.add(out)
                
            # Check for RST direct leaks in instruction/output
            # direct lists of sphinx keywords
            leak_keywords = [
                ".. toctree::", ".. image::", ".. figure::",
                ".. note::", ".. warning::", ".. list-table::",
                "========", "--------"
            ]
            for kw in leak_keywords:
                if kw in inst or kw in out:
                    rst_leaks += 1
                    break
                    
    avg_inst_len = sum(lengths_inst) / total if total > 0 else 0
    avg_out_len = sum(lengths_out) / total if total > 0 else 0
    
    print(f"[*] Total Examples:            {total}")
    print(f"[*] Unique Code/Content blocks: {len(duplicate_outputs)}")
    print(f"[*] Duplicate Blocks:           {total_duplicates} ({total_duplicates/total*100:.1f}%)")
    print(f"[-] Empty Instructions:         {empty_instruction}")
    print(f"[-] Too Short Instructions:     {too_short_instruction}")
    print(f"[-] Empty Outputs:              {empty_output}")
    print(f"[-] Too Short Outputs:          {too_short_output}")
    print(f"[-] Leaked Sphinx Directives:   {rst_leaks}")
    print(f"[*] Avg Instruction Length:     {avg_inst_len:.1f} chars")
    print(f"[*] Avg Output Length:          {avg_out_len:.1f} chars")
    
    # Print a few samples of short instruction to inspect quality
    print("\n--- SAMPLE SHORT INSTRUCTIONS (POSSIBLE QUALITY ISSUE) ---")
    count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            inst = data.get("instruction", "").strip()
            if len(inst) < 120 and "In Odoo 18" in inst and count < 3:
                print(f"Row: {inst[:120]}...")
                count += 1

if __name__ == "__main__":
    check_file("odoo_18_developer_dataset.jsonl")
    check_file("odoo_18_planner_dataset.jsonl")
