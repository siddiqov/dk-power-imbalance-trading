"""
Phase 0 helper: run this with the project venv Python 3.14 to extract
function signatures from live.pyc and pipeline.pyc.

Usage (from Basic_Approach folder):
  Nurex_V4_2\.venv\Scripts\python.exe _decompile_phase0.py
"""
import dis, marshal, struct, sys, os, textwrap
from pathlib import Path

BASE = Path(__file__).parent

def load_code(pyc_path):
    with open(pyc_path, 'rb') as f:
        f.read(4)   # magic
        f.read(4)   # flags
        f.read(8)   # mtime/hash + size
        return marshal.loads(f.read())

def extract_api(code, indent=0):
    lines = []
    pad = "    " * indent
    for c in code.co_consts:
        if not hasattr(c, 'co_name'):
            continue
        if c.co_name == '<module>':
            lines += extract_api(c, indent)
            continue
        args = list(c.co_varnames[:c.co_argcount])
        sig = ", ".join(args)
        lines.append(f"{pad}def {c.co_name}({sig}):")
        # look for docstring
        if c.co_consts and isinstance(c.co_consts[0], str):
            doc = c.co_consts[0][:200].replace('\n',' ')
            lines.append(f'{pad}    """{doc}"""')
        lines.append(f"{pad}    ...")
        # recurse into nested
        nested = extract_api(c, indent+1)
        if nested:
            lines[-1] = f"{pad}    # nested:"
            lines += nested
    return lines

for name in ['live', 'pipeline']:
    src_path = BASE / 'Nurex_V4_2' / 'nurex42' / f'{name}.pyc'
    if not src_path.exists():
        # try __pycache__
        src_path = BASE / 'Nurex_V4_2' / 'nurex42' / '__pycache__' / f'{name}.cpython-314.pyc'
    
    out_path = BASE / 'Nurex_V4_2' / 'nurex42' / f'{name}.py'
    
    print(f"\n=== {name}.pyc -> {name}.py ===")
    try:
        code = load_code(src_path)
        api_lines = extract_api(code)
        
        stub = [
            f'"""',
            f'Auto-extracted stubs from {name}.pyc (Python 3.14 bytecode).',
            f'Extracted by _decompile_phase0.py on {sys.version}.',
            f'WARNING: These are API stubs only — implementation body is missing.',
            f'Replace with actual source from the original developer.',
            f'"""',
            '',
        ]
        stub += api_lines
        
        with open(out_path, 'w') as f:
            f.write('\n'.join(stub) + '\n')
        
        print(f"Written: {out_path}")
        print("Functions found:")
        for l in api_lines:
            if l.strip().startswith('def '):
                print("  " + l.strip())
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()

print("\nDone. Now commit the .py files to git.")
