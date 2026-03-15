#!/usr/bin/env python3
"""
find_audit_csv.py
─────────────────
Diagnostic utility — shows exactly where sovereign_audit_assessor.py
will look for Sovereign_Audit_Master*.csv and what it finds.

Usage:
  python find_audit_csv.py
  python find_audit_csv.py /some/specific/path.csv
"""

import os
import sys
import glob

BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

DIVIDER = "─" * 60

def check_specific(path: str):
    print(f"\n{DIVIDER}")
    print(f"{BOLD}Checking specific path:{RESET} {path}")
    print(DIVIDER)
    abs_path = os.path.abspath(path)
    print(f"  Resolved absolute path : {abs_path}")
    if os.path.isfile(abs_path):
        size = os.path.getsize(abs_path)
        print(f"  {GREEN}✔  File exists{RESET}  ({size:,} bytes)")
        _peek_columns(abs_path)
    elif os.path.isdir(abs_path):
        print(f"  {RED}✖  That path is a directory, not a file{RESET}")
    else:
        print(f"  {RED}✖  File not found{RESET}")


DRIVE_PATTERN = (
    "/content/drive/MyDrive/judicial_results/"
    "Sovereign_Titan_v3.19.18_Entropy_Injection/Sov*.csv"
)


def check_auto_detect():
    print(f"\n{DIVIDER}")
    print(f"{BOLD}Auto-detect scan (mirrors sovereign_audit_assessor.py logic){RESET}")
    print(DIVIDER)

    cwd = os.getcwd()
    print(f"  Working directory : {cwd}")
    print(f"  Glob pattern 1    : **/Sovereign_Audit_Master*.csv  (recursive from cwd)")
    print(f"  Glob pattern 2    : {DRIVE_PATTERN}")
    print()

    # Mirror exact two-step logic from sovereign_audit_assessor.py
    candidates = sorted(
        glob.glob("**/Sovereign_Audit_Master*.csv", recursive=True),
        key=os.path.getmtime, reverse=True
    )
    source = "cwd"
    if not candidates:
        candidates = sorted(
            glob.glob(DRIVE_PATTERN),
            key=os.path.getmtime, reverse=True
        )
        source = "Google Drive"

    if not candidates:
        print(f"  {RED}✖  No matches found in either location{RESET}")
        print()
        print(f"  {YELLOW}Suggestions:{RESET}")
        print(f"    1. Make sure you have run sovereign_titan_v3_19_18.py at least once")
        print(f"       to generate a Sovereign_Audit_Master_<timestamp>.csv")
        print(f"    2. Check the factory's output directory and copy/move the CSV here:")
        print(f"       {cwd}")
        print(f"    3. Or pass the path explicitly:")
        print(f"       python sovereign_audit_assessor.py /path/to/your/file.csv")
    else:
        print(f"  {GREEN}✔  {len(candidates)} match(es) found  [source: {source}]{RESET}")
        print()
        for i, p in enumerate(candidates):
            abs_p  = os.path.abspath(p)
            size   = os.path.getsize(abs_p)
            mtime  = os.path.getmtime(abs_p)
            import datetime
            ts     = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            marker = f"{GREEN}← will use this one{RESET}" if i == 0 else ""
            print(f"  [{i+1}] {abs_p}")
            print(f"       Size: {size:,} bytes   Modified: {ts}  {marker}")
            if i == 0:
                _peek_columns(abs_p)
        print()

    # also scan common sibling locations
    _scan_common_locations(cwd)


def _peek_columns(path: str):
    """Read just the header row and validate expected columns."""
    required = {
        "Feature", "I_raw", "I_Norm", "UV%", "Max_R",
        "Is_Locked", "Iteration", "Brain", "Model_Type",
        "Timestamp", "Persistence", "A_Impact", "A_UV"
    }
    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=3)
        found    = set(df.columns)
        missing  = required - found
        extra    = found - required
        print()
        print(f"  {BOLD}Column check:{RESET}")
        print(f"    Columns in file  : {sorted(found)}")
        if missing:
            print(f"    {RED}Missing required : {sorted(missing)}{RESET}")
        else:
            print(f"    {GREEN}✔  All required columns present{RESET}")
        if extra:
            print(f"    Extra columns    : {sorted(extra)}")
        print(f"    Preview rows     : {len(df)}")
        if "Brain" in df.columns:
            print(f"    Brains in sample : {df['Brain'].unique().tolist()}")
        if "Iteration" in df.columns:
            print(f"    Iters in sample  : {df['Iteration'].unique().tolist()}")
    except ImportError:
        print(f"  {YELLOW}(pandas not available — skipping column check){RESET}")
    except Exception as e:
        print(f"  {RED}Error reading file: {e}{RESET}")


def _scan_common_locations(cwd: str):
    """Check common places the CSV might have been saved."""
    drive_dir = os.path.dirname(DRIVE_PATTERN)
    dirs_to_check = [
        cwd,
        os.path.join(cwd, "output"),
        os.path.join(cwd, "results"),
        os.path.join(cwd, "data"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        drive_dir,   # Colab Google Drive output folder
    ]
    print(f"  {BOLD}Scanning common locations:{RESET}")
    for loc in dirs_to_check:
        if not os.path.isdir(loc):
            continue
        # Use Sov*.csv for the Drive folder to match both naming conventions
        pattern = "Sov*.csv" if loc == drive_dir else "Sovereign_Audit_Master*.csv"
        hits = glob.glob(os.path.join(loc, pattern))
        if hits:
            for h in sorted(hits, key=os.path.getmtime, reverse=True):
                size = os.path.getsize(h)
                print(f"  {GREEN}✔  {h}  ({size:,} bytes){RESET}")
        else:
            print(f"  {YELLOW}—  {loc}  (no match){RESET}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_specific(sys.argv[1])
    else:
        check_auto_detect()
    print()
