#!/usr/bin/env python3
# run_and_collect.py
"""
Improved runner:
- Runs pipeline.py with --out -> /mnt/data/pipeline_result_<timestamp>.json
- Runs grid_search.py
- Captures stdout/stderr per command in /mnt/data/logs/
- Writes a combined human-readable log: /mnt/data/action_results.txt
- Copies grid_results.json and pipeline_result to /mnt/data/ if produced
- Writes summary JSON: /mnt/data/action_summary.json
"""

import subprocess
import sys
import shutil
import json
import traceback
import os
from pathlib import Path
from datetime import datetime

# Settings
PROJ_DIR = Path("")
OUT_DIR = Path("Result")
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp helpers
ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
pipeline_out_name = f"pipeline_result_{ts}.json"
pipeline_out_path = OUT_DIR / pipeline_out_name

# Commands to run
commands = [
    {
        "title": "Run pipeline.py",
        "cmd": [sys.executable, "pipeline.py", "--out", str(pipeline_out_path)],
        "cwd": PROJ_DIR
    },
    {
        "title": "Run grid_search.py",
        "cmd": [sys.executable, "grid_search.py"],
        "cwd": PROJ_DIR
    }
]

# Containers for combined log
combined_lines = []
combined_lines.append(f"Run started: {datetime.utcnow().isoformat()}Z")
combined_lines.append(f"Project dir: {PROJ_DIR}")
combined_lines.append("")

# Per-command run
timeout_seconds = 1800  # 30 minutes; adjust if you need more
results = []

for step in commands:
    title = step["title"]
    cmd = step["cmd"]
    cwd = step["cwd"]
    combined_lines.append("=" * 80)
    combined_lines.append(f"COMMAND: {title}")
    combined_lines.append(f"CMD LINE: {' '.join(map(str, cmd))}")
    combined_lines.append(f"CWD: {cwd}")
    combined_lines.append(f"START: {datetime.utcnow().isoformat()}Z")
    stdout_file = LOG_DIR / f"{title.replace(' ', '_')}_{ts}.stdout.txt"
    stderr_file = LOG_DIR / f"{title.replace(' ', '_')}_{ts}.stderr.txt"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

        # Save stdout/stderr to files
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        stdout_file.write_text(stdout_text)
        stderr_file.write_text(stderr_text)

        # Append to combined log (short summary)
        combined_lines.append(f"--- STDOUT (first 400 chars) ---\n{stdout_text[:400]}")
        combined_lines.append(f"--- STDERR (first 400 chars) ---\n{stderr_text[:400]}")
        combined_lines.append(f"STDOUT saved to: {stdout_file}")
        combined_lines.append(f"STDERR saved to: {stderr_file}")
        combined_lines.append(f"EXIT CODE: {proc.returncode}")

        results.append({
            "title": title,
            "cmd": cmd,
            "exit_code": proc.returncode,
            "stdout_file": str(stdout_file),
            "stderr_file": str(stderr_file)
        })

    except subprocess.TimeoutExpired as e:
        combined_lines.append("--- ERROR: TIMEOUT ---")
        combined_lines.append(f"Process timed out after {timeout_seconds} seconds.")
        combined_lines.append(str(e))
        combined_lines.append(f"STDOUT partial: {e.stdout[:400] if e.stdout else '<none>'}")
        combined_lines.append(f"STDERR partial: {e.stderr[:400] if e.stderr else '<none>'}")
        results.append({
            "title": title,
            "cmd": cmd,
            "error": "timeout",
            "stdout_file": str(stdout_file),
            "stderr_file": str(stderr_file)
        })
    except Exception:
        combined_lines.append("--- EXCEPTION ---")
        combined_lines.append(traceback.format_exc())
        results.append({
            "title": title,
            "cmd": cmd,
            "error": "exception",
            "traceback": traceback.format_exc()
        })

    combined_lines.append(f"END: {datetime.utcnow().isoformat()}Z")
    combined_lines.append("")

# After runs: copy important result files (if they exist)
copied_results = {}

# pipeline result
if pipeline_out_path.exists():
    copied_results["pipeline_result"] = str(pipeline_out_path)
else:
    # try find any pipeline_result_*.json under project
    found = list(PROJ_DIR.glob("pipeline_result_*.json"))
    if found:
        # copy the newest one to /mnt/data/
        newest = max(found, key=lambda p: p.stat().st_mtime)
        dest = OUT_DIR / f"{newest.name}"
        shutil.copy(newest, dest)
        copied_results["pipeline_result"] = str(dest)
    else:
        copied_results["pipeline_result"] = None

# grid results
grid_src = PROJ_DIR / "grid_results.json"
if grid_src.exists():
    dest = OUT_DIR / "grid_results.json"
    shutil.copy(grid_src, dest)
    copied_results["grid_results"] = str(dest)
else:
    copied_results["grid_results"] = None

# Save combined human-readable log
combined_log_path = OUT_DIR / "action_results.txt"
combined_lines.append("=" * 80)
combined_lines.append("Copied results summary:")
combined_lines.append(json.dumps(copied_results, indent=2))
combined_lines.append("")
combined_log_path.write_text("\n".join(combined_lines))

# Save a machine-readable summary JSON
summary = {
    "timestamp": ts,
    "project_dir": str(PROJ_DIR),
    "logs_dir": str(LOG_DIR),
    "combined_log": str(combined_log_path),
    "per_command": results,
    "copied_results": copied_results
}
summary_path = OUT_DIR / "action_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

# Print final status
print("Done.")
print("Combined log:", combined_log_path)
print("Summary JSON:", summary_path)
print("Per-command logs stored in:", LOG_DIR)
if copied_results.get("pipeline_result"):
    print("Pipeline result:", copied_results["pipeline_result"])
if copied_results.get("grid_results"):
    print("Grid results:", copied_results["grid_results"])
