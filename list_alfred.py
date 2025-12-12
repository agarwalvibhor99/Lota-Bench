#!/usr/bin/env python
"""
List ALFRED action skills used by the planner.

It first tries to read from a resource file to avoid loading the LLM.
If it can't find one, it falls back to AlfredTaskPlanner (which will
load the model specified in conf/config_alfred.yaml).
"""

import os
import sys
import json

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------
# 1. Try to get skills from a resource file (fast path)
# --------------------------------------------------------------------

CANDIDATE_SKILL_FILES = [
    os.path.join(REPO_ROOT, "resource", "alfred_skill_set.json"),
    os.path.join(REPO_ROOT, "resource", "alfred_skill_set.txt"),
]

skills = None

for path in CANDIDATE_SKILL_FILES:
    if os.path.isfile(path):
        print(f"✅ Found skill file: {path}")
        if path.endswith(".json"):
            with open(path, "r") as f:
                skills = json.load(f)
        else:
            with open(path, "r") as f:
                skills = [line.strip() for line in f if line.strip()]
        break

# --------------------------------------------------------------------
# 2. Fallback: use AlfredTaskPlanner (may load LLM)
# --------------------------------------------------------------------

if skills is None:
    print("No simple skill file found – falling back to AlfredTaskPlanner.")
    sys.path.append(os.path.join(REPO_ROOT, "src"))

    from omegaconf import OmegaConf
    from alfred.alfred_task_planner import AlfredTaskPlanner

    cfg_path = os.path.join(REPO_ROOT, "conf", "config_alfred.yaml")
    if not os.path.isfile(cfg_path):
        print("ERROR: Config file not found:", cfg_path)
        sys.exit(1)

    cfg = OmegaConf.load(cfg_path)
    print("Loading AlfredTaskPlanner (this may take a while)...")
    planner = AlfredTaskPlanner(cfg)
    skills = list(planner.skill_set)

# --------------------------------------------------------------------
# 3. Print skills
# --------------------------------------------------------------------

print("\n=== ALFRED skill / action list ===")
print("Total skills:", len(skills))
for s in skills:
    print(" -", repr(s))
