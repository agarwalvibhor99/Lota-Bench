#!/usr/bin/env python
"""
Debug script for ALFRED dataset in LLMTaskPlanning.

Checks:
- whether answer trajectories exist
- data splits and number of trajectories per split
- task types and counts per split
- prints one sample trajectory (instruction + high-level plan)
"""

import os
import sys
import glob
import json
import random
import collections

# --------------------------------------------------------------------
# 1. Locate ALFRED data root (adapted to YOUR repo layout)
# --------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_ROOTS = [
    os.path.join(REPO_ROOT, "dataset", "alfred", "data", "json_2.1.0"),
    os.path.join(REPO_ROOT, "alfred", "data", "json_2.1.0"),
]

DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.isdir(r):
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    print("ERROR: Could not find ALFRED data directory.")
    print("Tried:")
    for r in CANDIDATE_ROOTS:
        print("  -", r)
    sys.exit(1)

print("✅ Using ALFRED data root:", DATA_ROOT)

# --------------------------------------------------------------------
# 2. Check data splits + number of trajectories
# --------------------------------------------------------------------

SPLITS = ["train", "valid_seen", "valid_unseen", "tests_seen", "tests_unseen"]

print("\n=== 1. Number of trajectories per split ===")
split_traj_paths = {}  # store for later reuse

for split in SPLITS:
    split_dir = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"{split:12s} -> MISSING (no directory at {split_dir})")
        continue

    traj_files = glob.glob(os.path.join(split_dir, "*", "traj_data.json"))
    split_traj_paths[split] = traj_files
    print(f"{split:12s} -> {len(traj_files):5d} traj_data.json files")

if "valid_seen" not in split_traj_paths or not split_traj_paths["valid_seen"]:
    print("\nWARNING: valid_seen has no trajectories – evaluation will be empty.")
else:
    print("\n✅ valid_seen has trajectories, evaluation subset can be sampled.")

# --------------------------------------------------------------------
# 3. Check task types and counts per split
# --------------------------------------------------------------------

print("\n=== 2. Task type distribution per split ===")

for split, traj_files in split_traj_paths.items():
    if not traj_files:
        continue

    counter = collections.Counter()
    for p in traj_files:
        with open(p, "r") as f:
            d = json.load(f)
        counter[d.get("task_type", "UNKNOWN")] += 1

    total = sum(counter.values())
    print(f"\n--- {split} ---")
    print(f"Total trajectories: {total}")
    for t, c in counter.most_common():
        print(f"  {t:35s} {c:5d}")

# --------------------------------------------------------------------
# 4. Sample trajectory: instruction + answer plan
# --------------------------------------------------------------------

print("\n=== 3. Sample trajectory from valid_seen (answer trajectory check) ===")

if "valid_seen" not in split_traj_paths or not split_traj_paths["valid_seen"]:
    print("No valid_seen trajectories found – skipping sample print.")
    sys.exit(0)

sample_path = random.choice(split_traj_paths["valid_seen"])
print("Sample file:", sample_path)

with open(sample_path, "r") as f:
    d = json.load(f)

print("\nTop-level keys:", list(d.keys()))

# Natural language instruction
ann = d["turk_annotations"]["anns"][0]
high_desc = ann["high_descs"][0]
print("\nInstruction:")
print(" ", high_desc)

# High-level plan (this is the 'answer' trajectory)
print("\nHigh-level PDDL plan:")
for step in d["plan"]["high_pddl"]:
    print(" -", step)

print("\nTask type:", d["task_type"])
print("\n✅ Answer trajectory exists and is readable.")
