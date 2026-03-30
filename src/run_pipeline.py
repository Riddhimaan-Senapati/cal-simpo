"""
Master pipeline runner.

Usage examples:

  # Full pipeline (all steps, all methods):
  python run_pipeline.py

  # Only steps 1 and 2:
  python run_pipeline.py --steps 1,2

  # Only train one method:
  python run_pipeline.py --steps 5 --method dpo

  # Steps 5 through 7, all methods:
  python run_pipeline.py --steps 5,6,7
"""

import argparse
import subprocess
import sys
import os

SRC = os.path.dirname(os.path.abspath(__file__))

# Maps step key → (script filename, extra CLI args)
_STEP_MAP = {
    "1":          ("step1_reward_model.py",  []),
    "2":          ("step2_sft.py",           []),
    "3":          ("step3_pref_data.py",     []),
    "4":          ("step4_ref_logprobs.py",  []),
    "5_dpo":      ("step5_train.py",         ["--method", "dpo"]),
    "5_caldpo":   ("step5_train.py",         ["--method", "caldpo"]),
    "5_simpo":    ("step5_train.py",         ["--method", "simpo"]),
    "6":          ("step6_evaluate.py",      []),
    "7":          ("step7_plot.py",          []),
}

_ALL_STEPS = ["1", "2", "3", "4", "5_dpo", "5_caldpo", "5_simpo", "6", "7"]


def run(script, extra_args):
    cmd = [sys.executable, os.path.join(SRC, script)] + extra_args
    print(f"\n{'='*60}\n→ {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, cwd=SRC)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def resolve_steps(steps_str, method_filter):
    keys = []
    for s in steps_str.split(","):
        s = s.strip()
        if s == "5":
            if method_filter:
                keys.append(f"5_{method_filter}")
            else:
                keys.extend(["5_dpo", "5_caldpo", "5_simpo"])
        elif s in _STEP_MAP:
            keys.append(s)
        else:
            print(f"Unknown step '{s}'. Valid keys: {list(_STEP_MAP.keys())}")
            sys.exit(1)
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", default="all",
        help="Comma-separated step numbers / keys, or 'all'. "
             "Use '5' to run all three methods, or '--method' to pick one.",
    )
    parser.add_argument(
        "--method", default=None, choices=["dpo", "caldpo", "simpo"],
        help="When --steps includes 5, restrict to this method only.",
    )
    args = parser.parse_args()

    if args.steps == "all":
        step_keys = _ALL_STEPS
    else:
        step_keys = resolve_steps(args.steps, args.method)

    print(f"Running steps: {step_keys}")
    for key in step_keys:
        script, extra = _STEP_MAP[key]
        run(script, extra)

    print(f"\n{'='*60}\nAll requested steps completed successfully.\n{'='*60}")


if __name__ == "__main__":
    main()
