"""批量运行所有 baseline 方法 + 覆盖率评估"""
import subprocess
import sys
import os

SCRIPTS = [
    "0_sogn.py",
    "1_standard.py",
    "2_mc_dropout.py",
    "3_deep_ensemble.py",
    "4_deep_evidential.py",
    "5_selectivenet.py",
    "6_conformal.py",
]

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    for script in SCRIPTS:
        path = os.path.join(base_dir, script)
        print(f"\n{'#'*60}\n# Running {script}\n{'#'*60}")
        result = subprocess.run([sys.executable, path], cwd=base_dir)
        if result.returncode != 0:
            print(f"ERROR: {script} failed with code {result.returncode}")
        else:
            print(f"DONE: {script}")

    print(f"\n{'#'*60}\n# Running coverage evaluation\n{'#'*60}")
    eval_path = os.path.join(base_dir, "7_coverage_eval.py")
    subprocess.run([sys.executable, eval_path], cwd=base_dir)
    print("\nAll done!")
