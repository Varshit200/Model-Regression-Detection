
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(description, command):
    print(f"\n{'=' * 55}", flush=True)
    print(description, flush=True)
    print("=" * 55, flush=True)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    python = sys.executable

    try:
        run_step(
            "STEP 1: Run automated regression tests",
            [python, "-m", "unittest", "tests.test_regression", "-v"],
        )

        run_step(
            "STEP 2: Evaluate the golden dataset",
            [python, "-m", "evaluation.run_evaluation"],
        )

        run_step(
            "STEP 3: Compare candidate against baseline",
            [python, "-m", "evaluation.compare_results"],
        )

        print("\nPipeline completed successfully.")

    except subprocess.CalledProcessError as error:
        print(f"\nPipeline failed with exit code {error.returncode}.")
        sys.exit(error.returncode)


if __name__ == "__main__":
    main()
