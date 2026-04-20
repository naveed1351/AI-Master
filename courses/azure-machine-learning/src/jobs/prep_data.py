from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from sklearn.model_selection import train_test_split

from azureml_course.dataset_utils import load_iris_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_iris_frame()
    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame["species"],
    )
    train_frame.to_csv(output_dir / "train.csv", index=False)
    test_frame.to_csv(output_dir / "test.csv", index=False)
    print(f"Prepared data in {output_dir}")


if __name__ == "__main__":
    main()