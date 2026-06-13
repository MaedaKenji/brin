# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import os
import sys
import subprocess


def run_training(project_dir: Path, run_name: str):
    # Actual training path; called in the child process
    model = YOLO("yolo26n.yaml")  # build a new model from YAML
    model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo26n.yaml").load(
        "yolo26n.pt"
    )  # build from YAML and transfer weights

    model.train(
        # data=r"D:\Code\Python\brin\surrounding-awareness-5\data.yaml", #HARUS GANTI
        data=r"D:\Code\Python\brin\human action - coco.v1i.yolo26\data.yaml",
        epochs=200,
        imgsz=640,
        device=0,
        batch=8,
        workers=2,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        patience=20,
    )


def main():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir / "output"
    project_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = project_dir / "output.txt"

    if os.environ.get("YOLO_CAPTURED") == "1":
        run_training(project_dir, run_name)
        return

    existing_dirs = {p.name for p in project_dir.iterdir() if p.is_dir()}
    env = os.environ.copy()
    env["YOLO_CAPTURED"] = "1"

    # ensure stale output.txt is removed before writing
    if output_file.exists():
        output_file.unlink()

    with open(output_file, "w", encoding="utf-8", errors="replace") as f:
        subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve())],
            cwd=str(script_dir),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    # Identify YOLO-created run folder as any new directory in output/
    current_dirs = {p.name for p in project_dir.iterdir() if p.is_dir()}
    new_dirs = sorted(current_dirs - existing_dirs)

    if new_dirs:
        actual_run_dir = project_dir / new_dirs[-1]
    else:
        # Fallback: choose latest modified output run directory
        dirs = sorted(
            [p for p in project_dir.iterdir() if p.is_dir()],
            key=lambda x: x.stat().st_mtime,
        )
        actual_run_dir = dirs[-1] if dirs else project_dir / run_name
        if not actual_run_dir.exists():
            actual_run_dir.mkdir(parents=True, exist_ok=True)

    final_output_file = actual_run_dir / "output.txt"
    if final_output_file.exists():
        final_output_file.unlink()
    output_file.replace(final_output_file)

    print(f"Training finished, logs and weights saved in: {actual_run_dir}")


if __name__ == "__main__":
    main()
