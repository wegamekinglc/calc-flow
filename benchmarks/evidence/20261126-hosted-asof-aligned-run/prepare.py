import shutil
import sys
from pathlib import Path

d = Path(__file__).resolve().parent
p = Path(sys.argv[1]) / "crates/calc-flow/examples"
shutil.copyfile(d / "stage_job.rs", p / "stage_job.rs")
shutil.copyfile(d / "project.json", p / "job-project.json")
