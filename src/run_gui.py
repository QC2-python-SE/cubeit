import subprocess
import time
import sys
from pathlib import Path

here = Path(__file__).parent.resolve()

nb_path = here / "gui_notebook.ipynb"

cmd = [sys.executable, "-m", "voila", str(nb_path), "--MappingKernelManager.cull_idle_timeout=1", "--MappingKernelManager.cull_interval=1", "--MappingKernelManager.cull_connected=False"]

subprocess.run(cmd)