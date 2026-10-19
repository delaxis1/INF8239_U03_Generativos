import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "notebooks/06_ae_vae.ipynb"
nb = nbformat.read(p, as_version=4)
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [
    sys.executable,
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}",
]
try:
    NotebookClient(
        nb, km=km, timeout=7200, resources={"metadata": {"path": str(ROOT)}}
    ).execute()
finally:
    nbformat.write(nb, p)
print("Notebook completo y guardado:", p)
