import subprocess
import os, sys
import shutil
import importlib.metadata
from typing import Any
from tqdm import tqdm
import urllib.request
from packaging import version as pv

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(BASE_PATH, "requirements.txt")
models_dir = os.path.join(models_path, "insightface")

model_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)

def pip_install(*args):
    if shutil.which("uv"):
        subprocess.run(["uv", "pip", "install", "--python", sys.executable, *args])
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", *args])

def pip_uninstall(*args):
    if shutil.which("uv"):
        subprocess.run(["uv", "pip", "uninstall", "--python", sys.executable, *args])
    else:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *args])

def is_installed(package: str, version: str | None = None, strict: bool = True):
    clean_package = package.split("==")[0].split(">=")[0].split("<=")[0].strip()
    try:
        installed_version = importlib.metadata.version(clean_package)
        if version is None:
            return True
        if strict:
            return installed_version == version
        else:
            return pv.parse(installed_version) >= pv.parse(version)
    except importlib.metadata.PackageNotFoundError:
        return False

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading...', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(model_path):
    download(model_url, model_path)

last_device = None
first_run = False
available_devices = ["CPU", "CUDA"]

try:
    last_device_log = os.path.join(BASE_PATH, "last_device.txt")
    with open(last_device_log) as f:
        last_device = f.readline().strip()
    if last_device not in available_devices:
        last_device = None
except:
    last_device = "CPU"
    first_run = True
    with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
        txt.write(last_device)

with open(req_file) as file:
    install_count = 0
    ort = "onnxruntime-gpu"
    import torch
    cuda_version = None
    try:
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"CUDA {cuda_version}")
            if first_run or last_device is None:
                last_device = "CUDA"
        elif torch.backends.mps.is_available() or hasattr(torch,'dml') or hasattr(torch,'privateuseone'):
            ort = "onnxruntime"
            if first_run:
                pip_uninstall("onnxruntime", "onnxruntime-gpu")
            if last_device == "CUDA" or last_device is None:
                last_device = "CPU"
        else:
            if last_device == "CUDA" or last_device is None:
                last_device = "CPU"
        
        with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
            txt.write(last_device)

        target_ort_version = "1.20.0"
        if cuda_version is not None:
            if not is_installed(ort, target_ort_version, False):
                install_count += 1
                pip_uninstall("onnxruntime", "onnxruntime-gpu")
                pip_install(f"{ort}=={target_ort_version}")
        elif not is_installed(ort, target_ort_version, False):
            install_count += 1
            pip_install(ort, "-U")
    except Exception as e:
        print(e)
        print(f"\nERROR: Failed to install {ort} - ReActor won't start")
        raise e

    strict = True
    for package in file:
        package_version = None
        try:
            package = package.strip()
            if not package or package.startswith("#"):
                continue
            if "==" in package:
                package_version = package.split('==')[1]
            elif ">=" in package:
                package_version = package.split('>=')[1]
                strict = False
            
            clean_pkg_name = package.split("==")[0].split(">=")[0].strip()
            if not is_installed(clean_pkg_name, package_version, strict):
                install_count += 1
                pip_install(package)
        except Exception as e:
            print(e)
            print(f"\nERROR: Failed to install {package} - ReActor won't start")
            raise e

    if install_count > 0:
        print(f"""
        +---------------------------------+
        --- PLEASE, RESTART the Server! ---
        +---------------------------------+
        """)