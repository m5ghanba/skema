# SKeMa
[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F6790-blue)](https://doi.org/10.57967/hf/6790)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SKeMa-yellow)](https://huggingface.co/m5ghanba/SKeMa)
[![PyPI](https://img.shields.io/pypi/v/skema-kelp)](https://pypi.org/project/skema-kelp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model License: CC BY 4.0](https://img.shields.io/badge/Model%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Satellite-based Kelp Mapping using Semantic Segmentation on Sentinel-2 imagery**

`skema` is a Python tool for classifying kelp in Sentinel-2 satellite images using a deep learning segmentation model (PyTorch). It provides a command-line interface (CLI) for easy, reproducible inference. To run the tool you would need to download Sentinel-2 images from the Copernicus Browser. More detailed instruction on how to download these images can be found in Section Usage. The following instructions are provided for anyone with no knowledge of what a command line is, no knowledge of Python or virtual environments, etc. Just follow along step by step.

**Model available on Hugging Face**: [m5ghanba/SKeMa](https://huggingface.co/m5ghanba/SKeMa)  
**DOI**: [10.57967/hf/6790](https://doi.org/10.57967/hf/6790)

---

## ⚡ Quick Start (Experienced Users)
```bash
pip install skema-kelp

# Download static files (for model_full only) from sources listed below
# Download Sentinel-2 imagery from https://dataspace.copernicus.eu/browser/

skema --input-dir path/to/S2_scene.SAFE --output-filename output.tif

# For help and all options
skema --help
```

**For detailed installation instructions (beginner-friendly), see [Installation](#-installation) below.**

---

## Citation

If you use **SKeMa** in your research or work, please cite:

```bibtex
@software{skema_2025,
  author       = {Mohsen Ghanbari et al.},
  title        = {SKeMa: Satellite-based Kelp Mapping using Semantic Segmentation on Sentinel-2 imagery},
  year         = 2025,
  publisher    = {Hugging Face},
  doi          = {10.57967/hf/6790},
  url          = {https://huggingface.co/m5ghanba/SKeMa}
}
```

**Plain text**:  
Ghanbari, M., et al. (2025). *SKeMa: Satellite-based Kelp Mapping using Semantic Segmentation on Sentinel-2 imagery*. Hugging Face. https://doi.org/10.57967/hf/6790

---

## 🚀 Installation

Before you can set up SKeMa, you'll need **Python** (version 3.8 or higher) installed on your computer. Python is a free tool, and no accounts or sign-ups are required to install it. We'll install it using your terminal (command line) where possible for simplicity. If you're on Windows, ensure you're using **PowerShell** or **Command Prompt** as Administrator (right-click and select "Run as administrator") for some steps.

### Step 1: Install Python

#### On Windows
1. **Check if Winget is available** (it's built into Windows 10 version 2009 or later, or Windows 11, and most modern systems have it):
   - Open PowerShell or Command Prompt.
   - Type `winget --version` and press Enter.
   - If it shows a version number (e.g., "v1.8.0"), proceed. If not (error like "winget is not recognized"), download the App Installer from the Microsoft Store (search for "App Installer") or update Windows via Settings > Update & Security > Windows Update.

2. **Install Python 3.12** (the latest stable version as of October 2025; this meets the >=3.8 requirement):
   - In your terminal, run:
     ```
     winget install -e --id Python.Python.3.12
     ```
   - This downloads and installs Python automatically. It may take a few minutes.
   - **Important**: During installation (if prompted), ensure "Add Python to PATH" is selected (it usually is by default with winget).
   - Restart your terminal after installation.
   - Verify: Run `python --version`. It should output something like "Python 3.12.7". If not, close and reopen the terminal, or manually add Python to PATH (search online for "add Python to PATH Windows").

   *Alternative if winget fails*: Download the installer from [python.org](https://www.python.org/downloads/windows/) using your browser, run it, and follow the GUI prompts (make sure to check "Add Python to PATH"). Then verify as above.

#### On macOS
1. **Install Homebrew** (a package manager for CLI installations, if you don't have it):
   - Open Terminal (search for it in Spotlight with Cmd+Space).
   - Run:
     ```
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
   - Follow any on-screen prompts (it may ask for your password; this is normal). No account needed.
   - After installation, run the commands it suggests to add Homebrew to your PATH (e.g., `echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile` and then `eval "$(/opt/homebrew/bin/brew shellenv)"`).
   - Verify: Run `brew --version`. It should show a version like "4.3.0".

2. **Install Python 3.12**:
   - Run:
     ```
     brew install python@3.12
     ```
   - This installs Python and adds it to PATH.
   - Verify: Run `python3 --version` (note: use `python3` on macOS). It should output "Python 3.12.7".

   *Alternative*: Download the official installer from [python.org](https://www.python.org/downloads/macos/) using your browser, run it, and follow GUI steps.

#### On Linux (e.g., Ubuntu/Debian; adjust for other distros like Fedora)
1. **Update your package list**:
   - Open your terminal.
   - Run:
     ```
     sudo apt update
     ```
   - Enter your password when prompted (sudo is for admin privileges; no account needed beyond your user login).

2. **Install Python 3.12**:
   - Run:
     ```
     sudo apt install python3.12 python3.12-venv python3-pip
     ```
   - This installs Python, the venv module, and pip.
   - Verify: Run `python3 --version`. It should output "Python 3.12.x".
   - *For Fedora/RHEL*: Use `sudo dnf install python3.12` instead.

   *Note*: If your distro's repositories don't have Python 3.12, add a PPA (e.g., for Ubuntu: `sudo add-apt-repository ppa:deadsnakes/ppa` then update and install).

Once Python is installed and verified, proceed to the next section. If you encounter errors (e.g., "command not found"), search online for the exact error message + your OS.

### Step 2: Install SKeMa

Open your **terminal**:  
- On **Windows**, you can use Command Prompt or PowerShell.  
- On **macOS**, open the Terminal app.  
- On **Linux**, open your terminal emulator of choice.  

When you open a terminal, you start inside a **directory (folder)**. You can move to another directory with the command `cd`. For example:  

```  
cd C:\Users\YourName\Documents  
```  

On macOS/Linux:  

```  
cd /Users/yourname/Documents  
```  

👉 The easiest way to navigate is to open your file explorer, go to the folder you want, then copy its full path and paste it after `cd` on the command line. For more details, look up "basic terminal navigation" online.  

Now, navigate to a directory where you want to work with SKeMa, then run:

#### Option 1: Install with pip (Recommended)

```bash
# Create a virtual environment (optional but recommended)
python -m venv skema_env

# Activate the virtual environment
# On Windows:
skema_env\Scripts\activate
# On macOS/Linux:
source skema_env/bin/activate

# Install SKeMa
pip install skema-kelp
```

#### Option 2: Install from source (for developers)

If you want to modify the code or contribute to development:

```bash
# Install Git first (see system-specific instructions below)
# Then clone the repository
git clone https://github.com/m5ghanba/skema.git
cd skema

# Create and activate virtual environment
python -m venv skema_env
# On Windows:
skema_env\Scripts\activate
# On macOS/Linux:
source skema_env/bin/activate

# Install in development mode
pip install -e .
```

**Installing Git (only needed for Option 2):**
- **Windows**: `winget install --id Git.Git -e --source winget` or download from [git-scm.com](https://git-scm.com/download/win)
- **macOS**: `brew install git` or download from [git-scm.com](https://git-scm.com/download/mac)
- **Linux**: `sudo apt install git` (Ubuntu/Debian) or `sudo dnf install git` (Fedora/RHEL)

Each line explained:  
- `python -m venv skema_env`: Creates a virtual environment named `skema_env` to isolate project dependencies.  
- `skema_env\Scripts\activate` (Windows) or `source skema_env/bin/activate` (macOS/Linux): Activates the virtual environment, ensuring subsequent commands use its isolated Python and packages.  
- `pip install skema-kelp`: Installs SKeMa and all its dependencies from PyPI.

If you encounter packaging errors, make sure your pip and build tools are up to date:

```bash
pip install --upgrade pip setuptools wheel
```

#### Static files  
There are necessary **static files** that need to be downloaded. These are bathymetry and substrate files from the whole coast of British Columbia that `skema` uses when predicting kelp on a Sentinel-2 image.  

- The bathymetry file is a single TIFF raster (`Bathymetry_10m.tif`).  
- There are five substrate TIFF rasters (`NCC_substrate_20m.tif`, `SOG_substrate_20m.tif`, `WCVI_substrate_20m.tif`, `QCS_substrate_20m.tif`, `HG_substrate_20m.tif`), each covering a different region of the BC coast.  
- Place them inside:  

```text  
skema/skema/static/bathy_substrate/  
```

**⚠️ Note**: Static files (bathymetry and substrate) are only required when using the **full model** (`--model-type model_full`). If you plan to use only the **S2-only model** (`--model-type model_s2bandsandindices_only`), you can skip downloading these files.

**Sources**:  
- Canada's DEM/bathymetry model (10m resolution):  
  - Documentation: https://publications.gc.ca/collections/collection_2023/rncan-nrcan/m183-2/M183-2-8963-eng.pdf  
  - Dataset: https://maps-cartes.services.geo.ca/server_serveur/rest/services/NRCan/canada_west_coast_DEM_en/MapServer  

- Shallow substrate model (20m) of the Pacific Canadian coast (Haggarty et al., 2020):  
  https://osdp-psdo.canada.ca/dp/en/search/metadata/NRCAN-FGP-1-b100cf6c-7818-4748-9960-9eab2aa6a7a0  

#### GPU support  

For GPU users, install CUDA-supported PyTorch that matches your CUDA Toolkit. Check your CUDA version with:  

```bash  
nvcc --version  
```

For CUDA 12.1:  

```bash  
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121  
```

For CUDA 11.8:  

```bash  
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118  
```

Skip this step if you don't have a GPU.

---

## 🛰️ Usage

To use SKeMa after the initial installation, you must activate its virtual environment each time you start a new session (if you created one). Navigate to the directory where you created the virtual environment and activate it. If your command line prompt shows `(skema_env)`, the virtual environment is activated.

**On Windows:**
```
cd path\to\your\directory
skema_env\Scripts\activate
```

**On macOS/Linux:**
```
cd path/to/your/directory
source skema_env/bin/activate
```

This will activate the skema_env virtual environment, where SKeMa and its dependencies are installed, ensuring the tool runs correctly.

SKeMa uses Sentinel-2 satellite images, which can be downloaded from the [Copernicus Browser](https://dataspace.copernicus.eu/browser/). To access these images, you need to create a free account on the Copernicus Data Space website:
- Visit [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/) and click "Sign Up" to create an account.
- Follow the instructions to register with your email and set a password.
- Once logged in, use the Copernicus Browser to search for and download Sentinel-2 images, which will be provided as `.zip` files.

Now, you can run SKeMa on a new Sentinel-2 image:

```bash  
skema --input-dir path/to/sentinel2/safe/folder --output-filename output.tif  
```

- The first path (`--input-dir`) must be the full path to the `.SAFE` folder.  
  - Sentinel-2 images from the Copernicus Browser come as `.zip` files. Extract them first.  
  - Then, pass the full path to the `.SAFE` folder (e.g., `C:\...\S2C_MSIL2A_20250715T194921_N0511_R085_T09UUU_20250716T001356.SAFE`).  

- The second parameter (`--output-filename`) is the name of the output file (e.g., `output.tif`).  

### Model Types

SKeMa supports two model types:

1. **`model_full`** (default): Uses all available data including Sentinel-2 bands, bathymetry, and substrate information. This model provides the most accurate predictions but requires bathymetry and substrate static files.

2. **`model_s2bandsandindices_only`**: Uses only Sentinel-2 bands and derived spectral indices. This model does not require bathymetry or substrate files, making it suitable for areas outside British Columbia or when static files are unavailable.

To specify the model type, use the `--model-type` flag:

```bash
# Using the full model (default - includes bathymetry and substrate)
skema --input-dir path/to/sentinel2/safe/folder --output-filename output.tif --model-type model_full

# Using S2-only model (no bathymetry/substrate required)
skema --input-dir path/to/sentinel2/safe/folder --output-filename output.tif --model-type model_s2bandsandindices_only
```

If `--model-type` is not specified, the tool defaults to `model_full`.

### Output Files

After running, the tool generates a folder with the same name as the `.SAFE` file. Inside this folder, you'll find:

**For `model_full`:**
1. **`<SAFE_name>_B2B3B4B8.tif`**: a 10 m resolution, 4-band GeoTIFF containing Sentinel-2 bands B02 (Blue), B03 (Green), B04 (Red), and B08 (Near-Infrared).  
2. **`<SAFE_name>_B5B6B7B8A_B11B12.tif`**: a 20 m resolution, 6-band GeoTIFF containing Sentinel-2 bands B05, B06, B07, B8A, B11, and B12.  
3. **`<SAFE_name>_Bathymetry.tif`**: bathymetry data aligned and warped to the Sentinel-2 pixel grid.  
4. **`<SAFE_name>_Substrate.tif`**: substrate classification data aligned and warped to the Sentinel-2 pixel grid.  
5. **`output.tif`** (or the filename you specify): a **binary GeoTIFF**, where kelp is labeled as `1` and non-kelp as `0`.  

**For `model_s2bandsandindices_only`:**
1. **`<SAFE_name>_B2B3B4B8.tif`**: a 10 m resolution, 4-band GeoTIFF containing Sentinel-2 bands B02 (Blue), B03 (Green), B04 (Red), and B08 (Near-Infrared).  
2. **`<SAFE_name>_B5B6B7B8A_B11B12.tif`**: a 20 m resolution, 6-band GeoTIFF containing Sentinel-2 bands B05, B06, B07, B8A, B11, and B12.  
3. **`output.tif`** (or the filename you specify): a **binary GeoTIFF**, where kelp is labeled as `1` and non-kelp as `0`.

---

## ⚙️ Project Structure

```text  
skema/  
├── skema/  
│   ├── cli.py  
│   ├── lib.py  
│   ├── __init__.py  
│   │  
│   └── static/  
│       ├── __init__.py  
│       │  
│       └── bathy_substrate/  
│           ├── __init__.py  
│           ├── Bathymetry_10m.tif  
│           ├── NCC_substrate_20m.tif  
│           ├── SOG_substrate_20m.tif  
│           ├── WCVI_substrate_20m.tif  
│           ├── QCS_substrate_20m.tif  
│           └── HG_substrate_20m.tif  
├── pyproject.toml  
├── setup.py  
├── requirements.txt  
├── README.md  
```

---

## 📜 License
- **Code**: MIT License (see LICENSE file)
- **Model**: The trained model is licensed under **CC-BY-4.0** — please cite the DOI when using it.