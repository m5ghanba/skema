# SkeMa

**Satellite-based Kelp Mapping using Semantic Segmentation on Sentinel-2 imagery**

`skema` is a Python tool for classifying kelp in Sentinel-2 satellite images using a deep learning segmentation model (PyTorch). It provides a command-line interface (CLI) for easy, reproducible inference. The following instructions are provided for anyone with no knowledge of what a command line is, no knowledge of Python or virtual environments, etc. Just follow along step by step.

---

## 🚀 Installation

Before you can set up SkeMa, you'll need **Python** (version 3.8 or higher) and **Git** installed on your computer. These are free tools—no accounts or sign-ups are required to install them or to clone the repository from GitHub later. We'll install them using your terminal (command line) where possible for simplicity. If you're on Windows, ensure you're using **PowerShell** or **Command Prompt** as Administrator (right-click and select "Run as administrator") for some steps.

### Step 1: Install Prerequisites (Python and Git)

#### On Windows
1. **Check if Winget is available** (it's built into Windows 10 version 2009 or later, or Windows 11—most modern systems have it):
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

3. **Install Git**:
   - In your terminal, run:
     ```
     winget install --id Git.Git -e --source winget
     ```
   - This installs Git. It adds itself to PATH automatically.
   - Restart your terminal.
   - Verify: Run `git --version`. It should output something like "git version 2.46.0.windows.1".

   *Alternative if winget fails*: Download the installer from [git-scm.com](https://git-scm.com/download/win) using your browser, run it, and follow the GUI prompts (use defaults). Then verify as above.

#### On macOS
1. **Install Homebrew** (a package manager for CLI installations—if you don't have it):
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

3. **Install Git**:
   - Run:
     ```
     brew install git
     ```
   - Verify: Run `git --version`. It should output something like "git version 2.46.0".

   *Alternative*: Download the official installer from [python.org](https://www.python.org/downloads/macos/) and [git-scm.com](https://git-scm.com/download/mac) using your browser, run them, and follow GUI steps.

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
     sudo apt install python3.12 python3.12-venv
     ```
   - This installs Python and the venv module.
   - Verify: Run `python3 --version`. It should output "Python 3.12.x".
   - *For Fedora/RHEL*: Use `sudo dnf install python3.12` instead.

3. **Install Git**:
   - Run:
     ```
     sudo apt install git
     ```
   - Verify: Run `git --version`.
   - *For Fedora/RHEL*: Use `sudo dnf install git`.

   *Note*: If your distro's repositories don't have Python 3.12, add a PPA (e.g., for Ubuntu: `sudo add-apt-repository ppa:deadsnakes/ppa` then update and install).

Once Python and Git are installed and verified, proceed to the next section. If you encounter errors (e.g., "command not found"), search online for the exact error message + your OS.

### Step 2: Install SkeMa
We recommend creating a **virtual environment**. A virtual environment is like a clean sandbox that keeps all the Python packages for this project separate from your system-wide Python installation.

To do this, open your **terminal**:  
- On **Windows**, you can use Command Prompt, PowerShell, or Anaconda Prompt.  
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

Now, navigate to a directory where you want to download the SkeMa installation files, then run:

```  
python -m venv skema_env  
# On Windows:  
skema_env\Scripts\activate  
# On macOS/Linux:  
source skema_env/bin/activate  

# Clone the repository  
git clone https://github.com/m5ghanba/skema.git  
```  

This will clone the repository into a new folder named `skema` in your current working directory. 

#### Static files  
There are necessary **static files** that need to be downloaded and placed inside. These are bathymetry and substrate files from the whole coast of British Columbia that `skema` uses when predicting kelp on a Sentinel-2 image.  

- The bathymetry file is a single TIFF raster (`Bathymetry_10m.tif`).  
- There are five substrate TIFF rasters (`NCC_substrate_20m.tif`, `SOG_substrate_20m.tif`, `WCVI_substrate_20m.tif`, `QCS_substrate_20m.tif`, `HG_substrate_20m.tif`), each covering a different region of the BC coast.  
- Place them inside:  

```text  
skema/skema/static/bathy_substrate/  
``` 

**Sources**:  
- Canada’s DEM/bathymetry model (10m resolution):  
  - Documentation: https://publications.gc.ca/collections/collection_2023/rncan-nrcan/m183-2/M183-2-8963-eng.pdf  
  - Dataset: https://maps-cartes.services.geo.ca/server_serveur/rest/services/NRCan/canada_west_coast_DEM_en/MapServer  

- Shallow substrate model (20m) of the Pacific Canadian coast (Haggarty et al., 2020):  
  https://osdp-psdo.canada.ca/dp/en/search/metadata/NRCAN-FGP-1-b100cf6c-7818-4748-9960-9eab2aa6a7a0  


Now, install SkeMa:  

```bash  
cd skema                   # Move into the repository folder  
pip install build          # Install build tool  
python -m build            # Build the package into a wheel file  
pip install --upgrade pip setuptools wheel  # Update packaging tools  
pip install --force-reinstall dist/skema-0.1.0-py3-none-any.whl  # Install SkeMa  
``` 

Each line:  
- `cd skema`: changes into the SkeMa project folder.  
- `pip install build`: installs Python’s build helper.  
- `python -m build`: creates a Python package distribution.  
- `pip install --upgrade ...`: ensures packaging tools are up to date.  
- `pip install --force-reinstall ...`: installs the SkeMa wheel file you just built.  

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

Skip this step if you don’t have a GPU.


#### ⚠️ GDAL Installation Issue on Windows

If you encounter an error when installing `gdal` after running:

```bash
pip install --force-reinstall dist/skema-0.1.0-py3-none-any.whl
```

it is recommended to install GDAL using a **precompiled wheel** from:

👉 [https://github.com/cgohlke/geospatial-wheels/releases](https://github.com/cgohlke/geospatial-wheels/releases)

##### 🪟 For Windows Users:

1. Visit the link above.
2. Under the **Assets** section of the desired release (e.g., `v2023.9.30`), download the appropriate wheel for your Python version. For example, if you're using **Python 3.9 on 64-bit Windows**, download: GDAL-3.7.2-cp39-cp39-win_amd64.whl. 
3. Install it using pip:

```bash
pip install path/to/GDAL-3.7.2-cp39-cp39-win_amd64.whl
```

---

## 🛰️ Usage

SkeMa uses Sentinel-2 satellite images, which can be downloaded from the [Copernicus Browser](https://dataspace.copernicus.eu/browser/). To access these images, you need to create a free account on the Copernicus Data Space website:
- Visit [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/) and click "Sign Up" to create an account.
- Follow the instructions to register with your email and set a password.
- Once logged in, use the Copernicus Browser to search for and download Sentinel-2 images, which will be provided as `.zip` files.

Now, you can run skema on a new Sentinel-2 image:

```bash  
skema --input-dir path/to/sentinel2/safe/folder --output-filename output.tif  
``` 

- The first path (`--input-dir`) must be the full path to the `.SAFE` folder.  
  - Sentinel-2 images from the Copernicus Browser come as `.zip` files. Extract them first.  
  - Then, pass the full path to the `.SAFE` folder (e.g., `C:\...\S2C_MSIL2A_20250715T194921_N0511_R085_T09UUU_20250716T001356.SAFE`).  

- The second parameter (`--output-filename`) is the name of the output file (e.g., `output.tif`).  

After running, the tool generates a folder with the same name as the `.SAFE` file. Inside this folder, there are **five TIFF files**:  

1. **`<SAFE_name>_B2B3B4B8.tif`** — a 10 m resolution, 4-band GeoTIFF containing Sentinel-2 bands B02 (Blue), B03 (Green), B04 (Red), and B08 (Near-Infrared).  
2. **`<SAFE_name>_B5B6B7B8A_B11B12.tif`** — a 20 m resolution, 6-band GeoTIFF containing Sentinel-2 bands B05, B06, B07, B8A, B11, and B12.  
3. **`<SAFE_name>_Bathymetry.tif`** — bathymetry data aligned and warped to the Sentinel-2 pixel grid.  
4. **`<SAFE_name>_Substrate.tif`** — substrate classification data aligned and warped to the Sentinel-2 pixel grid.  
5. **`output.tif`** (or the filename you specify) — a **binary GeoTIFF**, where kelp is labeled as `1` and non-kelp as `0`.  


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

## 📦 Environment (for Conda users)

```bash  
conda env create -f conda/environment.yml  
conda activate skema_env  
``` 

---

## 📜 License

MIT License.  
"""

