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

## Table of Contents
- [Quick Start (Experienced Users)](#-quick-start-experienced-users)
- [Citation](#citation)
- [Installation](#-installation)
  - [Step 1: Install Python](#step-1-install-python)
  - [Step 2: Install SKeMa](#step-2-install-skema)
  - [Static Files](#static-files)
  - [GPU Support](#gpu-support)
- [Usage](#️-usage)
  - [Activating the Virtual Environment](#activating-the-virtual-environment)
  - [Downloading Sentinel-2 Images](#downloading-sentinel-2-images)
  - [Running SKeMa](#running-skema)
  - [Model Types](#model-types)
  - [Output Files](#output-files)
- [Project Structure](#️-project-structure)
- [License](#-license)

---

## ⚡ Quick Start (Experienced Users)
```bash
pip install skema-kelp

# Download static files (for model_full only) from sources listed below
# Download Sentinel-2 imagery from https://dataspace.copernicus.eu/browser/

skema --input-dir "path/to/S2_scene.SAFE" --output-filename output.tif

# For help and all options
skema --help
```

**For detailed installation instructions (beginner-friendly), see [Installation](#-installation) below.**

---

## Citation

If you use **SKeMa** in your research or work, please cite:

```bibtex
@software{skema_2025,
  author       = {Mohsen Ghanbari, Neil Ernst, Taylor A. Denouden, Luba Y. Reshitnyk, Piper Steffen, Alena Wachmann, Alejandra Mora-Sotoa, Eduardo Loos, Margot Hessing-Lewis, Nic Dedeluke, Maycira Costa}, 
  title        = {SKeMa: Satellite-based Kelp Mapping using Semantic Segmentation on Sentinel-2 imagery},
  year         = 2026,
  publisher    = {Hugging Face},
  doi          = {10.57967/hf/6790},
  url          = {https://huggingface.co/m5ghanba/SKeMa}
}
```

---

## 🚀 Installation

Before you can set up SKeMa, you'll need **Python** (version **3.8 to 3.12**) installed on your computer. Python is a free tool, and no accounts or sign-ups are required to install it. We'll install it using your terminal (command line) where possible for simplicity. If you're on Windows, ensure you're using **PowerShell** or **Command Prompt** as Administrator (right-click and select "Run as administrator") for some steps. In some cases, however, you may need to use Command Prompt instead of PowerShell because PowerShell can handle command resolution and PATH variables differently, which may prevent it from recognizing Python even if it is installed. Additionally, if Python was installed only for your user account (and not system-wide), it may only be accessible from a normal Command Prompt session rather than one opened as Administrator, since elevated terminals can use a different set of environment variables.

### Step 1: Install Python

**Checking Your Current Python Version:**

Before proceeding, check if you already have Python between versions 3.8 and 3.12 on your computer.
Open a terminal (PowerShell, Command Prompt, or macOS/Linux terminal).

Run:
```
python --version
```
or, on macOS/Linux, try:
```
python3 --version
```

If the version shown is between 3.8 and 3.12 (e.g., Python 3.9.13, Python 3.11.5, Python 3.12.7), you already meet the requirement and can skip the installation steps below. Otherwise, follow the instructions below to install Python.

#### On Windows
1. **Check if Winget is available** (it's built into Windows 10 version 2009 or later, or Windows 11, and most modern systems have it):
   - Open PowerShell or Command Prompt.
   - Type `winget --version` and press Enter.
   - If it shows a version number (e.g., "v1.8.0"), proceed. If not (error like "winget is not recognized"), download the App Installer from the Microsoft Store (search for "App Installer") or update Windows via Settings > Update & Security > Windows Update.

2. **Install Python 3.12.7** (the most reliable and stable subversion of Python 3.12):
   - In your terminal, run:
     ```
     winget install -e --id Python.Python.3.12
     ```
   - This downloads and installs Python automatically. It may take a few minutes.
   - **Important**: During installation (if prompted), ensure "Add Python to PATH" is selected (it usually is by default with winget).
   - Restart your terminal after installation.
   - Verify: 
      - Run `python --version`. It should output something like "Python 3.12.7". If not, close and reopen the terminal, or manually add Python to PATH (search online for "add Python to PATH Windows").
      - ⚠️ Note: You may need to run Command Prompt (not Windows PowerShell) and open it normally (not as Administrator) because PowerShell and elevated terminals can handle PATH variables differently, which may prevent Python from being recognized.

   *Alternative if winget fails*: Download the installer directly from the Python website:
   - For Python 3.12.7: [https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe](https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe)
   - Run the downloaded installer.
   - **IMPORTANT**: During installation, make sure to check the box that says **"Add Python to PATH"** at the beginning of the installation process. This option is **unchecked by default**, so you must manually select it.
   - Follow the GUI prompts to complete the installation.
   - Verify: 
      - Run `python --version` in a new terminal window. It should output "Python 3.12.7".
      - ⚠️ Note: You may need to run Command Prompt (not Windows PowerShell) and open it normally (not as Administrator) because PowerShell and elevated terminals can handle PATH variables differently, which may prevent Python from being recognized.

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

   *Alternative*: Download the official installer from [python.org](https://www.python.org/downloads/macos/) using your browser, run it, and follow GUI steps. **Make sure Python is added to your PATH during installation.**

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
# Create a virtual environment 
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

### Static files  
There are necessary **static files** that need to be manually downloaded and placed inside the corresponding directory as described below. These are bathymetry and substrate files from the whole coast of British Columbia that `skema` uses when predicting kelp on a Sentinel-2 image.  

- The bathymetry file is a single TIFF raster (`Bathymetry_10m.tif`).  
- There are five substrate TIFF rasters (`NCC_substrate_20m.tif`, `SOG_substrate_20m.tif`, `WCVI_substrate_20m.tif`, `QCS_substrate_20m.tif`, `HG_substrate_20m.tif`), each covering a different region of the BC coast.  

When SKeMa is installed via `pip`, there is a folder named `bathy_substrate` located in the following directory. Place all six static files inside this folder:

On Windows:
```
skema_env\Lib\site-packages\skema\static\bathy_substrate
```
On macOS/Linux:
```
skema_env/lib/python3.11/site-packages/skema/static/bathy_substrate/
```
(Adjust the Python version number and virtual-environment name as appropriate for your system.)

**⚠️ Note**: Static files (bathymetry and substrate) are only required when using the **full model** (`--model-type model_full`). If you plan to use only the **S2-only model** (`--model-type model_s2bandsandindices_only`), you can skip downloading these files.

If you installed `skema` by cloning the **GitHub repository** instead of using `pip`, please place the downloaded files inside:
```
skema/static/bathy_substrate/
```

**Sources**:  
- Canada's DEM/bathymetry model (10m resolution):  
  - Documentation: https://publications.gc.ca/collections/collection_2023/rncan-nrcan/m183-2/M183-2-8963-eng.pdf  
  - Dataset: https://maps-cartes.services.geo.ca/server_serveur/rest/services/NRCan/canada_west_coast_DEM_en/MapServer  

- Shallow substrate model (20m) of the Pacific Canadian coast (Haggarty et al., 2020):  
  https://osdp-psdo.canada.ca/dp/en/search/metadata/NRCAN-FGP-1-b100cf6c-7818-4748-9960-9eab2aa6a7a0  

If you encounter any issues downloading these files, please don't hesitate to contact us for assistance.


### GPU support  

For GPU users, install CUDA-supported PyTorch that matches your CUDA Toolkit. Check your CUDA version with:  

```bash  
nvcc --version  
```

For CUDA 12.1:  

```bash  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  
```

For CUDA 11.8:  

```bash  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
```

This will install the latest compatible versions of PyTorch, torchvision, and torchaudio for your CUDA version.

Skip this step if you don't have a GPU.

---

## 🛰️ Usage

### Activating the Virtual Environment

To use SKeMa after the initial installation, you must activate its virtual environment each time you start a new session (if you created one). Follow these steps each time you want to run the tool:

1. **Open a terminal** (Command Prompt, PowerShell, or Terminal).
2. **Navigate to the directory** where you created the virtual environment:
   - **On Windows:**
     ```
     cd path\to\your\directory
     ```
   - **On macOS/Linux:**
     ```
     cd path/to/your/directory
     ```
3. **Activate the virtual environment:**
   - **On Windows:**
     ```
     skema_env\Scripts\activate
     ```
   - **On macOS/Linux:**
     ```
     source skema_env/bin/activate
     ```
4. **Run SKeMa** using the commands described below.

If your command line prompt shows `(skema_env)`, the virtual environment is activated and you're ready to proceed.

### Downloading Sentinel-2 Images

SKeMa uses Sentinel-2 satellite images, which can be downloaded from the [Copernicus Browser](https://dataspace.copernicus.eu/browser/). You will need a free account to access and download these images, which are provided as `.zip` files.

#### Sentinel-2 Image Download Instructions

Follow these steps to download Sentinel-2 images:

1. **Go to the Copernicus Browser**: Navigate to [https://browser.dataspace.copernicus.eu/](https://browser.dataspace.copernicus.eu/)

2. **Register**: If you haven't already, create a free account by clicking on the "Register" option.

3. **Sign in**: Log in to your account using your credentials.

4. **Create an area of interest**: Click on the top right option **"Create an area of interest"** (the pentagon shape icon).
   
   ![Create Area of Interest](docs/images/area_of_interest.png)

5. **Draw your polygon**: Click on the option **"Draw polygon of interest..."** (the pencil shape icon).
   
   ![Draw AOI](docs/images/draw_aoi.png)

6. **Define your area**: Draw a polygon around your area of interest by clicking on the map to create points. Close the polygon by clicking on the first point you placed.
   
   ![AOI Drawn](docs/images/aoi_drawn.png)

7. **Set parameters and search**: 
   - Select the **Time Range** option and set the dates in the **"From"** and **"Until"** fields.
   - Optionally, set the **cloud coverage** filter.
   - Click on **"Find products within selected time range"**.
   - Verify that **Sentinel-2 L2A** is selected as the data level (default setting).
   
   ![Find Products](docs/images/find_products.png)

8. **Choose your Sentinel-2 scene**: From the map, select the desired Sentinel-2 scene by clicking on it.
   
   ![Choose Scene](docs/images/choose_the_favorite_scene.png)

9. **Preview and download**: 
   - Click on the **"i"** option to view a preview of the image and quick information.
   - Click on the **download icon** (located at the bottom right) to download the scene.
   
   ![Thumbnail and Download](docs/images/thumbnail_and_download.png)


Readme section · MD
Copy

### Running SKeMa

SKeMa can be run on a single Sentinel-2 scene or on a directory of multiple scenes at once.

#### Single Scene

```bash  
skema --input-dir "path/to/sentinel2/scene.SAFE" --output-filename output.tif  
```

- `--input-dir` must be the full path to the `.SAFE` folder.  
  - Sentinel-2 images from the Copernicus Browser come as `.zip` files. Extract them first.  
  - Then, pass the full path to the `.SAFE` folder (e.g., `"C:\...\S2C_MSIL2A_20250715T194921_N0511_R085_T09UUU_20250716T001356.SAFE"`).
  - **Note**: If your path or output filename contains spaces, enclose it in double quotation marks (as shown in the example).

- `--output-filename` is the name of the output file (e.g., `output.tif`). You only need to provide the filename, not the full directory path. The output will be saved in a folder created alongside the `.SAFE` folder.

#### Batch Mode (Multiple Scenes + Mosaic)

If you have multiple Sentinel-2 scenes to process, use the `--batch-dir` flag instead of running the tool repeatedly. Place all your unzipped `.SAFE` folders inside a single directory and point `--input-dir` to that directory:

```bash
skema --input-dir "path/to/folder/with/safe/files/" --output-filename output.tif --batch-dir
```

SKeMa will:
1. Detect all `.SAFE` folders inside the directory automatically.
2. Process each scene individually, exactly as in single-scene mode.
3. Save each scene's prediction inside a subfolder named after that scene. The output filename will be the scene name with your `--output-filename` value appended as a suffix. For example, if the scene is `S2B_MSIL2A_20220806T191919_N0510_R099_T10UCV_20240718T204439.SAFE` and `--output-filename` is `output.tif`, the prediction will be saved as:
   ```
   path/to/folder/with/safe/files/
   └── S2B_MSIL2A_20220806T191919_N0510_R099_T10UCV_20240718T204439/
       ├── S2B_MSIL2A_20220806T191919_N0510_R099_T10UCV_20240718T204439_B2B3B4B8.tif
       ├── S2B_MSIL2A_20220806T191919_N0510_R099_T10UCV_20240718T204439_B5B6B7B8A_B11B12.tif
       ├── ...
       └── S2B_MSIL2A_20220806T191919_N0510_R099_T10UCV_20240718T204439_output.tif
   ```
4. After all scenes are processed, generate a single mosaic by merging all predictions into one GeoTIFF saved as `mosaic_kelp_map.tif` in the input directory:
   ```
   path/to/folder/with/safe/files/
   ├── S2B_MSIL2A_.../
   ├── S2B_MSIL2A_.../
   └── mosaic_kelp_map.tif   ← combined prediction for all scenes
   ```

The mosaic is reprojected to BC Albers (EPSG:3005) at 10 m resolution. Overlapping pixels between scenes are resolved by taking the maximum value, meaning a pixel is classified as kelp if any contributing scene predicted kelp there.

### Model Types

SKeMa supports two model types:

1. **`model_full`** (default): Uses all available data including Sentinel-2 bands, bathymetry, and substrate information. This model provides the most accurate predictions but requires bathymetry and substrate static files.

2. **`model_s2bandsandindices_only`**: Uses only Sentinel-2 bands and derived spectral indices. This model does not require bathymetry or substrate files, making it suitable for areas outside British Columbia or when static files are unavailable.

To specify the model type, use the `--model-type` flag:

**Note**: Scroll right to see the complete command if it extends beyond your screen.

```bash
# Using the full model (default - includes bathymetry and substrate)
skema --input-dir "path/to/sentinel2/safe/folder" --output-filename output.tif --model-type model_full

# Using S2-only model (no bathymetry/substrate required)
skema --input-dir "path/to/sentinel2/safe/folder" --output-filename output.tif --model-type model_s2bandsandindices_only

# Batch mode also supports --model-type
skema --input-dir "path/to/folder/with/safe/files/" --output-filename output.tif --batch-dir --model-type model_s2bandsandindices_only
```

If `--model-type` is not specified, the tool defaults to `model_full`.

### Output Files

#### Single Scene

After running, the tool generates a folder with the same name as the `.SAFE` file (without the `.SAFE` extension), located alongside it. Inside this folder, you'll find:

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

#### Batch Mode

In addition to the per-scene output folders described above (with `<SAFE_name>_output.tif` inside each), batch mode produces one additional file in the input directory:

- **`mosaic_kelp_map.tif`**: a single binary GeoTIFF mosaic merging all scene predictions, in BC Albers projection (EPSG:3005) at 10 m resolution.
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