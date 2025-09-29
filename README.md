# SkeMa

**Satellite-based Kelp Mapping using Semantic Segmentation on Sentinel-2 imagery**

`skema` is a Python tool for classifying kelp in Sentinel-2 satellite images using a deep learning segmentation model (PyTorch). It provides a command-line interface (CLI) for easy, reproducible inference. The following instruction is provided for anyone with no knowledge of what command line is, no knowledge of Python or virtual environments, etc. just follow along step by step.

---

## 🚀 Installation

We recommend creating a **virtual environment**. A virtual environment is like a clean sandbox that keeps all the Python packages for this project separate from your system-wide Python installation.  

To do this, open your **terminal**:  
- On **Windows**, you can use Command Prompt, PowerShell, or Anaconda Prompt.  
- On **macOS**, open the Terminal app.  
- On **Linux**, open your terminal emulator of choice.  

When you open a terminal, you start inside a **directory (folder)**. You can move to another directory with the command `cd`. For example:  

```bash  
cd C:\\Users\\YourName\\Documents  
``` 

On macOS/Linux:  

```bash  
cd /Users/yourname/Documents  
``` 

👉 The easiest way to navigate is to open your file explorer, go to the folder you want, then copy its full path and paste it after `cd` on the command line. For more details, look up "basic terminal navigation" online.  

Now, navigate to a directory where you want to download the SkeMa installation files, then run:


```bash  
python -m venv skema_env  
skema_env\\Scripts\\activate  # On Windows  
# or  
source skema_env/bin/activate  # On macOS/Linux  

# Clone the repository  
git clone https://github.com/m5ghanba/skema.git  
``` 

This will clone the repository into a new folder named `skema` in your current working directory.  

### Static files  
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

---

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

---

### GPU support  

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

---

## 🛰️ Usage

Run semantic segmentation on a new Sentinel-2 image:

```bash  
skema --input-dir path/to/sentinel2/safe/folder --output-filename output.tif  
``` 

- The first path (`--input-dir`) must be the full path to the `.SAFE` folder.  
  - Sentinel-2 images from the Copernicus Browser come as `.zip` files. Extract them first.  
  - Then, pass the full path to the `.SAFE` folder (e.g., `C:\\...\\S2C_MSIL2A_20250715T194921_N0511_R085_T09UUU_20250716T001356.SAFE`).  

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

