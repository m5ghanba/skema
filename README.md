# SkeMa

**Satellite-based Kelp Mapping using Semantic Segmentation on Sentinel-2 imagery**

`skema` is a Python tool for classifying kelp in Sentinel-2 satellite images using a deep learning segmentation model (PyTorch). It provides a command-line interface (CLI) for easy, reproducible inference.

---

## 🚀 Installation

We recommend creating a virtual environment:

```bash
python -m venv skema_env
skema_env\Scripts\activate  # On Windows
# or
source skema_env/bin/activate  # On macOS/Linux

# Clone and install
git clone https://github.com/m5ghanba/skema.git
cd skema
pip install build
python -m build
pip install --upgrade pip setuptools wheel
pip install --force-reinstall dist/skema-0.1.0-py3-none-any.whl  
```


---

## 🛰️ Usage

Run classification on a new Sentinel-2 image:

```bash
skema --input-dir path/to/image_folder \
      --output-filename output.tif
```

---

## ⚙️ Project Structure

```text
skema/
├── skema/
│   ├── cli.py
│   ├── lib.py
│   └── __init__.py
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
