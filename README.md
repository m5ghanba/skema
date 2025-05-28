# skema

**Satellite-based Kelp Mapping using Semantic Segmentation on Sentinel-2 imagery**

`skema` is a Python package for classifying kelp in Sentinel-2 satellite images using a deep learning semantic segmentation model (PyTorch). It offers a command-line interface (CLI) to run inference on new imagery using a trained model.

---

## 📦 Features

- Semantic segmentation model trained on Sentinel-2 imagery
- Kelp and no-kelp classification
- Normalization and index calculation built-in
- GeoTIFF input and output support
- Easy command-line interface
- Designed for scientific reproducibility and modular development

---

## 🚀 Installation

We recommend using a virtual environment (e.g., Conda):

```bash
conda create -n skema_env python=3.10
conda activate skema_env

# Clone and install
git clone https://github.com/m5ghanba/skema.git
cd skema
pip install .
```
### 🔹 Alternatively (pure pip)

```bash
pip install git+https://github.com/yourusername/skema.git
```

---

## 🛰️ Usage

Run classification on a new Sentinel-2 image using the CLI:

```bash
skema \
  --input path/to/image.tif \
  --output path/to/prediction.tif \
  --model path/to/model.pth \
  --mean 0.3,0.3,0.3 \
  --std 0.1,0.1,0.1
```

### 📋 CLI Options

| Option       | Description                                  |
|--------------|----------------------------------------------|
| `--input`    | Path to input Sentinel-2 image (GeoTIFF)     |
| `--output`   | Path to save the output TIFF file            |
| `--model`    | Path to the `.pth` trained PyTorch model     |
| `--mean`     | Comma-separated list of mean values per band |
| `--std`      | Comma-separated list of std dev per band     |
| `--device`   | (Optional) Device to run on: `cpu` or `cuda` |

---

## 📁 Project Structure

```text
skema/
├── skema/
│   ├── model.py
│   ├── cli.py
│   ├── dataset.py
│   ├── io.py
│   ├── config.py
│   └── __init__.py
├── notebooks/
│   └── kelpNokelp_maskKelpOnly_usingSemanticSegmentationPytorch.ipynb
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── README.md
└── LICENSE.txt
```

---

## ⚙️ Development & Testing

### Editable install:

```bash
pip install -e .
```

### Linting and formatting:

```bash
black skema/
flake8 skema/
isort skema/
```

### Run tests:

```bash
pytest tests/
```

---

## 🐳 Docker Support

Build and run:

```bash
docker build -t skema .
docker run --rm -v "$PWD/data:/data" skema \
  --input /data/image.tif \
  --output /data/output.tif \
  --model /data/model.pth \
  --mean 0.3,0.3,0.3 --std 0.1,0.1,0.1
```

---

## ☁️ Binder Support

Try the demo notebook online (requires model + test image):  
[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/skema/HEAD?labpath=notebooks/kelpNokelp_maskKelpOnly_usingSemanticSegmentationPytorch.ipynb)

---

## 📜 License

MIT License. See [LICENSE.txt](LICENSE.txt) for full details.

---

## 🤝 Acknowledgements

Built with:
- PyTorch + PyTorch Lightning
- segmentation-models-pytorch
- rasterio, albumentations
- Sentinel-2 open data

Inspired by [Kelp-O-Matic](https://github.com/HakaiInstitute/kelp-o-matic)
