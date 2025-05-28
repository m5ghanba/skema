from setuptools import setup, find_packages

setup(
    name="skema",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
    "click>=8.1,<9",
    "torch>=2.1,<3",
    "torchvision>=0.16,<1",
    "pytorch-lightning>=2.1,<3",
    "albumentations>=1.3,<2",
    "rasterio>=1.3,<2",
    "numpy==1.26.4",
    "pandas==2.2.3",
    "matplotlib==3.7.5",
    "tifffile==2023.4.12",
    "scikit-learn==1.3.2",
    "scipy==1.11.4",  # ✅ added here
    "segmentation-models-pytorch==0.5.0",
    "torchmetrics==1.7.1",
    "tqdm>=4.66,<5"
    ],
    entry_points={
        "console_scripts": [
            "classify = skema.cli:main",
        ],
    },
)
