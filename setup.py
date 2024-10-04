from setuptools import setup, find_packages

setup(
    name='mmdetection_ablation_studies',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "pycocotools>=2.0.7",
        "scipy>=1.10.1",
        "numpy>=1.24.1",
        "scikit-learn>=1.3.2",
        "Pillow>=9.3.0",
        "tqdm>=4.65.0",
        "mmdet>=3.0.0",
        "mmengine>=0.8.4",
        "pandas>=2.0.3",
        "matplotlib>=3.7.3",
        "ipywidgets>=8.0.4",
        "IPython>=8.12.3"
    ],
    description='Library for Ablation Studies on mmdetection Models',
    author='Florian Hoelken',
    author_email='hoelken@uni-wuppertal.de',
    keywords='ablation studies mmdetection models'
)