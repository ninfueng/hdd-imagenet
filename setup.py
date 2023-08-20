from os import path

from setuptools import find_packages, setup

curdir = path.abspath(path.dirname(__file__))

with open(path.join(curdir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hdd-imagenet",
    version="0.0.0",
    description="A faster PyTorch ImageNet loader for HDDs.",
    long_description=long_description,
    url="https://github.com/ninfueng/hdd-imagenet",
    author="Ninnart Fuengfusin",
    author_email="ninnart.fuengfusin@yahoo.com",
    license="MIT",
    keywords="pytorch",
    packages=find_packages(),
    install_requires=["lmdb", "tqdm"],
)
