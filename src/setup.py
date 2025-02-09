from setuptools import setup, find_packages, Extension
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the extension module
ext_modules = [
    Extension(
        "ctc_forced_aligner.align_ops",
        ["ctc_forced_aligner/main.cpp"],
        extra_compile_args=["-std=c++17"],
        language="c++",
    ),
]


setup(
    name="ctc_forced_aligner",
    version="1.0.2.dev2",
    author="Deskpai.com",
    author_email="dev@deskpai.com",
    description="CTC Forced Alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deskpai/ctc_forced_aligner",
    packages=find_packages(),
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=[
        "requests",
        "librosa>=0.10.2.post1",
        "numpy",
        "onnxruntime"
    ],
    package_data={
        "ctc_forced_aligner": ["punctuations.lst"],
        "README.md": ["README.md"],
        "LICENSE": ["LICENSE"],
    },
    extras_require={
        "gpu": ["onnxruntime-gpu"],
        "torch": ["torch", "torchaudio"],
        "all": ["torch", "torchaudio", "onnxruntime-gpu"],
    },
    include_package_data=True,
)
