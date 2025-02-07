from setuptools import setup, find_packages, Extension

# Define the extension module
ext_modules = [
    Extension(
        "ctc_forced_aligner.align_ops",  # Use a submodule for clarity
        ["ctc_forced_aligner/main.cpp"],  # Ensure correct path
        extra_compile_args=["-std=c++17"],  # Use C++17 standard
        language="c++",
    ),
]

# Setup configuration
setup(
    name="ctc_forced_aligner",
    version="1.0.1",
    author="Deskpai",
    author_email="dev@deskpai.com",
    description="CTC Forced Alignment",
    packages=find_packages(),  # Include all Python packages
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=[
        "requests>=2.32.3",
        "librosa>=0.10.2.post1",
        "numpy==1.26.3",
        "onnxruntime>=1.20.1"
    ],
    package_data={
        "ctc_forced_aligner": ["punctuations.lst"],  # Include punctuations.lst in package
    },
    include_package_data=True,  # Ensure package data is included
)
