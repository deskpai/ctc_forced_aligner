from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "ctc_forced_aligner",  # Name of the module
        ["main.cpp"],  # Source file
        cxx_std=17,  # Use C++17 standard
    ),
]

# Setup configuration
setup(
    name="ctc_forced_aligner",
    version="0.0.1",
    author="Deskpai",
    author_email="dev@deskpai.com",
    description="CTC Forced Alignment",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    options={
        'build_ext': {
            'inplace': True,  # Build the shared library in the source directory
        },
    },
)

