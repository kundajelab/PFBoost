import os, sys
import numpy
from setuptools import setup, Extension, find_packages

from Cython.Build import cythonize

def main():
    setup(
        name = "boosting_2D",
        version = "0.9.1beta1",

        author = "Peyton Greenside",
        author_email = "pgreens@stanford.edu",

        install_requires = [ 'scipy', 'numpy'>=1.11],

        extra_requires=['matplotlib'],

        packages= ['boosting_2D'],

        description = ("Boosting will solve all the problems -- Peyton"),

        license = "GPL3",
        keywords = "boosting",
        url = "https://github.com/kundajelab/boosting_2D",

        long_description="""
        2 dimensional boosting with Alternating Decision Trees. Learn the regulatory programs - transcriptional regulators and their corresponding motifs - that govern dynamic patterns of chromatin accessibility or gene expression across conditions such as time courses, different cell types, or experimental perturbations.
        """,
        classifiers=[
            "Programming Language :: Python :: 2",
            "Development Status :: 3 - Alpha",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        ],

        ext_modules=cythonize("boosting_2D/util_functions.pyx"),
        include_dirs=[numpy.get_include()]
    )

if __name__ == '__main__':
    main()
