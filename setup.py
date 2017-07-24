import os, sys
import numpy
from setuptools import setup, Extension, find_packages

def main():
    setup(
        name = "boosting2D",
        version = "0.9.1beta1",

        author = "Peyton Greenside",
        author_email = "pgreens@stanford.edu",

        install_requires = [ 'scipy', 'numpy>=1.11', 'pandas', 'sklearn'],

        extra_requires=['matplotlib'],

        packages= ['boosting2D'],

        description = ("2-dimensional boosting for gene regulatory networks"),

        license = "GPL3",
        keywords = "boosting",
        url = "https://github.com/kundajelab/boosting2D",

        long_description="""
        2-dimensional boosting with Alternating Decision Trees. Learn the regulatory programs - transcriptional regulators and their corresponding motifs - that govern dynamic patterns of chromatin accessibility or gene expression across conditions such as time courses, different cell types, or experimental perturbations.
        """,
        classifiers=[
            "Programming Language :: Python :: 2",
            "Development Status :: 3 - Alpha",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        ],

    )

if __name__ == '__main__':
    main()
