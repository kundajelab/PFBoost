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

        install_requires = [ 'scipy', 'numpy', 'grit' ],

        extra_requires=['matplotlib'],

        packages= ['boosting_2D'],

        description = ("Boosting will solve all the problems -- Peyton"),

        license = "GPL3",
        keywords = "boosting",
        url = "https://github.com/kundajelab/boosting_2D",

        long_description="""
        Peyton will fill in later
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
