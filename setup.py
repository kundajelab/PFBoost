import os, sys
import numpy
from setuptools import setup, Extension, find_packages

def main():
    setup(
        name = "boosting_2D",
        version = idr.__version__,

        author = "Peyton Greenside",
        author_email = "pgreens@stanford.edu",

        install_requires = [ 'scipy', 'numpy'  ],

        extra_requires=['matplotlib'],

        packages= ['boosting_2D',],

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
    )

if __name__ == '__main__':
    main()
