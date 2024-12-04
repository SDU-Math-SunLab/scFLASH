#!/usr/bin/env python

# This is a shim to hopefully allow Github to detect the package, build is done with poetry

from setuptools import setup, find_packages

setup(
    name='scFLASH',
    version='1.0.0',
    description='Integrating multi-sample single-cell data with phenotypic expression heterogeneity.',
    author='Qingbin Zhou',
    author_email='sdsxdszqb@163.com',
    url='https://github.com/qingbinzhou/scFLASH',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.23.5',
        'pandas>=1.5.3',
        'scib>=1.1.4',
        'louvain>=0.8.2',

    ],
    zip_safe=False,
    license='GPL-3.0'
)
