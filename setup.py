#!/usr/bin/env python3.9
from setuptools import setup, find_packages

setup(
    name='mini_Genie',
    version='',
    packages=find_packages(),
    #
    url='https://github.com/ruben970105/mini_Genie',
    author='Ruben Fernandez Carbon',
    author_email='Ruben@MoonShoot.codes',
    description='An efficient ab-initio method of simulating the financial markets on billions of parameter combinations'
)
from os import system

system('pip3 install .')
