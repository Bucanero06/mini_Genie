import os
import sys

from setuptools import setup

setup(
    name='mini_Genie',
    version='0.1',
    packages=['Utilities', 'Data_Handler', 'Error_Handler', 'Analysis_Handler', 'Equipment_Handler',
              'mini_Genie_Object', 'Simulation_Handler', 'Optimization_Module_mini_genie'],
    package_dir={'': 'mini_genie_source'},
    url='',
    license='',
    author='ruben',
    author_email='',
    description='An efficient ab-initio method of simulating the financial markets on billions of parameter combinations'
)

os.system("hiiijoii")