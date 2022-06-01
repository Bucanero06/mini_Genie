from os import system

from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='mini_Genie',
    version='0.1.0',
    packages=['mini_genie_source'],
    package_dir={'mini_genie_source': 'mini_genie_source'},
    description='An efficient ab-initio method of simulating the financial markets on billions of parameter combinations',
    url='',
    license='',
    author='Ruben Fernandez Carbon',
    author_email='',
    long_description=long_description,

)

system('pip install -r requirements.txt')

system(
    'pip install -U \"vectorbtpro[full] @ git+https://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@github.com/polakowo/vectorbt.pro.git\"')
