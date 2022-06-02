from setuptools import setup

setup(
    name='mini_Genie',
    version='',
    packages=['mini_genie_source', 'mini_genie_source.Utilities', 'mini_genie_source.Strategies',
              'mini_genie_source.Data_Handler', 'mini_genie_source.Error_Handler', 'mini_genie_source.Analysis_Handler',
              'mini_genie_source.Equipment_Handler', 'mini_genie_source.mini_Genie_Object',
              'mini_genie_source.Simulation_Handler', 'mini_genie_source.Configuration_Files',
              'mini_genie_source.Optimization_Module_mini_genie'],
    url='',
    license='',
    author='Ruben Fernandez Carbon',
    author_email='',
    description=''
)
# from os import system
#
# from setuptools import setup
#
# with open("README.md", 'r') as f:
#     long_description = f.read()
#
# setup(
#     name='mini_Genie',
#     version='0.1.0',
#     packages=['mini_genie_source'],
#     package_dir={'mini_genie_source': 'mini_genie_source'},
#     description='An efficient ab-initio method of simulating the financial markets on billions of parameter combinations',
#     url='',
#     license='',
#     author='Ruben Fernandez Carbon',
#     author_email='',
#     long_description=long_description,
# )


# ##########################################################
# def set_up_mini_genie():
#     from os import system, path
#     print("Setting Up mini-Genie "
#           "\N{Smiling Face With Heart-Shaped Eyes}"
#           "\N{Smiling Face With Smiling Eyes And Hand Covering Mouth}"
#           "\N{money-mouth face}"
#
#           )
#     #
#     # system("python3 -m venv .")
#     #
#     if path.exists('.working_directory_.txt'):
#         system('rm -f .working_directory_.txt')
#         # system('python3 -m venv .')
#     #
#     #
#
#     system("pip install pipenv")
#     system(
#         "pipenv install -e \"vectorbtpro[full] @ git+https://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@github.com/polakowo/vectorbt.pro.git\"")
#     system('pipenv install -r requirements.txt')
#     system('pipenv install .')
#     # system(
#     #     'pip3 install -U \"vectorbtpro[full] @ git+https://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@github.com/polakowo/vectorbt.pro.git\"')
#     # #
#     # system('pip3 install .')
#     # system('pip3 install -r requirements.txt')
#     # create_dir('Datas')
#     # create_dir('Studies')
#     # system('touch .working_directory_.txt')
#
#     #
#     if not path.exists('mini_genie.py'):
#         system('chmod +x mini_genie_source/main_mini_genie.py')
#         system('ln -s mini_genie_source/main_mini_genie.py genie_trader.py')
#     #
#     print("\n\n"
#           "Im done getting ready, check me out 🦾\N{Smiling Face With Smiling Eyes}\n"
#           "Im done getting ready, check me out 🦿")
#     system('./genie_trader.py --help')