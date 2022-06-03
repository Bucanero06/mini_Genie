#!/usr/bin/env python3.9
def create_dir(directories):
    """

    Args:
        dir:

    Returns:
        object:
    """
    from os import path, mkdir

    if not isinstance(directories, str):
        for dir in directories:
            if not path.exists(directories):
                logger.info(f'Creating directory {dir}')
                mkdir(directories)
            else:
                logger.info(f'Found {dir}')
    else:
        if not path.exists(directories):
            logger.info(f'Creating directory {directories}')
            mkdir(directories)
        else:
            logger.info(f'Found {directories}')


def set_up_mini_genie():
    from os import system, path
    print("Setting Up mini-Genie "
          "\N{Smiling Face With Heart-Shaped Eyes}"
          "\N{Smiling Face With Smiling Eyes And Hand Covering Mouth}"
          "\N{money-mouth face}"

          )
    #
    # system("python3 -m venv .")
    #
    if path.exists('.working_directory_.txt'):
        system('rm -f .working_directory_.txt')
    #
    system('pipenv --rm')
    system('pipenv install')
    create_dir('Datas')
    create_dir('Studies')
    system('touch .working_directory_.txt')

    #
    print("\n\n"
          "Im done getting ready, check me out 🦾\N{Smiling Face With Smiling Eyes}\n"
          "Im done getting ready, check me out 🦿")
    system('./genie_trader.py --help')

    #
