#!/usr/bin/env python
from logger_tt import setup_logging, logger

setup_logging(full_context=1)


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
    from os import system
    logger.info("Setting Up mini-Genie "
                "\N{Smiling Face With Heart-Shaped Eyes}"
                "\N{Smiling Face With Smiling Eyes And Hand Covering Mouth}"
                "\N{money-mouth face}"
                )
    #
    system('touch .working_directory_.txt')
    system('pip install -r requirements.txt')
    create_dir('Datas')
    create_dir('Studies')
    system(
        'pip install -U \"vectorbtpro[full] @ git+https://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@github.com/polakowo/vectorbt.pro.git\"')
    #
    system('chmod +x mini_genie_source/main_mini_genie.py')
    #
    system('ln -s mini_genie_source/main_mini_genie.py mini_genie.py')
    system('Im done getting ready, check me out \N{Smiling Face With Smiling Eyes}')
    #


if __name__ == "__main__":
    set_up_mini_genie()
