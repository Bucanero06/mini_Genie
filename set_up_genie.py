#!/usr/bin/env python

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
                print(f'Creating directory {dir}')
                mkdir(directories)
            else:
                print(f'Found {dir}')
    else:
        if not path.exists(directories):
            print(f'Creating directory {directories}')
            mkdir(directories)
        else:
            print(f'Found {directories}')


def set_up_mini_genie():
    from os import system, path
    print("Setting Up mini-Genie "
          "\N{Smiling Face With Heart-Shaped Eyes}"
          "\N{Smiling Face With Smiling Eyes And Hand Covering Mouth}"
          "\N{money-mouth face}🦾"

          )
    #
    system("python3 -m venv .")
    #
    if path.exists('.working_directory_.txt'):
        system('rm -f .working_directory_.txt')
    #
    system('pip3 install .')
    system('pip3 install -r requirements.txt')
    create_dir('Datas')
    create_dir('Studies')
    system('touch .working_directory_.txt')
    system(
        'pip3 install -U \"vectorbtpro[full] @ git+https://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@github.com/polakowo/vectorbt.pro.git\"')
    #
    if not path.exists('mini_genie.py'):
        system('chmod +x mini_genie_source/main_mini_genie.py')
        system('ln -s mini_genie_source/main_mini_genie.py genie_trader.py')
    #
    print("\n\n"
          "Im done getting ready, check me out 🦾\N{Smiling Face With Smiling Eyes}\n"
          "Im done getting ready, check me out 🦿")
    system('./genie_trader.py --help')

    #


if __name__ == "__main__":
    set_up_mini_genie()
