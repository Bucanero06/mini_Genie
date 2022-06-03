from Utilities.general_utilities import create_dir


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
