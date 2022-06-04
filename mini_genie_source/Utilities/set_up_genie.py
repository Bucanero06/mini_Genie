#!/usr/bin/env python3
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


def Find(filename, *args):
    directories = [*args]
    foundfile = False
    for searchdirectory in directories:
        if path.exists(searchdirectory + "/" + filename):
            if searchdirectory == ".":
                print("Found " + str(filename) + " inside the current directory")
            else:
                print("Found " + str(filename) + " inside " + str(searchdirectory) + " directory")
            foundfile = True
            return searchdirectory
            exit()
    # if not exited by now it means that the file was not found in any of the given directories thus rise error
    if foundfile != True:
        print(str(filename) + " not found inside " + str(directories) + "\n exiting...")
        sys.exit()


def set_up_mini_genie():
    from os import system, path
    print("Setting Up mini-Genie "
          "\N{Smiling Face With Heart-Shaped Eyes}"
          "\N{Smiling Face With Smiling Eyes And Hand Covering Mouth}"
          "\N{money-mouth face}"

          )
    #

    # Find .working_directory file path

    # Go to directory

    # Do stuff
    if path.exists('.working_directory.txt'):
        system('rm -f .working_directory.txt')
    #
    system('pip install pipenv')
    system('pip install pipenv --upgrade')
    system('pipenv --rm')
    system('pipenv install')
    create_dir('Datas')
    create_dir('Studies')
    system('touch .working_directory.txt')

    #
    print("\n\n"
          "Im done getting ready, check me out 🦾\N{Smiling Face With Smiling Eyes}\n"
          "                                    🦿")
    #


if __name__ == "__main__":
    set_up_mini_genie()
