# mini_Genie

An efficient ab-initio method of simulating the financial markets on billions of parameter combinations

Preparation Steps:

    1. Make sure you've got Python & pip
        Before  you  go  any  further, make sure you have Python and that it's available from your
        command line. You can check this by simply running:
        
            $ python --version
        
        You should get some output like 3.9.7. If you do  not  have  Python,  please  install  the
        latest  3.x (genie uses 3.9) version  from  python.org  or  refer  to the Installing Python 
        section of The Hitchhiker's Guide to Python. 
        
        Additionally, you'll need to make sure you have pip  available.  You  can  check  this  by
        running:

            $ pip --version
        
        If  you  installed  Python from source, with an installer from python.org, or via Homebrew
        you should already have pip9. If you're on Linux  and  installed  using  your  OS  package
        manager, you may have to install pip separately.

    2. Install Pipenv
        Use pip to install Pipenv:

            $ pip install --user pipenv
        
        Note:
          If  pipenv isn't  available  in  your shell after installation, you'll need to add the 
          user base's binary directory to your PATH. On Linux and macOS you can find the user base 
          binary directory  by  running  python  -m   site  --user-base  and  adding  bin  to the 
          end. For example, this will typically print ~/.local (with ~ expanded to the absolute path 
          to your home directory) so  you'll  need  to  add  ~/.local/bin  to  your  PATH.  You  can 
          set your PATH permanently by modifying ~/.profile.

*******
Installation Steps:

	3. Clone git repository

        https://{GH_TOKEN}github.com/ruben970105/mini_Genie.git

	4. $cd mini_genie

    5. pipenv install

	5. $pipenv run (*python3.9) mini_genie_source/main_mini_genie.py
    
    Note:
        pipenv --rm (if reset needed)


