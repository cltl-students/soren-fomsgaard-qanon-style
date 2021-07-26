# NOTE: 
# This is run from the top level src project folder calling ex.
# \src> python -m collecting -user_csv "../../../Data/NonQanon/NonQanon twitter accounts.csv" -handle_col 1 -save_file "../../../../Data/NonQanon/mumbo.csv" #TODO: Figure out why this argument has to be one level deeper in the folder structure. Because you pass this arg down to mumbo.run_queries?
# This code is important for now.
# TODO: Add default= options to the parser so you don't have to type it every time.
 
# %%
from utils.collect import queries
#from utils.grasp import *  # import grasp modules.
import re
import os

import argparse
import configparser
from collections import ChainMap # alternative to combining config args with commandline.

from . import mumbo

#%% # TODO implement a Chainmap / combination of these defaults and cmdline args.
config = configparser.ConfigParser()
config.read('config.ini')
defaults = config['collect']



#%% CMD argument parser
### adapted from https://stackoverflow.com/a/55983010 ###
def cmdlineparse(args):
    parser = argparse.ArgumentParser()

    # Username csv path
    parser.add_argument("-user_csv", dest='csv_file' , help="A path to a csv with user handles", type=str, required=True)
    parser.add_argument("-handle_col", dest='col' , help="The column in the file with user handles", type=int, required=True)
    parser.add_argument("-save_file", dest='outfile' , help="The filepath to save the collected tweets to (csv)", type=str, required=True)

    args = parser.parse_args(args)
    return args

#%%
# This wraps the main function calls of this script.
def main(args=None):
    cmd_args = cmdlineparse(args)
    #combined_args = ChainMap(cmd_args , defaults)
    q = queries.twitter(cmd_args.csv_file , cmd_args.col)
    print(f'Querying {len(q)} Twitter handles....')
    mumbo.run_queries(q, cmd_args.outfile)       

#%%
if __name__ == "__main__":
    main()

### last accessed 17-02-2021  ###

