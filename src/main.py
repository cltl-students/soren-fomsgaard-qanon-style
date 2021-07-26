
# Here are two ways of calling scripts in daughter dirs.
# There are many more, but these are still readable.

#%% Methods 1
#from data_collection import collect_nonqanon

#%% Method 2 - this is the most barebones and works without defining function
# or using the if __name__ == "__main__" call.
#exec(open("collecting/collect_nonqanon.py").read())

#%%
import collecting #, exploring , processing , classifying

import utils

import configparser

#%%

config = configparser.ConfigParser()

#%%

config.read('./config.ini')

# %%

# Collect Data

# leave all the code here commented out.

#collecting.main(["../../../Data/NonQanon/NonQanon twitter accounts.csv" , 1 , "../../../../Data/NonQanon/mumbo.csv"])

# %%
