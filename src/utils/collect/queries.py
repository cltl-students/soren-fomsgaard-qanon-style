#%%
import sys
import os
sys.path.append(os.path.abspath('../../src'))

import pandas as pd
from string import Template

# %%
# from utils import grasp
# %%

# Body code goes here

def twitter(csv, handlecol):
    """Return tuple of user names to query on Twitter API
       using a csv and and index for the twitter handle.

    Args:
        csv (filepath): A filepath to a csv file.
        handlecol (int): An index for the column containing the user handles.
    Returns:
        query (tuple): A tuple of user handle strings to query.
    """
    
    template = Template("from:$handle") #TODO make sure to make it comma separated.
    
    with open(csv, 'r', encoding='utf-8') as infile:
        data = pd.read_csv(csv)
        
        handles = data.iloc[ : , handlecol]
    
    # Transform handles into the query schema: 'from:HANDLE1','from:HANDLE2'...
    queries = [] # via list.
    for handle in handles:
       query = template.substitute({'handle':handle[1:]})   # add the handle from the @ onwards.     ))
       queries.append(query) 
    
    return tuple(queries)

#%%
if __name__ == "__main__":
    #main()
    #print('hey!')
    test = twitter('../../../../Data/NonQanon/NonQanon twitter accounts.csv', 1)
# %%

# %%
