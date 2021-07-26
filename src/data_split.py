#%% ANCHOR IMPORTS

import pandas as pd, numpy as np

#from fractions import Fraction
from sklearn.model_selection import train_test_split

#%%[markdown]
## Split processed data into datasets


#%% ANCHOR: SPLIT()

def merge_split(df1, df2, ratio=0.5): #size=[0.50, 0.10, 0.40]):
    """Merge and split source data into train, validation and test sets.

    Args:
        df1 ([type]): [description]
        df2 ([type]): [description]
        ratio (int, optional): [description]. Defaults to 50.
        size (list, optional): [description]. Defaults to [50, 40, 10].

    Returns:
        [type]: [description]
    """
    
    # Name dfs for easy reference
    
    # 1. determine df2's proportion of df1. should be df2/df1 2 Store in var(proportion)
    
    #proportion = df2.divide(other=df1) # equivalent to df2/df1   
    # print sizes of source data.
    
    #df1.sort_index().sort_index(axis=1) == df2.sort_index().sort_index(axis=1)
    # class ReferenceDict():
    #     def __init__(self, dict):
    #         self.Title = dict.keys()
    #         #pass
    
    
    # Use tuple instead ("largest" ) + (int ,) ? 
    # A dict to reference dfs with sizes.
    reference = {"largest" : int , "smallest" : int}
    
    # Option 1
    # reference = {"largest" : {str : int} , "smallest" : {str: int}}
    # if len(df1) > len(df2):
    #   reference["largest"] = {df1.columns.name : len(df1)}
    #   reference["smallest"] = {df2.columns.name : len(df1)}
    # else:
    #   reference["largest"] = {df2.columns.name : len(df2)}
    #   reference["largest"] = {df1.columns.name : len(df1)}
    
    # Option 2
    # reference = {df1.column.name : len(df1) , df2.columns.name : len(df2)}
    # smallest = min(reference[df1.columns.name] , reference[df2.columns.name])
    # largest = max(reference[df1.columns.name] , reference[df2.columns.name)
    
    # proportion = round(smallest / largest , ndigits=3)
    
    reference["largest"] = max(len(df1), len(df2))
    
    reference["smallest"] = min(len(df1), len(df2))
    
    
    # Find size of the smaller dataset compared to the bigger one.
    proportion = round(reference["smallest"] / reference["largest"] , ndigits=3)
    
    #proportion = Fraction(str(len(df2) / len(df1)))
    
    #print(f"Proportion of source data 1 : source data 2 is {proportion}")
    
    # Concat the source data sets.
    #2. if proportion <= 50  ; sample with full size of smaller
    
    if proportion < 0.5:
        print(f"One df is smaller than the other")
        print("Sampling the larger using the full size of the smaller...")
        
        # Make this use dynamic... use names inherited with df1.columns.name
        df_sample = df1.sample(frac=proportion)
    # Else, sample with user specified ratio.
    else:
        print("The df's are of comparable size")
        print(f"Sampling using a ratio of {ratio}...")
        df_sample = df1.sample(frac=ratio)
     
    
    # Concatenate the sample of the larger with the smaller src data. 
    
    df_combined = df2.append(df_sample)
    # print size of combined data.
    print(f"Size of combined source data is {len(df_combined)} instances.")
    
    
    
    # 
    #df_combined = df_combined.sample(frac=1)
    
    # TODO
    # Extract the labels
    y = df_combined.pop('cons')
    # X = df_combined('text_clean')
    # instance_columns = ['text_clean' ,'tokens_clean']
    # X = pd.concat([df_combined.pop(x) for x in instance_columns] , axis=1) # pop multiple columns into new DataFrame.
    X = df_combined
    
    # TODO make the X (feature vector) a parameter feature_col=[7, 8] or ["text_clean", "tokens_clean"]
    
    # TODO apply feature extraction on df_combined text col or tokens here?
    
    #3. split combined into df_train, df_val, df_test.
    
    # Split into train 50% and test+validation 50% (out of the combined data - 100)
    # Stratify to keep class balance (50/50).
    X_train, X_test,\
    y_train, y_test  = train_test_split(X, y, test_size=0.30, stratify=y, shuffle=True, random_state=18) 
   # org split here was 50
    
    # Split test 50% further into test 60/50 (30%) and validation 40/50 (20%) partitions. 
    
    
    # NOTE test_size here refers to VAL (train refers to test)!
    X_test, X_val,\
    y_test, y_val  = train_test_split(X_test, y_test, test_size=0.68, stratify=y_test, random_state=18) # This originally split the train set, not the test further.
    # ord split was was 0.60
    
    # print split parameters, sizes of split sets.
    
    print(f"Train is {len(X_train)} instances ({round(len(X_train)/len(X)*100)}% of org data) \n\
            Test is {len(X_test)} instances ({round(len(X_test)/len(X)*100)}% of org data) \n\
            Validation is {len(X_val)} instances ({round(len(X_val)/len(X)*100)}% of org data)")
    
    #4. return splits.
    assert len(X_test) + len(X_train) + len(X_val) == len(df_combined) , "The split ratios do not add up!"

    return X_test, y_test, X_train, y_train, X_val, y_val  
    #return train, validation, test
    

#%% ANCHOR Main call.

if __name__ == "__main__":
    
#%%
    qanon = pd.read_pickle('../../../Data/qanon/preprocessed/classify/prep_qanon.pkl')
    
    qanon.columns.name = "QANON"
    
    nonqanon = pd.read_pickle('../../../Data/nonqanon/preprocessed/classify/prep_nonqanon.pkl')
    
    nonqanon.columns.name = "NONQANON"
    
#%%    
    X_test, y_test,\
    X_train, y_train,\
    X_val, y_val = merge_split(qanon, nonqanon)
    
# %%
    
    # Write data splits to directories.
    
    train_folder = "../../../Data/train/"
    test_folder = "../../../Data/test/"
    val_folder = "../../../Data/validation/"
    
    #%%
    print(f'Saving train data and labels to {train_folder}')
    #with open(train_folder , "wb") as train:
    X_train.to_pickle(f'{train_folder}train.pkl')
    y_train.to_pickle(f'{train_folder}train_labels.pkl')
    
    print(f'Saving test data and labels to {test_folder}')
    #with open(test_folder , "wb") as test:
    X_test.to_pickle(f'{test_folder}test.pkl')
    y_test.to_pickle(f'{test_folder}test_labels.pkl')
    
    
    print(f'Saving validation data and labels to {val_folder}')
    #with open(val_folder , "wb") as validation:
    X_val.to_pickle(f'{val_folder}validation.pkl')
    y_val.to_pickle(f'{val_folder}val_labels.pkl')
    
    
    print("### DONE ! ###")

#%% ANCHOR Writing csv for Inspection



# %% ANCHOR get a random sample of 100 for inspaction

    q_sample = pd.read_csv('../../../Data/train/train_and_labels.csv').query('"Q" in label').sample(n=100)
    
    q_sample.to_csv('../../../Data/qanon/preprocessed/classify/q_sample_feat_inspection.csv', index=False)

    
# %% Data for Guy
    cols = ['id','text' , 'text_clean', 'tokens_clean', 'hashtags', 'emotes']
    old_train = pd.read_pickle()
    
    g_train = old['cols']
    
    old_val = pd.read_pickle()
    
    old_test = pd.read_pickle()