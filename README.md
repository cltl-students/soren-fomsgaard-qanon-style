# HLT Thesis Code
 
# By Søren Kirkegaard Fomsgaard, July 26, 2021.

This is the code used during my research on the style of QAnon on Twitter, as part of the thesis: "In the eye of the storm with style: Investigating style features in the language of QAnon on Twitter" as part of the program Human Language Technology at the CLTL, Vrije Universiteit, Amsterdam.




## Dependencies

To use my code, install the required dependencies using pip: pip install -r requirements.txt --- and (if using conda): conda env create -f environment.yml


## Structure

The repo is structured as follows:
```
├───collecting <- collects data.
├───exploring <- explores the data (stats, keywords and topics)
│   ├───keywords
│   ├───stats
│   ├───topic
│   │   ├───nonqanon
│   │   ├───qanon
│   │   └───visualization
│   └───wordcloud
├───models <- stores trained classifiers.
│   ├───baseline
│   │   ├───nb
│   │   │   ├───results
│   │   │   └───unscaled
│   │   ├───overfit
│   │   ├───randomforest
│   │   │   ├───old_splits
│   │   │   ├───results
│   │   │   └───unscaled
│   │   └───svm
│   │       ├───old_splits
│   │       ├───results
│   │       └───unscaled
│   ├───classifier
│   │   ├───development
│   │   │   ├───classifier_base_case_no_posgrams
│   │   │   │   └───model
│   │   │   ├───classifier_base_case_with_posgrams
│   │   │   │   └───model
│   │   │   ├───classifier_full_case_no_posgrams
│   │   │   │   └───model
│   │   │   └───classifier_full_case_with_posgrams
│   │   │       └───model
│   │   ├───final
│   │   │   ├───classifer_base_case_with_posgrams
│   │   │   ├───classifier_base_case_no_posgrams
│   │   │   │   └───model
│   │   │   ├───classifier_base_case_own_features
│   │   │   │   └───model
│   │   │   ├───classifier_base_case_without_own_features
│   │   │   │   └───model
│   │   │   ├───classifier_base_case_with_posgrams
│   │   │   │   └───model
│   │   │   ├───classifier_full_case_no_posgrams
│   │   │   │   └───model
│   │   │   ├───classifier_full_case_own_features
│   │   │   │   └───model
│   │   │   ├───classifier_full_case_without_own_features
│   │   │   │   └───model
│   │   │   └───classifier_full_case_with_posgrams
│   │   │       └───model
│   │   └───sample
│   │       └───sample
│   ├───evaluation <- gets samples for qual. error inspection.
│   │   └───errors
│   └───topic <- evaluates topic model.
│       └───lda
│           ├───dictionary
│           ├───nonqanon
│           └───qanon
└───utils
    ├───collect
    └───preproc
```

## Data

In order to run my code, you need some conspiratorial data, some non-conspiratorial data, the modified LIWC 2007.xlsx as well as a copy of Alice in Wonderland.txt placed like so:
 placed in the following directory structure:

(relative to the source code folder of this repo:)
```
src/../../
│   LIWC2007dictionary poster.xls <- put the LIWC file here.
│
├───LIWC
│   │
│   ├───for debugging
│   │
│   └───train_only
│
├───noncons 
│   ├───converted
│   │       alice.txt
│   │
│   ├───preprocessed
│   │       alice.csv
│   │       alice_with_Q.csv
│   │
│   └───source <- put alice in wonderland here.
│           alice_in_wonderland.txt
│
├───nonqanon <- put NONQANON / non conspiratorial data here + a list of accounts to crawl in .csv
│   │   non-qanon-feb-mar.csv
│   │   NonQanon twitter accounts.csv
│   │   │
│   │   └───post collections
│   │           
│   │
│   └───preprocessed
│       ├───classify
│       │   │  
│       │   └───backups
│       │          
│       │
│       └───topic
│             
│
│
├───qanon <- put Qanon / conspiratorial data here.
│   │   qanon-dec-jan.csv
│   │
│   ├──preprocessed
│      ├───classify
│      │
│      └───topic
│
├───test
│
├───train
│       
└───validation
```

## Usage

1. All scripts are run from the source code dir (/src/) of the repo.
2. Data collection is done by running: 
   
    `python -m collecting -user_csv "../../../Data/NonQanon/NonQanon twitter accounts.csv" -handle_col 1 -save_file "../../../../Data/NonQanon/NONQANON.csv"`
3. The data is by running preprocess.py
4. ...and split by running data_split.py
5. The data sets can be explored by running data_exploration.py
6. The topic model is trained by running topic_model.py
7. All the classification experiments are run using model.py

### Notes
* I have removed all sensitive data, including the models, since it is possible to extract information about the data from them.
* You need to adapt the file paths to whatever data you are using.


