
#%%[markdown]  
# Imports
#%% ANCHOR IMPORTS
import logging
import logging.config 
import pprint
import json
import pickle
import pandas as pd


from nltk import bigrams

import multiprocessing
import joblib
# Gensim modules
from gensim import corpora, similarities
from gensim.models import Phrases, LdaModel, CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

# visualization
import pyLDAvis, pyLDAvis.gensim

# Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
# import matplotlib.pyplot as plt
#%matplotlib inline


#%%[markdown]



#%%[markdown]

#Read the preprocessed tweets as docs

def read_data(pickle_file, token_column='tokens_norm'):
    """Read in a pickle file with preprocessed (aggressively normalized) text.

    Args:
        pickle_file (str): a filepath to a pickle file with data.
        token_column (str , optional): the name of the column with normalized tokens. (defaults to 'tokens_norm') 
    Returns
        docs (list): list of strings (tokens) for each document (tweet) in the data file. 
    """

    data = pd.read_pickle(pickle_file)

    docs = data[token_column].values
    
    return docs
# %%[markdown]

##Compute n-grams.

# This is a to-do point. It is very computationally expensive.
#%%


# Could obtain bigrams like this?
#bgs = list(bigrams(docs))
#%%


### Taken from Gensim tutorial in the documentation @https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py ### 
# ANCHOR

def compute_corpus(docs, name='dataset'):
    """Compute a (bigram) dictionary, corpus and corresponding term-document frequencies on a list of documents. (tweets)

    Args:
        docs (list): list of strings containing normalized tokens from tweets.
        dataset (str, optional): Name of the dataset used. Defaults to "dataset". 

    Returns:
        corpus, id2word (tuple): a tuple with the corresponding corpus and its term-document freqs.
    """

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    
    bigram = Phrases(docs, min_count=20)
    
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token) 
    
    # You obtain trigram by calling the bigram model on 
    
    # Dictionary creation - the unique words / vocabulary of all the docs.
    dictionary = corpora.Dictionary(docs)
    
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    
    # Save dictionary for experimenting with different versions.
    if name != "dataset":
        dictionary.save(f'./models/topic/lda/dictionary/{name}_bigram.dict')
    
    
    
    # Bag-of-words representation of the documents - the corpus. 
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    
    
    
    
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    
    
    
    
    # Make a index to word dictionary for all term frequencies.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token # term-to-document frequency
    
    
    #print(type(corpus) , len(corpus), type(id2word), len(id2word))
    return corpus, id2word








#%%
# Training with multicore model
# use models.ldamulticore.LDAMulticore(workers=3) to parellize for speed.
### Adapted from tutorial at https://radimrehurek.com/gensim/models/ldamulticore.html ###

if __name__ == '__main__':
    
    dataset = 'nonqanon' # qanon or nonqanon.

    # Set up log to external log file
    logging.basicConfig(filename=f'./logging/models/lda/{dataset}/lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # Load data - the docs.
    data = read_data(f'../../../Data/{dataset}/preprocessed/topic/prep_{dataset}.pkl') 

    # Compute corpus and term frequencies.
    corpus, id2word =  compute_corpus(data, name=dataset)
    
#%%
    # Set model parameters.
    num_topics = 20
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    
#%%
# ANCHOR Train the model
    multiprocessing.freeze_support()
    model = LdaMulticore(
        workers=3,
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        num_topics=num_topics,
    )
    

# Singlecore.
    # model = LdaModel(
    #     corpus=corpus,
    #     id2word=id2word,
    #     chunksize=chunksize,
    #     alpha='auto',
    #     eta='auto',
    #     iterations=iterations,
    #     num_topics=num_topics,
    #     passes=passes,
    #     eval_every=eval_every
    # )
    #model.save('./models/topic/lda/qanon/10_topics_single_2.model')



### Accessed 22-02-2021 ### 


# Save model

    model.save(f'./models/topic/lda/{dataset}/{num_topics}_topics_multi_bigram.model')

#%%
# Inspect model topics.

    top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    
    from pprint import pprint
    pprint(top_topics)

# %%

# Save model topics
    with open(f'./exploring/topic/{dataset}/{num_topics}_topics_multi_bigram.txt' , 'w') as outfile:
        outfile.write(str(avg_topic_coherence) + str(top_topics))


#%%


