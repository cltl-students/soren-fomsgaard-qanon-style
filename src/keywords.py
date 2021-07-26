
#%%
import pandas as pd

import re
#from gensim.summarization import keywords
#from gensim.summarization.keywords import get_graph

#from RAKE import Rake

from collections import Counter

#import multiprocessing

#from preprocess import stopwords #import (spacy) stopwords used during preprocessing.

# visualizations

#import networkx as nx
import matplotlib.pyplot as plt
#from gensim.models import TfidfModel # only relevant per document/tweet. Or by pseudo documents

from wordcloud import WordCloud as wc
from PIL import Image
import numpy as np


#%%
# Topic - normalized
qanon_topic = pd.read_pickle(
    '../../../Data/qanon/preprocessed/topic/prep_qanon.pkl')

nonqanon_topic = pd.read_pickle(
    '../../../Data/nonqanon/preprocessed/topic/prep_nonqanon.pkl')


# Classification - non-aggressively normalized.
qanon_classify = pd.read_pickle(
    '../../../Data/qanon/preprocessed/classify/prep_qanon.pkl')

nonqanon_classify = pd.read_pickle(
    '../../../Data/nonqanon/preprocessed/classify/prep_nonqanon.pkl')

#%%[markdown]

#Get keywords, hashtags and emotes from a df.

def get_kws(data, normalized=True):
        #corpus = data.text.str.cat(sep=' ') # rejoin tweets after tokenization required by gensim module.
        
        
        # Concatenate all tokens in all tweets to single string.
        
        if normalized == True:
                #corpus = ' '.join(data.tokens_norm.str.join(sep=' ')) # joins rows twice (to themselves and their neighbors)
                corpus = ' '.join(data.tokens_norm.str.join(sep=' '))
                
                # Most frequent words: 
                top_words = Counter(corpus.split()).most_common(200)
        
        if normalized == False:
                #corpus = ' '.join(data.text_clean)
            
            # unpack the tuples in data.emotes. emotes[0] is always emojis, emotes [1] emoticons.
            # Otherwise, just count all emotes as one category..
            
            #emotes = ' '.join(data.emotes.str.join(sep=' '))
            emotes = pd.DataFrame(data.emotes.to_list() , columns=['emojis' , 'emoticons'])
            
            
            # Either join or .explode the values. -> count them.
            
            
            # count them.
            #top_emojis = Counter(emotes.emojis.split().most_common(200))
            top_emojis = emotes.emojis.explode().value_counts()
            top_emoticons = emotes.emoticons.explode().value_counts()
        
        #print(corpus[:500])
        
        
        # Concatenate all hashtags to single string - same for all data
        hashtags = ' '.join(data.hashtags.str.join(sep=' '))
        
        #%%
        
       
                
                
        # This becomes unnecessary
        #ht = re.compile(r'(#\b\S+)')
        
        
        #top_htags = re.findall(ht , corpus)
        
        #top_htags = (data.text.str.findall(ht) , data.text.str.count(ht))
        
        top_htags = Counter(hashtags.split()).most_common(200)
        
        
        
        if normalized == True:
                return top_words, top_htags
        
        if normalized == False:
                return top_emojis , top_emoticons, top_htags #, top_htags 
                

#%% 

#%%[markdown]

### Visualizations


#%%
# Wordclouds.


def wcloud(data, name='dataset', mask=None):        
        """Draw a wordcloud given list of tokens with frequencies.
        
        Args:
                data (list): List of tuples with [('token' , frequency)].
                name (str, optional): Name for the output image. Defaults to "dataset".
                mask (str, optional): Path to image to use as mask. Defaults to None.
        """            
        

        # Cast input list of tuples to dict.
        d = dict(data)
        
        # Add image mask
        mask = np.array(Image.open(mask))
        
        cloud = wc(width=800, height=600, background_color='white' ,max_words=50, mask=mask).generate_from_frequencies(d)
        #wc.generate_from_frequencies(cloud, data)
        
        # Show wordcloud.
        plt.figure(figsize= (20, 10))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
        
        # Save to file.
        cloud.to_file(f'./exploring/wordcloud/{name}.png')
#%%




#%%[markdown]

## Main call

if __name__ == "__main__":
        
        dataset = 'qanon' # nonqanon or qanon.
        
        
        
        # qanon
        top_words, top_htags = get_kws(qanon_topic, normalized=True)
        top_emojis, top_emoticons = get_kws(qanon_classify, normalized=False)

        
        # nonqanon
        #top_words, top_htags = get_kws(nonqanon_topic, normalized=True)
        #top_emojis, top_emoticons = get_kws(nonqanon_classify, normalized=False)
        
        #print(data.columns)
        
        with open(f'./exploring/keywords/{dataset}_keywords.txt' , 'w') as kw_file,\
                open(f'./exploring/keywords/{dataset}_hashtags.txt', 'w') as ht_file,\
                open(f'./exploring/keywords/{dataset}_emojis.txt' , 'w') as emoj_file,\
                open(f"./exploring/keywords/{dataset}_emoticons.txt" , "w") as emot_file:
                kw_file.write('\n'.join(str(item) for item in top_words))
                ht_file.write('\n'.join(str(item) for item in top_htags))
                top_emojis.to_csv(emoj_file)
                top_emoticons.to_csv(emot_file)

#%%
# Generate wordclouds for qanon, nonqanon data.
        
# Qanon
        # Keywords
        wcloud(get_kws(qanon_topic)[0], name="Qanon_kws",
               mask="./exploring/wordcloud/letter_Q.png")
        
#%%
        # # Hashtags
        wcloud(get_kws(qanon_classify, normalized=False)[2], name="Qanon_htags",
               mask="./exploring/wordcloud/letter_Q.png")
#%%
# Nonqanon
        # Keywords
        wcloud(get_kws(nonqanon_topic)[0], name="Nonqanon_kws",
               mask="./exploring/wordcloud/letter_N.png")
        
#%%
        # Hashtags
        wcloud(get_kws(nonqanon_classify, normalized=False)[2], name="Nonqanon_htags",
               mask="./exploring/wordcloud/letter_N.png")

# %%
