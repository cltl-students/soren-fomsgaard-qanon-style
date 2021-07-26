
#%% ANCHOR IMPORTS

#import pickle
#import multiprocessing
#from os import read

## Preprocessing modules
import re
#import nltk
#from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer, word_tokenize
#from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords
#import string
from string import punctuation
from collections import defaultdict, Counter
from ekphrasis.classes.spellcorrect import SpellCorrector

import emoji

#from gensim.parsing.preprocessing import remove_stopwords
import spacy
from spacy.tokenizer import _get_regex_pattern

# ekphrasis for social tokenization
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts import emoticons

#from spacy_langdetect import LanguageDetector # language detector for parsing telegram data.

from langdetect import detect


from num2words import num2words

# Use modin. Doesn't work under windows with dask engine.
#import modin.pandas as pd
import pandas as pd
import numpy as np
import swifter

## Anonymization
#import uuid
from utils.grasp import URL, deflood

#from anonymizedf.anonymizedf import anonymize
#from faker import Faker



#%% Read the data

def read_data(filepath , strict_lang='en'):
    """Read data in csv format in order to preprocess.

    Args:
        filepath (str): a filepath to a csv file with twitter data.
        strict_lang (str, optional): whether to select only tweets with explicit language metadata.Defaults to 'en'.
        
    Returns:
        a pandas DataFrame: [description]
    """
    
    
    data = pd.read_csv(filepath , names=["id", 
                                        "user",
                                        "language",
                                        "text",
                                        "date",
                                        "favs"])
    
    
    
    # Apply language selection if specified.
    if strict_lang != None:
        data = data.loc[data['language'] == strict_lang]
       
        
    # # drop duplicate tweets.    
    data.drop_duplicates(subset=['text'] , inplace=True)
    
    # # Anonymize mentions in tweets
    mention = re.compile("@\w+")
    
    data.text = data.text.str.replace(mention, '@USER')
    
    # # Anonymize urls in tweets.
    data.text = data.text.str.replace(URL, 'URL')
    
    return data

#data = pd.read_csv('../../../Data/NonQanon/non-qanon-feb-mar.csv', names=["id",
                                                                #    "user",
                                                                #    "language",
                                                                #    "text",
                                                                #    "date",
#                                                                   "favs"])

# %%

# find rows that langdetect doesn't like:


# texl70 = test['text']
# langdet = []
# for i in range(len(test)):
#     try:
#         lang = detect(texl70[i])
#     except:
#         lang = 'no'
#         print("This row throws error:", texl70[i])
#     langdet.append(lang)

# Select just the english subset (for the topic model only!)
#data = data.loc[data['language'] == 'en']


#%%[markdown]
## Check for empty tweets

#%%
#data['text'].isnull().values.any()

#%%
# Drop duplicates

#data.drop_duplicates(inplace=True)
#%%[markdown]
## Anonymize users - avoid this.

#%%
# Numerical method.
#data.assign(user=data.user.factorize()[0] + 1)


#%%
# Name method. Uses English names.
### adapted from https://stackoverflow.com/a/59929112 ###
#faker = Faker()

# Seed random generator
# Faker.seed(1881)

# anon_names = {name: faker.name() for name in data['user'].unique()}
# data['user'] = data['user'].map(anon_names)

### Acessed 18-02-2021 ###
#%%
# Anonymize mentions in tweets

#mention = re.compile("@\w+")


# use lambda, str.replace or something else?
#data.text = data.text.str.replace(mention, '@USER')


# Anonymize urls in tweets.

#data.text = data.text.str.replace(URL, 'URL')

#%%
# Return substitutions for inspection.
#subs = data[data['text'].str.contains('URL')]



#%%[markdown]

## Initialize objects for preprocessing.

#%% # ANCHOR SPACY MODEL(s)
# Load models and stopwords
# Load spacy to experiment with it

# English spacy model
english = spacy.load('en_core_web_sm')

# English stopwords
en_stopwords = spacy.lang.en.stop_words.STOP_WORDS

# Dutch spacy model
dutch = spacy.load('nl_core_news_sm')

# Dutch stopwords
nl_stopwords = spacy.lang.nl.stop_words.STOP_WORDS 

#%%


# Modify spacy's default tokenizer to ignore in word hyphens and hastags.

models = [english, dutch]

for model in models:
    ### Adapted from https://stackoverflow.com/a/58053407 ###
    # default pattern for tokens that do not get split
    re_token_match = _get_regex_pattern(model.Defaults.token_match)
    # add your patterns (here: hashtags and in-word hyphens)
    re_token_match = f"({re_token_match}|#\w+|\w+-\w+)"
    
    # overwrite token_match function of the tokenizer
    model.tokenizer.token_match = re.compile(re_token_match).match
    
    ### Acessed 23-02-2021 ###

#%%

# translator object for removeing punctation. Equvalient to re.subbing [^\w\s]
#translator = str.maketrans('', '', punctuation)

#stopwords = set(stopwords.words("english")) # NLTK

#stopwords = remove_stopwords(text) # Gensim

#stopwords = nlp.Defaults.stop_words default spacy stopwords
    
    
# Add extra stopwords here "user" is already included in spacy.
stop_list = [en_stopwords , nl_stopwords]    

for stops in stop_list:
    stops.update(['rt' , 'url'])
    

# Non-agressive stopwords

non_agg_stopwords = ['RT', 'URL']

#%%

# Ekphrasis preprocessing pipeline 

# text_preprocesor = TextPreProcessor(
    
#     # Less agressive normalization.
#     #omit = ['email', 'percent', 'money', 'phone', 'hashtag']
#     omit = ['hashtag', 'user'],
    
#     normalize = ['email'],
    
#     #annotate = { "allcaps", "elongated", "repeated", 'emphasis', 'censored'}
    
#     fix_html = True,
    
    
#     segmenter = "twitter",
#     corrector = "twitter",
    
#     unpack_hashtags = False,
    
#     unpack_contractions = False,
#     spell_correct_elong = False,
    
    
    
#     tokenizer = SocialTokenizer(lowercase=False).tokenize,
    
#     #dicts = [emoticons]
    

# )

#%%

# Just the social tokinzer from ekphrasis.
ek_tok_inc = SocialTokenizer(lowercase=False, emails=False).tokenize

ek_tok_ex = SocialTokenizer(lowercase=False, emojis=False, emoticons=False, emails=False, hashtags=False).tokenize # numbers=False
#%% #ANCHOR EXTRACT_EMOTES()

# Vactorizing these substitutions would be much faster.
# function to extract emojis
emoj_pat = re.compile(emoji.get_emoji_regexp())


econ_set = {econ for econ in emoticons.emoticons.keys()}

econ_table = {econ:"" for econ in econ_set}

htag_pat = re.compile(r"#\w+")

rt_pat = re.compile(r"\bRT\b" , re.M)

# Doesn't work
#econ_string = "|".join(econ_set)
#econ_regexp = re.compile(econ_string)

# This works function works but is very bad...

def mini_clean(instr):
    """Specifically remove emoticons, emojis and hashtags from a string.

    Args:
        input (str): a string to have emoticons removed.

    Returns:
        str: a copy of the string with emoticons, emojis and hashtags removed.
    """
    
    # regex subs.
    instr = re.sub(emoj_pat , "", instr) # emojis
    instr = re.sub(htag_pat , "" , instr) # hashtags
    instr = re.sub(rt_pat , "", instr) # retweets
    #input = re.sub(r"@USER[\W\S]?\w?", "", input, 
    #               re.IGNORECASE)  # subbed mentions
    instr = instr.replace("URL" , "") # subbed urls
    
    for con in econ_set:
        if con in instr:
            #print("match!")
            instr = instr.replace(con , "")
    return instr

#%%
# trying to vectorize
# see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html and https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.replace.html

# df.replace(to_replace={"text" : econ_table})

#%%
def extract_emotes(tokens):
    """Extract emojis and emoticons from a tokenized string.

    Args:
        tokens (list): A list of strings (tokens)

    Returns:
        tuple: A typle of two dictionaries to be unpacked.
    """
    
    # Using default dicts to count rightaway
    #emojis = defaultdict(int)
    #econs = defaultdict(int)
    
    # Alternative, two in one.
    #matches = {"emojis" : defaultdict(int),
    #          "emoticons" : defaultdict(int)}
    
    # Using lists. Handle counting later in stats.
    emojis =  []
    econs = []
    
    
    ## UNCOMMENT ##
    # Regexp for most common emojis. #Moved to global variable
    #emoj_pat = re.compile(emoji.get_emoji_regexp()) 
            
    # Set of specific emoticons.
    #econ_set = {econ for econ in emoticons.emoticons.keys()}   # moved to global variable.
    
    ## UNCOMMENT ##
            
    ### OLD CODE
    # Regexp for emoticons
    # grasp.EMOTICON # much sparser than ekphrasis
    
    #econ_pat = re.compile(r"[:;=B\*\-\(\)\[\]x0oOpPdD\#\<\>8\.'|\{\}\@=;:]+(?:(?=\s))", flags= re.U)
    
    # Alternative econ_pat, not as robust , adatped from https://stackoverflow.com/questions/28077049/regex-matching-emoticons:
    # (\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8Xx]'?[\-\^]?[3CcDdOoPpSs\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)
    
    ### accessed 16-03-2021
    
    ### OLD CODE ###
    
        
    for token in tokens:
        if re.match(emoj_pat , token):
            #emojis[token] += 1
            emojis.append(token)
        if token in econ_set:
            econs.append(token)
            #econs[token] += 1
    #emojis = re.findall(emoj_pat , string)
        
    #econs = re.findall(econ_pat , string)
    
    # if re.match()
    
    # emoji.emoji_count(emoj_pat)
    
    # make dict with emote:count
    
    
    
    return emojis , econs
    
    



# %% ANCHOR: PREPROCESS()
def preprocess(instr, normalize=True):
    """Preprocess a string.

    Args:
        instr (str): A string to be processed. 
        normalize (bool, optional): Indicate whether to apply (agressive) text normalization or not. Defaults to True.
    
    Returns:
        str: A Preprocessed copy of the input string.
    """
    
    
    
    
    
    if normalize == False:
        tokenized = ek_tok_inc(instr)
        
        #tokenized_clean = ek_tok_ex(instr)
        
        text_clean = mini_clean(instr)
        
        # Extract hashtags
        hashtags = [token.lower() for token in tokenized if token.startswith('#')]
        
            
        # Extract emojis and emoticons
        emotes = extract_emotes(tokenized)
        
    # Apply aggressive normalization for topic modeling.
    if normalize == True:
        
        # TODO, disable this / rewrite for consistency with read_data(strict_lang).
        # detect language
        #if detect(instr) == 'nl':
        # nlp = dutch
        # stopwords = nl_stopwords
        #else:
        nlp = english
        stopwords = en_stopwords

        
        ## Reduce repeated characters to max 3 using grasp.deflood(). Here instead under 'normalization' for computational reasons.
        deflooded = deflood(instr , n=3)
        
        ## Tokenize
        #tokenized = word_tokenize(no_punct.lower()) 
        
    
    
        ## Spacy  ##
        
        
        doc = nlp(deflooded) # This can be optimized.
        tokenized = [token for token in doc]
        
        
        ## Remove stopwords
        ## NLTK ##
        #rm_swrd = [word for word in tokenized if word not in stopwords]
        
        rm_swrd = [token for token in tokenized if token.norm_ not in stopwords]
        
    
        ## Normalization
        ## Remove punctuation
        #no_punct = instr.translate(translator) # not as reboust as re.sub
        
        #word_pat = re.compile(r'[^\w\s]' , re.M)
        #no_punct = re.sub(word_pat , ' ' , instr) # NOTE: The whitespace used to sub here is interpreted differently by NLTK versus spacy during tokenization.
        
        no_punct = [token for token in rm_swrd if token.norm_ not in punctuation]
        
        
        
        # Rejoin tokenized and normalized string.
        #text_norm = ' '.join([token.norm_ for token in no_punct if token.norm_.startswith# ('#') == False])
        
        ## Spelling correction
        
        
        # Lemmatize
        
        ## NLTK ## 
        #lemmatized = [lemmatizer.lemmatize(word) for word in rm_swrd]    
         
        ## Spacy ## Remove stopwords simultaneously. NOTE: Post-correction with regex match here to remove blanks and newlines.
        #lemmatized = [token.lemma_ for token in tokenized if token.norm_.lower() not in stopwords and re.match(r'\w' , token.lemma_)] 
        
        # Lemmatize. Store only lemmas that are word characters and longer than 1.
        
        lemmatized = [token.lemma_.lower() for token in no_punct if\
                                                                 (re.match(r'#?\w+', token.norm_, re.I) and len(token.norm_) > 1)
                                                                 ]
        # getting full adjectives should be [token.lemma_.lower() if condition 1 else token.nom_ if token.tag_ == 'JJ']... but it isn't.
                 
        # Get lemmas without hashtags                                                    
        lemmas = [lemma for lemma in lemmatized if lemma.startswith('#') == False]
    
        # Get hashtags separately - hashtags are always normalized, since that is inconsequential.
        hashtags = [word.norm_ for word in tokenized if word.norm_.startswith('#')]
    
    
        text_norm = ' '.join(lemmas)
    
    
    
    
    
    # add more logic for non agrress. preproc.
    # preserve punctuation, contractions
    # get emojis, emoticons
        
   
    if normalize == True:
        return text_norm , lemmas , hashtags
    if normalize == False:
    
        # A list of cleaned tokens.
        tokens_clean = [token for token in tokenized if\
                                                    token not in non_agg_stopwords\
                                                    and token not in emotes[0]\
                                                    and token not in emotes[1]\
                                                    and token.lower() not in hashtags]
        # Join cleaned string for obtaining ngrams etc. More efficient, but requires de-toknization to recover contractions.                                            
        #text_clean = ''.join([token for token in tokens_clean])   
            
    
        return text_clean, tokens_clean, hashtags, emotes
#%%

# Debug using slices of input data.
#test = data.iloc[205][3]

#test = data.iloc[200:220 , 3]

#print(test)

#result = [preprocess(text) for text in test]

test = "@DrDingus 😀 :) 🤔 we're more :D awake :-) are the most gloriously large. And FEAR is 😀 their favorite D: tool to hold people over their dominion. Those extremists are the darkest beings and will never change. It  is Victory of the Light! #GameOver #soyboi"

# You could do Series.apply(preprocess).agg(data['text']) for more of an overview.

#%%[markdown]
#Save the results for topic modeling.
#%%


#data['text'] = data['text'].apply(lambda x: preprocess(x)) # NOTE: This overwrites the input data.

#%%



#df_test = df_output[204:300]

#df_test['tokens'] , df_test['hashtags'] = zip(*df_test['text'].apply(lambda x: preprocess(x)))




#%%



#%%[markdown]
##Sampling and spliting data


#%% ANCHOR processing unrelated data to test for overfitting.

def prep_non_cons(infile, outfile):
    
    with open(infile, 'r', encoding='utf-8') as d:
        # read file contents
        rt = d.read()
        #text = re.split(r'\(\d+/\d+\)\s' , rt)
        # split lines
        lines = rt.splitlines(False)
        #texts, ids = map(list , zip(*(line.split(None, 1) for line in lines)))
        # collect ids and text.
        pt = defaultdict(str)
        for line in lines:
            i, sep, text = line.partition(' ')
            pt[i] = text
        
        # Put in DF
        o = pd.DataFrame(pt.items(), columns=['id', 'text'])
        
        # Add label
        o['label'] = 'NONQ' 
        
        # Apply non-aggressive preprocessing.
        o.text = o.text.swifter.apply(lambda x: preprocess(x, normalize=False)[0])
        
        o.to_csv(outfile, index=False)
    
#%%

# Uncomment for investigating baseline overfitting.
#prep_non_cons('../../../Data/noncons/converted/alice.txt',
#              '../../../Data/noncons/preprocessed/alice.csv')



#%% ANCHOR MAIN CALLS


# Driver code, try parallelizing:
if __name__ == "__main__":
    
    # Switches in lack of argparsing..
    dataset = "nonqanon_full" # 'qanon' or 'nonqanon'
    use_case = "classify" # 'topic' or 'classify'
    
    non_qanon = read_data(f'../../../Data/{dataset}/non-qanon-feb-mar - post collection.csv', strict_lang='en')
    
    qanon = read_data('../../../Data/qanon/qanon-dec-jan.csv' , strict_lang='en')
    
    #tg_chats = read_data(f'../../../Data/qanon/telegram/{dataset}', strict_lang=None)
    
    # Output data
    data = qanon # qanon or non_qanon above.
    df_output = data.copy(deep=True)
    
    #with multiprocessing.Pool(processes=3) as p:
        #data['text'] = data['text'].apply(lambda x: preprocess(x)) # NOTE: This overwrites the input data.
    #import multiprocessing.popen_spawn_win3
    #import modin.pandas as pd #
    #multiprocessing.freeze_support()
    
    # Label the data
    
    if dataset == 'qanon':
        df_output['cons'] = 'Q'
    else:
        df_output['cons'] = 'NONQ'
    
    # Apply aggressive preprocessing
    # This works 10-03-21 17:58
    if use_case == "topic":
        df_output['text_norm'],\
        df_output['tokens_norm'],\
        df_output['hashtags'] = zip(*df_output['text']\
                                    .swifter.apply((lambda x:\
                                    preprocess(x , normalize=True)))) 
        # Remove processed tweets with len < 1
        df_output['tokens_norm'] = df_output['tokens_norm'].apply(lambda x: np.nan if len(x) <= 1 else x)
        df_output.dropna(subset=['tokens_norm'], inplace=True)
    
    # Apply non-aggressive preprocessing
    if use_case == "classify":
        df_output['text_clean'] ,\
        df_output['tokens_clean'],\
        df_output['hashtags'] ,\
        df_output['emotes'] = zip(*df_output['text'].\
                                    swifter.apply((lambda x:\
                                    preprocess(x , normalize=False)))) 
        
            
        df_output['tokens_clean'] = df_output['tokens_clean'].apply(lambda x: np.nan if len(x) <= 1 else x)
        df_output.dropna(subset=['tokens_clean'], inplace=True)
        
    # Remove duplicates in output.
    df_output.drop_duplicates(subset=['text'] , inplace=True)
    
    # Remove tweets shorter than 1:
    #   
    # Write all the cleaned data (not telegram) to file
    df_output.to_pickle(f'../../../Data/{dataset}/preprocessed/{use_case}/prep_{dataset}.pkl')
        

    
