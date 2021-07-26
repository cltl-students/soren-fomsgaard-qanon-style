

#%%
# ANCHOR IMPORTS
import sys
import pandas as pd, numpy as np
import pickle

import re

from sklearn import feature_extraction , feature_selection

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer

from tqdm.autonotebook import trange, tqdm
import swifter

# Libraries for feature engineering.
import string
from collections import Counter # not necessary?
#from nnsplit import NNSplit
import spacy# .tokenizer.tokenize
from spellchecker import SpellChecker

# Other neat features.
from nltk.metrics.distance import edit_distance
from lexicalrichness import LexicalRichness
import syllables
import itertools
import textstat
# Stats
from scipy.stats import chisquare
#from statistics import mean

#%% Get spacy docs and save them to data to speed up development.


def get_docs(data, text_col='text_clean'):
    nlp = spacy.load('en_core_web_sm')
    nlp.enable_pipe("senter")
    
    data['docs'] = data[tect_col].apply(lambda x: nlp(x))


#%%
def listify(series, feature_name=str): 

    return [{feature_name: x[1]} for x in series.items()]

#%%
# Extract Baseline feature

    # Character trigrams (morphological/lexical/semantic?). 
    
def ngrams(train, test, params): 
    """Extract character ngrams.

    Args:
        train (list): list of texts to fit the vectorizer.
        test (list): list of texts to transform to feature space.
        params (dict): parameters for the vectorizer construction

    Returns:
        [type]: [description]
    """    
    
    vectorizer = CountVectorizer(lowercase=params['ngrams']['lowercase'],
                                 ngram_range=params['ngrams']['size'], # experiment with ranges, e.g. ngram_range=(3,3)
                                 analyzer=params['ngrams']['type'], #, also try "char_wb" 
                                 max_features=params['ngrams']['max_vocab']) # max_features=10000
                                 
    # fit count vecotorizer to preprocessed tweets.
    #vectorizer.fit(train)
    
    
    # Transform into input vectors for train and test data.
    train_vectors = vectorizer.fit_transform(train) # using fit_transform due to better implementation.
    
    #train_vectors = vectorizer.transform(train) #.toarray()
    
    test_vectors = vectorizer.transform(test) #.toarray()
    
    
    
    # Inspect with vectorizer.get_feature_names() and .toarray()
    #inverse = vectorizer.inverse_transform(train)
    #feature_names = vectorizer.get_feature_names()
    
    #print(f'Train ({type(train_vectors)}) feature matrix has shape: {train_vectors.shape}')
    #print(f'Test ({type(test_vectors)}) feature matrix has shape: {test_vectors.shape}')
    
    
    #return vectorizer
    return vectorizer, train_vectors , test_vectors
    #return inverse


#%% ANCHOR EXTRACT LIWC

def parse_liwc(file, **args):
    """Parse a (left) aligned version of the LIWC lexicon.

    Args:
        file (str): filepath to lexcion (excel).

    Returns:
        DataFrame: df or dict
    """    

    df = pd.read_excel(file, skiprows=2)
    
    
    # Handling merged columns in file
    
    ### Adapted from  https://stackoverflow.com/a/64179518 ###
    
    df.columns = df.columns.to_series()\
        .replace('Unnamed:\s\d+', np.nan, regex=True).ffill().values
    
    # Multindex to represent multiple columns for some categories.
    df.columns = pd.MultiIndex.from_tuples([(x, y)for x, y in
                                            zip(df.columns, df.columns.to_series().groupby(level=0).cumcount())])
    
    ### Accessed 26-04-2021 ###
    # d = data.to_dict(orient='list')
    
    ### Adapted from https://stackoverflow.com/a/50082926
    # dm = data.melt()
    
    # data = dm.set_index(['variable', dm.groupby('variable').cumcount()]).sort_index()['value'].unstack(0)
    
    
    
    ### Accessed 26-04-2021 ###
    
    # Concat the terms by column.
    # d = dict()
    
    #d = {column: value for key, value in dd.items()}
    
    # for ki, wl in dd.items():
    #     nl = []
    #     k, i = ki
        
    #     # for w in wl:
    #     #     if w not in nl:
                
        
    #     # d[k].append(wl)
    
        
    #     if k in d:
    #         d[k].append(wl)
    #     else:
    #         d[k] = wl
    ### Solution from https://stackoverflow.com/a/48298420 ###
    # TODO experiment with not sorting the index? or reesrorting columns to mach the multiindex or just original df.columns.
    df = df.stack().sort_index(level=1).reset_index(drop=True)
    
    ### Accessed 26-04-2021 ###
    
    # Check that merged columns have the right number of terms.
    # sum(isinstance(x, str) for x in terms['Funct'])
    
    return df.to_dict(orient='list')
    


#%%
# Extract LIWC matches (lexical/semantic)
def liwc_match(parsed, d, extract=False, text_col='text_clean'):
    """Search a corpus for matches against LIWC (2007) categories.

    Args:
        parsed (DataFrame): a pandas df with the all categories of LIWC prepared. 
        d (str): a filepath to a pickle file with a corpus to search.
        extract (bool, optional): Switch specifying whether or not to return a Dict for feature extraction or feature inspection/analysis. Defaults to False.

    Returns:
        dict: a dict with {liwc_cat1...n : count} for each datapoint in the corpus OR a dict a, a dataFrame and a Series with results of searching the categories against the matches (absolute counts per datapoint (as dict and DF) totals per category (Series)).
    """    
    # load data to search.
    # Could do Series.count(regex) or df[clean_text] -> (joined) list?
    if isinstance(d, pd.DataFrame) == False: # the ... analysis case.
        data = pd.read_pickle(d)  
        
        text = list(d) # a single row/tweet?
    if extract == True: # The extract case
        data = d
        text = data[text_col]
    
    
    
    # Dict for search results.
    results = dict()
    pats = dict() # save patterns to dict for debugging.
    # Loop through category-termlist pairs.
    for cat, terms in tqdm(parsed.items()):
        
        # Remove nans from term lists.
        terms = [term.strip(' ') for term in terms if isinstance(term, str)]
        
        
        # Compile re pattern from term list.
        #pat = re.compile('|'.join(terms), flags=re.MULTILINE) 
        
        #pat = re.compile('|'.join(
        #    [r'\b' + t[:-1] if t.endswith('*') else r'\b' + t + r'\b' for t in #terms]))
        
        ### Adapted from https://stackoverflow.com/a/65140193 ###
        pat = re.compile('|'.join([r'\b' + t[:-1] + r'\w*' if t.endswith('*') else r'\b' + t + r'\b' for t in terms]) , flags=re.MULTILINE | re.IGNORECASE)
        
        ### Accessed 27-04-2021 ###
        
        pats[cat] = pat
        
        #i, char = enumerate(j_terms)
        # for term in terms:
        #     i = 0
        #     try:
        #         pat = re.compile(term) 
        #         #print(pat, counter,'\n')
        #         i +=1
        #     except:
        #         print('error here:\n'.upper(),pat, i)
                
        
        
        # Aggregate matches per category into dict. storing tweet id's preserved in the source data.
        #results[cat] = pat.finditer(text.values)
        # For that, join values into list of lists -> re.match -> see below
        # results[cat][re.match(pat)] = re.finditer(pat, row_list)  
        # if extract == True: You can't normalize since this isn't tokenized.
        #     results[cat] = text.apply(lambda x: x.str.count(pat) / len(x))
        
        # else:
        results[cat] = text.str.count(pat)
        
        
        #results[cat] = text.swifter.apply(lambda x: re.finditer(pat, x))
    
    
    # Store results in DataFrame
    df_results = pd.DataFrame.from_dict(results)
    
    # Totals per category
    df_totals = df_results.sum().sort_values(ascending=False)
    
    
    if extract == True:    
        # Export results to {index : {cat : count}...} for easy vectorization.
        results_per_row = df_results.to_dict(orient='records')  # or orient='index'?  -> DictVectorizer
        
        return results_per_row
        
        
    return {'results' : 
                {'matches_dict' : results,
                'matches_df' : df_results,
                'matches_total': df_totals
                },
            'regex_pats' : pats   
            } 

#%%
def norm_freqs(data, expression, count_name=str, normalize=True, analyze=True):
    """Get frequencies (normalized = optional) of a regex pattern in a Series with one or more strings. 

    Args:
        data (DataFrame): a dataframe with texts to extract frequencies from.
        expression (re.compile): a regex pattern to count occurrences of in each text.
        count_name (str, optional): a name for the counted feature. Defaults to str.
        normalize (bool, optional): [description]. Defaults to True.

    Returns:
        list: list of dicts with key = frequency name, value = frequency.
    """
    # List to store frequencies
    # freqList = list()

    # Loop through each entry in the list of strings.
    # for e in stringList:
    #     # Join to a regular string
    #     text = ' '.join(e)

    #     # Construct a dict for each entry with freuncies.
    #     c = {count_name : len([char for char in text if char in expression])}

    # Get frequencies of a regex in a pandas column, normalize if set to True.
    c = data.apply(lambda x: len(re.findall(
        expression, x))/len(x) if normalize == True else len(re.findall(expression, x)))

    ### Adapted from https://stackoverflow.com/a/45452966 ###

    # Cast frequencies Series to list of dicts.
    cList = [{count_name: x[1]} for x in c.items()]

    ### Accessed 10-05-2021 ###
    if analyze == True:
        return cList
    else:
        return c
        
def binary_freq(data, expression, feature_name=str, analyze=True):
    """Search data for occurrences of a binary feature as a regex.

    Args:
        data (pd.Series): a series with text instances.
        expression (re.compile): a regex or string to search for. 
        feature_name (str, optional): a name for the feature to extract. Defaults to str.

    Returns:
        list: a list with a dict mapping feature name to 1 or 0 (true/false) based on occurrence in texts.
    """

    b = data.str.contains(expression).astype(int)  # cast bools to 0/1
    
    if analyze == True:
        bList = [{feature_name: x[1]} for x in b.items()]
        
        return bList

    else:
        return b
#%% ANCHOR extract character and word level features
        
# Extract character-level features (lexical/morphological).
def get_cl(data, text_col='text_clean', analyze=True):
    
    

    
    # 0. Cast data text col .to_list()
    
  
            
            
    # 1. Normalized punctation frequency.
    
    # # Using pandas instead of lists + counter + dicts.    
    # df_results = pd.DataFrame({'text': textList})
    
    # #p_pat = re.compile(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*')    
    # p_pat = re.compile(re.escape(string.punctuation))
    
    # df_results['punct'] = df_results.text.str.count(p_pat)
    
    # the whole series
    #train['text_clean'].str.count(p_pat)
    
    
    df_punc_freq = data[text_col].apply(lambda x: len([char for char in ' '.join(x) if char in string.punctuation]) / len(' '.join(x)))
    
   
    
    #return punc_freq, df_punc_freq
    
    #df_punc_freq = pd.DataFrame.from_records(punc_freq)
    # Add to cl dict.
    #cl_results['punc_freq'] = punc_freq
    
    #2. Specific characters (also normalized)
    
    # 2.1 digits
    
    d_pat = re.compile(r'\d' , re.M)
    
    
    
    df_digits = norm_freqs(data[text_col], d_pat, count_name='digit_freq',normalize=True, analyze=False)
    #return df_digits
    
    # 2.2 Whitespace chars.
    ws_pat = re.compile(r' ', re.M) # NOTE just using actual whitespace instead of \s
   
    
    df_whitespaces = norm_freqs(data[text_col], ws_pat, count_name='whitespace_freq', normalize=True, analyze=False)
    
    # 2.3 tab characters NOTE Doesn't occur in either corpus.
    # tab_pat = re.compile(r'\t', re.M)
    # tabs = norm_freqs(data[text_col], tab_pat, count_name='tab_freqs', normalize=True)
    
    # 2.4 line break characters
    br_pat = re.compile(r'[\r\n\f]', re.M)
   
    
    df_lbreaks = norm_freqs(data[text_col], br_pat, count_name='line_break_freq', normalize=True, analyze=False)
    
    # 2.5 Upperchase chars (per all chars)
    up_pat = re.compile(r'[A-Z]', re.M) # Decide whether to be greedy about *all* uppercase chars or to be lazy (below). Also, @USER mentions are counted now. Can be excluded with \b(?!USER\b)[A-Z]. Try doing [^a-z\W] - caret negates the range of chars.
    
    #up_pat = re.compile(r'(?<![a-z])*[A-Z](?![a-z])*' , re.M) # Only count chars if they are not a one-off in the beginning of words.
    
    
    
    df_upchars = norm_freqs(data[text_col], up_pat, count_name= 'upper_char_freq', normalize=True, analyze=False)
    
    # 2.6 Special chars other than punctuation. NOTE Doesn't make much sense when using a full punctuaion set..
    spc_pat = re.compile(r"[^a-z \.,!?':;\s]", re.M)
   
    
    df_spc = norm_freqs(data[text_col], spc_pat, count_name="special_characters", analyze=False) 
    
    #3. Repeated characters (binary features) # NOTE if you want counts of each repeated char, consider just defining it with regexes and then using norm_freqs, normalize=False?
    
    # 3.1 question marks
    quest_pat = re.compile(r'\?{2,}', re.M)
   
    
    df_rep_quest = binary_freq(data[text_col] , quest_pat, feature_name='quest_rep', analyze=False)
     
    
    # 3.2 periods (ellipsis)
    per_pat = re.compile(r'\.{2,}', re.M)
   
    
    df_rep_per = binary_freq(data[text_col] , per_pat, feature_name='period_rep', analyze=False)
    
    
    # 3.3  exclamation marks
    excl_pat = re.compile(r'!{2,}', re.M)
   
    
    df_rep_excl = binary_freq(data[text_col] , excl_pat, feature_name='excl_rep', analyze=False)
    
    # 4 Contains equal signs
    eq_pat = re.compile(r'=', re.M)
   
    
    df_equals = binary_freq(data[text_col] , eq_pat , feature_name='equals', analyze=False)
    
    # 5 Quotes in chars
    
    #quotes = data[text_col].apply(lambda x: len(re.findall(quot_pat, x)) / len(x)) # per character --- works.
    #quotes_char = [{'quotes' : x[1]} for x in qoutes.items()]
    
    
    if analyze == True:
        #punc_freq = listify(df_punc_freq, feature_name='char_punc_freq') # new Alternative to punc_freq with dict comprehension.
        textList = data[text_col].to_list()
    ### Old approach to punc_freqs for analysis.
        cl_results = dict() # dict to store results.
        punc_freq = list()
        for e in textList:
            text = ' '.join(e)
            # Build dict with counts of all punct characters. 
            # The first c example does it per punctuation character, the second for all.
            # Each count is normalized by total number of chars in the each string.
            # NOTE not using regexes here. Single quotes/apostrophes/contractions are counted as well.
            #c = {char:count/len(text) for char, count in Counter(text).items() #if char in string.punctuation}
            
            # This should generalize to regex matches.
            c = {'char_punc_freq': len([char for char in text if char in string.punctuation])/len(text)}     
            punc_freq.append(c) 
        
        digits = norm_freqs(data[text_col], d_pat, count_name='digit_freq',normalize=True)
        whitespaces = norm_freqs(data[text_col], ws_pat, count_name='whitespace_freq', normalize=True)
        lbreaks = norm_freqs(data[text_col], br_pat, count_name='line_break_freq', normalize=True)
        upchars = norm_freqs(data[text_col], up_pat, count_name= 'upper_char_freq', normalize=True)
        spc = norm_freqs(data[text_col], spc_pat, count_name="special_characters") 
        rep_quest = binary_freq(data[text_col] , quest_pat, feature_name='quest_rep')
        rep_per = binary_freq(data[text_col] , per_pat, feature_name='period_rep')
        rep_excl = binary_freq(data[text_col] , excl_pat, feature_name='excl_rep')
        equals = binary_freq(data[text_col] , eq_pat , feature_name='equals')
        
        # Store results
        cl_results['char_punc_freq'] = punc_freq
        cl_results['digit_freq'] = digits 
        cl_results['whitespace_freq'] = whitespaces 
        #cl_results['tab_freq'] = tabs does not occur in either corpus.
        cl_results['linebreak_freq'] = lbreaks
        cl_results['uppercased_char_freq'] = upchars
        cl_results['special_char_freq'] = spc
        
        cl_results['repeated_questionmark'] = rep_quest
        cl_results['repeated_periods'] = rep_per 
        cl_results['repeated_exclamation'] = rep_excl
        cl_results['contains_equals'] =  equals
        
        
        
       
        return cl_results #punc_freq  # (punc_freq , cl_results) 
    
    # Store results as df for much easier vectorization...
    else:
        cl_results_df = pd.DataFrame()
        
        cl_results_df['char_punc_freq'] = df_punc_freq #✅
        #pd.concat(cl_results_df)
        
         # Store results
        
        cl_results_df['digit_freq'] = df_digits #✅
        
        
        cl_results_df['whitespace_freq'] = df_whitespaces #✅
        #cl_results['tab_freq'] = tabs does not occur in either corpus.
        
        cl_results_df['linebreak_freq'] = df_lbreaks #✅
        cl_results_df['uppercased_char_freq'] = df_upchars #✅
        cl_results_df['special_char_freq'] = df_spc #✅
        
        cl_results_df['repeated_questionmark'] = df_rep_quest #✅
        
        cl_results_df['repeated_periods'] = df_rep_per #✅
        cl_results_df['repeated_exclamation'] = df_rep_excl #✅
        cl_results_df['contains_equals'] =  df_equals #✅
    
    return cl_results_df        
#%%
# Debugging 
# test_df = train.iloc[:50,:]
# test = get_cl(test_df, text_col='text_clean', analyze=False)


# Extract word-level features (lexical/morphological)
def get_wl(data, text_col='text_clean', analyze=False, docs=[]): 
    
    # SpaCy pipe for rule based sentence splitting.
    #blank_nlp = spacy.blank('en')  # spacy.load('en_core_web_sm')
    # sentencizer = blank_nlp.add_pipe("sentencizer")
    # morphologizer = blank_nlp.add_pipe('morphologizer')
    # blank_nlp.initialize() # 
    # print(nlp.pipe_names)
    print('Configuring spacy for word level')
    nlp = spacy.load('en_core_web_sm', disable=["lemmatizer", 'ner'])
    
    # disable parser in favor of senter and sentencizer due to speed https://spacy.io/models
    nlp.disable_pipe("parser")
    nlp.enable_pipe("senter")
    
    # Load spellchecker
    spell = SpellChecker()
    # load exceptions to spellchecker (Twitter, covid specifc)
    try:
        spell.word_frequency.load_text_file('./utils/spell_additions.txt')
    except:
        pass
    
    
    
    # 1 Get lengths (total/avg words, sentence)
    #  rewrite features as attributes of Lengths objects?
    # class Lengths:
    #     def __init__(self, first_feat, second_feat):
    #         pass
            
    #textList = data[text_col].to_list()
    
    wl_results = dict()
    
    # print('TOKENIZING WORD-LEVEL FEATURES')
    # data to docs
    if len(docs) <= 0:
        
        docs = data[text_col].swifter.apply(lambda x: nlp(x))
        
    #assert len(docs) == len(data[text_col])
    
    # get list of sentences.
    sents_c = docs.apply(lambda x: [s for s in x.sents])   
    
    
    # Words only (including numbers and @mentions)
    sents_w = docs.apply(lambda x: [[t.text for t in s if\
                                        t.is_punct == False and
                                        t.is_space == False]\
                                        for s in x.sents])
    
        
    # list of *word* tokens in entire tweet.
    toks = docs.apply(lambda x: [t.text for t in x if t.is_punct == False and\
                                                        t.is_space == False]) # could have used data['tokens_clean]
    # alphabetic tokens only. (for spell checking)
    toks_alpha = docs.apply(lambda x: [t.text for t in x if t.is_alpha == True])
    
    # Debugging getting empty lists of alphabetic tokens.
    #return pd.DataFrame({'tokens' : toks, 'alpha_tokens': toks_alpha})
    toks_morph = docs.apply( lambda x: [t for t in x if t.is_alpha == True])    
        
        
    # print('\n GETTING WORD-LEVEL FEATURES')
    # 1.1 total length of tweet in words
    # c = {'total_words' : int}
    # for doc in docs:
    
    w_total_series = toks.map(len)
    
    # 1.2 avg word length
    awl = toks.apply(lambda x: sum(len(w) for w in x) / len(x))
   
    
    
    
    # build dict with keys from list contained in feature_params value for lexical features > word_level. Check if they are there and populate them with the dicts below accordingly. Else don't.
    
    
    # 1.3.1 avg sentence length (words)
    asl_w = sents_w.apply(lambda x: sum(len(s) for s in x) / len(x))
    
    
    # 1.3.2 avg sentence length (characters)
    #asl_c = apply(lambda x: sum([len(''.join(s.text)) for s in x])) 
    asl_c = sents_c.apply(lambda x: sum(len(''.join(s.text)) for s in x) / len(x)) 
    
    
    # 2.1 number of uppercased words.
    uws = toks_alpha.apply(lambda x: len([t for t in x if t.isupper() == True]) / len(x) if len(x) > 0 else 0.0)
    
    
    
    # 2.2 number of short words 
    # use len of token <=3
    sws = toks_alpha.apply(lambda x: len([t for t in x if len(t) <=3]) / len(x) if len(x) > 0 else 0.0)
    
    
    # 2.3 number of elongated words 
    # use regex \b\w{3,}\b
    elw_pat = re.compile(r'(\w)\1{2,}', re.M)
    elws = toks_alpha.apply(lambda x: len([t for t in x if elw_pat.search(t)]) / len(x) if len(x) > 0 else 0.0)
    
    
    # 2.4 number of number-like tokens (both digits and numerals)
    nss = docs.apply(lambda x: len([t for t in x if t.like_num == True]) / len(x))
    
    
    # 2.5 frequency of specific verb tenses
    pst = toks_morph.apply(lambda x: [t.morph for t in x if t.morph.get('Tense') == ['Past']]).map(len).divide(toks_alpha.map(len))
    
    
    prs = toks_morph.apply(lambda x: [t.morph for t in x if t.morph.get('Tense') == ['Pres']]).map(len).divide(toks_alpha.map(len)) #NOTE using series.divide instead for if/else check with regular might give a problem with vectorizers.
    
    
    adj_pos = toks_morph.apply(lambda x: [t.morph for t in x if t.morph.get('Degree') == ['Pos']]).map(len).divide(toks_alpha.map(len))
    
    
    adj_c_s = toks_morph.apply(lambda x: [t.morph for t in x if t.morph.get('Degree') == ['Cmp'] or t.morph.get('Degree') == ['Sup']]).map(len).divide(toks_alpha.map(len))
    
    
    
    # Here you could add future tense, mood etc.
    
    # 2.6 Frequency of OOV words (according to spaCy model)
    # token.is_oov
    
    # 3. Frequencies of emotes/jis. 
    e = data['emotes'].apply(lambda x: len(x[0] + x[1])).divide(toks.map(len)) # normalized by tokens.
   
    
    
    # 4. Non-standard spelling.  Reconsider including this. It mostly captures proper names and acronyms if it has to be this fast.
    sc = toks_alpha.apply(lambda x: spell.unknown(x)).map(len).divide(toks_alpha.map(len))
    
        
    # 5. number of quoted words
    # NOTE normalized by words (in match / in tweet)
    quot_pat = re.compile(r"(\".+?\"|\B'.+?'\B)") # should this be quot_pat = re.compile(r("\".+?\"|\B'.+?'\B")) # 
    
    #quotes = data[text_col].apply(lambda x: re.findall(quot_pat, x).split(' ')).map(len).divide(toks_alpha.map(len)) # per word (split on whitespace).

    print('Tokenizing quote spans')
    quotes = data[text_col].swifter.apply(lambda x:
                                    [t for t in nlp(' '.join(re.findall(quot_pat, x))) if t.text.isalnum()]).map(len).divide(toks.map(len))
    
    
    #return pd.DataFrame({'org_text': data[text_col],'alpha_toks': toks_alpha, 'quoted_toks' : quotes, 'quoted_lens' : quotes_lens})
    
    #quotes = data[text_col].apply(lambda x: re.findall(quot_pat, x)).map(len).divide(toks_alpha.map(len)) # not finished. need to tokenize matches.
            
    #quotes = sents_c.apply(lambda x: len([re.findall(quot_pat, s) for s in x]) / len(x))# per sentence - doesn't work.
    
   
    
    
    # 6. Vocab richness/complexity
    # 6.1 Type-token ratio.
    tt = toks_alpha.apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0.0) # could use Counter instead of set()
    
    
    # 6.2.1 Hapax legomena
    ### Adapted from https://stackoverflow.com/a/1801676 ###
    hlg = toks_alpha.apply(lambda x: len([word for word, count in Counter(map(str.lower, x)).items() if count == 1]) / len(x) if len(x) > 0 else 0.0) # could also lower with list comprehension.
    ### accessed 13-05-2021 ###
    
    
    # 6.2.2 Hapax dislegomena (words that occur twice only)
    hdlg = toks_alpha.apply(lambda x: len([word for word, count in Counter(map(str.lower, x)).items() if count == 2]) / len(x) if len(x) > 0 else 0.0)
    
    
    # Here you would implement complexity measures 
    #- Brunet's W Measure
    #- Yule's K Characteristic
    #- Honore's R Measure
    #- Sichel's S Measure
    #- Simpson's Diversity Index
    
    # 7. syllable frequencies #NOTE this is averaged/normalized syllable frequncies. NOTE the syllables docs suggest using cmudict for accuracy over speed.
    sfr = toks_alpha.apply(lambda x: sum([syllables.estimate(w) for w in x]) / len(x) if len(x) > 0 else 0.0) # could also use statistics.mean for all of these averages..
   
    
    # 8. Readability 
    # Flesch-Kincaid reading ease
    fk = data[text_col].apply(lambda x: textstat.flesch_reading_ease(x))
   
    
    # # 8.1 Automated Readability Index  
    # ari = data[text_col].swifter.apply(lambda x: textstat.automated_readability_index(x))
    # r_ari = listify(ari, feature_name='automated_readability_index')
    
    # # 8.2 Coleman-Liau index
    # cli = data[text_col].swifter.apply(lambda x: textstat.coleman_liau_index(x))
    # r_cli = listify(cli, feature_name='coleman_liau_index')
    
    # # 8.3 Dale Chall Readability Index
    # dci = data[text_col].swifter.apply(lambda x: textstat.dale_chall_readability_score(x))
    # r_dci = listify(dci, feature_name='dale_chall_index')
    
    # # 8.4 Gunning Fog Index
    # gfi = data[text_col].swifter.apply(lambda x: textstat.gunning_fog(x))
    # r_gfi = listify(gfi, feature_name='gunning_fog_index')
    
    # 8.5 Consensus based on all tests in textstat.
    # consensus = data[text_col].swifter.apply(lambda x: textstat.text_standard(x, float_output=True))
    # r_consensus = listify(consensus, feature_name='readability_consensus_score')
    
    # Could add basic sentiment with doc.token.sentiment? 
    
    # Store results TODO store each list of dicts in separate dict on the same level.
    # wl_results = {
    #             {'length_features' : w_total, w_len_avg, asl_w, asl_c},
    #              {'specific_w_frequencies' : upper_ws, shortws, elongws, nums, past_freq, pres_freq, adj_positives, adj_cmp_sup ,ems},
    #              {'nonstandard_spelling' : s_check},
    #              {'words_in_quotes' : quot_ws},
    #              {'richess/complexity' : ttr, hlgs, hldgs},
    #              {'syllable frequencies' : syl_freq},
    #              {'readability' : r_fk, r_ari, r_cli, r_dci, r_gfi, r_consensus} 
    
    #              }
    # print('\nSTORING RESULTS')
    
    
    
    # print('DONE')
    
    if analyze == True:
        w_total = [{'len_total_words': x[1]} for x in toks.map(len).items()]
        w_len_avg = [{'avg_word_length' : x[1]} for x in awl.items()] 
        asl_w_avg = [{'avg_sent_len_words': x[1]} for x in asl_w.items()]
        asl_c_avg = [{'avg_sent_len_chars' : x[1]} for x in asl_c.items()] # move this to character level.
        upper_ws = [{'upper_words': x[1]} for x in uws.items()]
        shortws = [{'short_words': x[1]} for x in sws.items()]
    
        elongws = [{'elongated_words' : x[1]} for x in elws.items()]
        nums = listify(nss, feature_name='numerical_tokens_frequency')
        past_freq = listify(pst, feature_name = 'past_tense_frequency')
        pres_freq = listify(prs, feature_name='present_tense_frequency')
        adj_positives = listify(adj_pos, feature_name='positive_adjectives')
        adj_cmp_sup = listify(adj_c_s, feature_name='comp_and_sup_adjectives')
        ems = [{'emote_frequencies': x[1]} for x in e.items()]
        s_check = [{'nonstandard_words': x[1]} for x in sc.items()]
        quot_ws = listify(quotes, feature_name = 'quotes_in_words')
        ttr =  [{'type-token_ratio': x[1]} for x in tt.items()]
        hlgs = listify(hlg, feature_name= 'hapax_legomena')
        hdlgs = listify(hdlg, feature_name='hapax_dislegomena')
        syl_freq =  [{'avg_syllable_freq': x[1]} for x in sfr.items()]
        r_flk = [{'flesch_kincaid_reading_ease' : x[1]} for x in fk.items()]
    
        # Store results in dict.
        wl_results['total_word_len'] = w_total
        wl_results['avg_word_len'] = w_len_avg
        wl_results['avg_sentence_len_words'] = asl_w_avg
        wl_results['avg_sentence_len_chars'] = asl_c_avg
        
        wl_results['uppercased_words'] = upper_ws
        wl_results['short_words'] = shortws
        wl_results['elongated_words'] = elongws
        wl_results['numberlike_tokens'] = nums
        
        wl_results['past_tense_words'] = past_freq
        wl_results['present_tense_words'] = pres_freq
        wl_results['positive_adjectives'] = adj_positives
        wl_results['comp_and_sup_adjectives'] = adj_cmp_sup
        
        wl_results['emotes'] = ems
        wl_results['nonstandard_spelling'] = s_check # exclude?
        wl_results['quoted_words'] = quot_ws
        
        wl_results['type_token_ratio'] = ttr
        wl_results['hapax_legomena'] = hlgs
        wl_results['hapax_dislegomena'] = hdlgs
        wl_results['syllable_freqs'] = syl_freq #takes too long?
        
        wl_results['readability_flesch_kincaid'] = r_flk
        # wl_results['readability_ari'] = r_ari
        # wl_results['readability_coleman_liau'] = r_cli
        # wl_results['readability_dale_chall'] = r_dci
        # wl_results['readability_gunning_fog'] = r_gfi
        #wl_results['readability_consensus'] = r_consensus
    
        return wl_results 
        
        
    else:
        # Build dataframe
        wl_results_df = pd.DataFrame()
            
        wl_results_df['total_word_len'] = w_total_series #✅
        wl_results_df['avg_word_len'] = awl #✅
        wl_results_df['avg_sentence_len_words'] = asl_w #✅
        wl_results_df['avg_sentence_len_chars'] = asl_c #✅
        
        wl_results_df['uppercased_words'] = uws #✅
        wl_results_df['short_words'] = sws #✅
        wl_results_df['elongated_words'] = elws #✅
        wl_results_df['numberlike_tokens'] = nss #✅ 
        
        wl_results_df['past_tense_words'] = pst #✅
        wl_results_df['present_tense_words'] = prs #✅
        wl_results_df['positive_adjectives'] = adj_pos #✅
        wl_results_df['comp_and_sup_adjectives'] = adj_c_s #✅
        
        wl_results_df['emotes'] = e #✅
        wl_results_df['nonstandard_spelling'] = sc #✅
        wl_results_df['quoted_words'] = quotes # ✅
        
        wl_results_df['type_token_ratio'] = tt #✅
        wl_results_df['hapax_legomena'] = hlg #✅
        wl_results_df['hapax_dislegomena'] = hdlg #✅
        wl_results_df['syllable_freqs'] = sfr #✅ 
        
        wl_results_df['readability_flesch_kincaid'] = fk #✅
        
        return wl_results_df
      
        
    
    #return get_wl(data)#get_cl(data) , get_wl(data)
    
#%%
# Debugging
# test_df = train.iloc[:50, :] 
# test = get_wl(test_df, analyze=False)
# %%


#%%

# Extract sentence-level features (syntactic)
def get_sl(data, text_col = 'text_clean',cv=None , train=False, analyze=False):



    # load spacy model.
    print('Loading spacy model') 
    nlp = spacy.load('en_core_web_sm')
    nlp.enable_pipe("senter") #TODO Added senter to get_sl while passing on docs for speed.
    # For POS tags, you could map a pos tag sequence/vector to the tweet.

    # Initialize CounVectorizer for pos ngrams.  store pos tags in separate column and transform with sklearn-pandas per column instead.
    if train == True:
        cv = CountVectorizer(analyzer='word', ngram_range=(1,3))
    else:
        cv = cv
        
    # Retoknize the text
    docs = data[text_col].swifter.apply(lambda x: nlp(x))
    #toks = docs.apply(lambda x: [t.text for t in x]) # not used.
    
    #return pd.DataFrame({'docs' : docs.map(len) , 'toks': toks.map(len)})
        
    # Frequencies
    # 1.1 frequencies of stop words (i.e. function words)
    sts = docs.apply(lambda x: len([t.text for t in x if t.is_stop == True]) / len(x)) # normalized by all tokens (including numbers and punct.)
   
    
    
    
    # 1.2 frequencies of punctuation
    pnct = docs.apply(lambda x: len([t.text for t in x if t.is_punct == True]) / len(x))
    
    
    # 1.3 Frequencies of roots (normalized by total number of words in tweet).
    rts = docs.apply(lambda x: len([(t, t.dep_) for t in [t for t in x if t.is_space == False] if t.dep_ == 'ROOT']) / len(x)) # This still includes number-like tokens, punctuation and mentions, since these are relevant in the dependency trees. Normalization could account for whitespaces, but doesn't have to.
    
    
    
    # 3. POS frequencies.
    # Extract pos tags:count (use Counter)
    pos = docs.apply(lambda x: [t.pos_ for t in x if t.text.isalnum() == True])
    pos_freq = docs.apply(lambda x: {p:c/len([t for t in x if t.text.isalnum() == True]) for p, c in Counter([t.pos_ for t in x if t.text.isalnum() == True ]).items()}) # normalized by alphanumeric tokens (since punctuation frequencies are captured separately). 
    
    #pos_freq = [{k:v} for k, v in pfreq.items()]
    #return pd.DataFrame({'text' : data[text_col] , 'tokens' : toks, 'pos' : pos}) 
    
    
  
    # 4. POS ngrams (n=uni-bi-tri) - TODO move to ngrams
    # join pos tags into strings for CountVectorizer -> return as special case.  Do a type check in the lookup or vectorize function that just passes the matrix on. OR pass on POS strings to vectorize in the vectorize function?
    
    #print('fit/transforming posgrams')
    pgrams = pos.str.join(' ').to_list() 
    if train == True:
        pgram_matrix = cv.fit_transform(pgrams)
        #return cv, pgram_matrix
    
    else:
        pgram_matrix = cv.transform(pgrams)
    
    # Sketch of countvectorizing pos ngrams.
    #cv.fit_transform(test.str.join(sep=' ').to_list()) # This works.  consider how to get pos ngrams and still make them interpretable in the corpora - e.g. most frequent triplets? Does that even really tell you anthing? You could Counter or use a pandas method to get most frequent combination?
    # {k:v for k, v in Counter(cv.get_feature_names()).items()} 
    
    # Note Counter has counter.most_common(n)
    
    # Could use nltk.util.ngrams(sequence, n) as suggested here https://stackoverflow.com/questions/11763613/python-list-of-ngrams-with-frequencies
    
    # 6. Sentiment?
   # sentis = docs.apply(lambda x: sum([t.sentiment for t in x])) # doesn't work. needs training?

    #return pd.DataFrame({'n_sents_spacy' : n_sents, 'n_sents_tstat' : n_sents_tstat})
    
    if analyze == True:
        # Store results.
        stop_freq = listify(sts, feature_name='stopword_frequency')
        punct_freq = listify(pnct, feature_name='punctuation_freq')
        root_freq = listify(rts, feature_name='root_frequencies')
        syn_results = {'stopword_freq': stop_freq,         
                       'syn_punc_freq' : punct_freq,
                       'root_freq': root_freq, 
                       'pos_freq' : list(pos_freq),
                       'pos_ngrams' : pgram_matrix}
        
        return cv,  syn_results
    
    else:
        syn_results_df = pd.DataFrame()
        
        syn_results_df['stopword_freq'] = sts
        
        syn_results_df['syn_punc_freq'] = pnct
        syn_results_df['root_freq'] = rts
        
        #syn_results_df['pos_freq'] = list(pos_freq) 
        
        #syn_results_df['pos_ngrams'] = pgram_matrix 
        return docs, cv, pgram_matrix, syn_results_df
# To call on test data, remember to call it on the cv returning after calling it on the training data - call it 'train_cv' in model.py

#%%
# Debugging
# test_df = train.iloc[:50,:]
# test = get_sl(test_df, train=True, analyze=True)
#%% ANCHOR testing get_syn

# extract_feats(test_df, analyze=True, train=True)

# NOTE when extracting in model.py, call twice instead of once.
#train.columns.get_loc('text_clean')
# test_df = train.iloc[:50, :]  # versus list version: train_text[:20]
# test = get_syn(test_df)
# # val_test = get_lexical(train_text[:5])

#%%


#%%

# Extract document-level features (structural)
def get_dl(data, text_col='text_clean', analyze=True, docs=[]):
    
    
    # 1. Number of sentences
    
    if len(docs) <= 0:
        print('Configuring spacy model for document level')
        nlp = spacy.load('en_core_web_sm', disable=['lemmatizer', 'parser','tagger','ner'])
        nlp.enable_pipe('senter') #  this is the main diff between wl, sl and dl.
        docs = data[text_col].swifter.apply(lambda x: nlp(x))
    ns = docs.apply(lambda x: len([s for s in x.sents])) #en_web_sm is not as accurate as blank or textstat.
    # ns = data[text_col].apply(
    #     lambda x: textstat.sentence_count(x))
    
    
    # 2. Number of user mentions - absolute counts.
    ms = data[text_col].str.count('@user', flags=re.I|re.M)
    
    
    # Could be expanded to include hashtags and urls in the future here.
    if analyze == True:
        n_sents = listify(ns, feature_name = 'number_of_sentences')
        ments = listify(ms, feature_name = 'number_of_mentions')
        struc_results = {'n_sents': n_sents, 'n_mentions': ments} # before skiping listify.
        #struc_results = {'n_sents' : ns, 'n_mentions' : ms}
        return struc_results
    else:
        struc_results_df = pd.DataFrame()
        struc_results_df['n_sents'] = ns #✅
        struc_results_df['n_mentions'] = ms #✅
        return struc_results_df
#%%
# Testing get_struc.
#test = get_dl(test_df, analyze=False)

#%%
# ANCHOR function to lookup and get specific [{features: x.x}] from extraction funct.
def feature_lookup(f_param_dict, extracted_features):
    feature_name1 = [{'feature_name' : 0.0}]
    for var in locals():
        if var in f_param_dict['some_feature_cat1']:
            return locals()[var]
            
            
# also look into dpath, dict-toolbox2
#%%
# Test feature_lookup
# t = {'some_feature_cat1': ['feature_name1', 'feature_name2']}
# feature_lookup(t)

#%%
def conc_features(matrixList):
    # Concatenate feature vectors
    
    # pass a list or dict of matrices and do list/dict comprehension/unpacking?
    
    #combined_features = hstack([feature_vector1, feature_vector2], 'csr')
    
    combined_features = hstack(matrixList, 'csr')
    
    return combined_features
#%%

def d_vectorize(selected_feats, train=False, dv=None):
    
#  Old approach: Vectorize all generated lists of dicts (stored in a dict or list?).
    
    # if train == True:
    #     dv = DictVectorizer()
    
    # #X = d.fit_transform(dictList)
    # # Either store as list.
    # dvList = []
    # matList = []
    
    # # Or in single dict
    # #matDict = dict() using dv as a key just overwrites the value since they are all identical. Nesting the dict just complicates things even more...
    
    # if train == True:
    # # Iterate through feature lists of dictionaries (lexical, syntactic, structural)
    #     for feature_name, feat_list in selected_feats.items():
    #         #print(feature_name, feat_list)
    #         #return
    #         if feature_name == 'pos_ngrams':  # Check for pos_ngrams (already vectorized)
    #             matList.append(feat_list) # if pos_ngrams feat matrix, just append it.
    #             #matDict[dv] = feat_list
    #             continue
    #         if train == True:
    #             feat_matrix = dv.fit_transform(feat_list)
            
           
            
    #      # NOTE storing each vectorizer 
    #         dvList.append(dv)
    #         matList.append(feat_matrix)
    
    
    
    # # This is the test case 
    # # The test case. transforming test data to fitted individual dvs.    
    # if train == False: #iterate through each dv and all the feature lists.
    #     feat_lists = []
    # # this has to only fit once per feature dv-featurelist pair.
    #     for feature_name, feat_list in selected_feats.items():
    #         if feature_name == 'pos_ngrams':
    #             matList.append(feat_list)
    #             continue
                
    #         feat_lists.append(feat_list)
        
    #     #return(feat_lists)
        
    #     for dv, featList in list(zip(dvs, feat_lists)): # enable this to loop through both dvs and features.    
    #                 #print(dv, featList)
    #                 feat_matrix = dv.transform(featList) # this needs to be passed its corresponding dv. if you store in zip/list, it should have the same, fixed order. but how to iterate?
                    
    #                 matList.append(feat_matrix)
    
        
    #     #matDict[dv] = feat_matrix
        
    # # Is LIWC a separate case? Should be the same as engineered features.
    
    
    # #return matDict#dv, matList #matDict.values() should be list of matrices equal to number of features. To be concatenated.
    # return dvList, matList

# New approach - using dfs with selected features.
    # 1. Get list of dicts, row-wise from selected features DF. 
    feats = selected_feats.to_dict('records')
    
    if train == True:
        dv = DictVectorizer()
        
        feats_vecs = dv.fit_transform(feats)
        
        return dv , feats_vecs
        
    else:
        feats_vecs = dv.transform(feats)
        
        return dv, feats_vecs
        
    
#%%
####
# test_df = train.iloc[:50,:]
# sent_cv_train, extracted_train = extract_feats(test_df, text_col='text_clean', analyze=False, train=True, feature_pms=feature_params)  

# sent_cv_test, extracted_test = extract_feats(val.iloc[:50,:], text_col='text_clean', analyze=False, train=False, cv=sent_cv_train, feature_pms=feature_params)

# train_dv, train_vecs = d_vectorize(train_selected_feats_df, train=True)
# test_dv, test_vecs = d_vectorize(test_selected_feats_df, train=False, dv=train_dv)
####

#test = d_vectorize(extracted_test, train=False, dvs=train_dvs)



# Then d_vectorize LIWC matches.
# Then concat all of the vectorized features.
# Then fit model!

#%%

def extract_feats(data, text_col='text_clean', feature_pms=dict(), analyze=False, cv=None, train=False):
    
    # Data = dataframe - can be recast by child functions.
    # See if resetting data index speeds up extraction.
    data.reset_index(drop=True, inplace=True)

    # lowercase all @USER mentions. An artifact from preprocessing.
    data[text_col] = data[text_col].str.replace(
        '@USER', '@user')  # , inplace=True)
    
    all_features_dict = dict()
    
    all_features_df_list = []
    
    selected_features = dict()
    
    # 1. Call each of the extractor functions
    # 1.3 Sentence-level # TODO moved up to pass docs to other extraction functs for speed.
    print('Sentence level features')
    if analyze == True:
        docs = []
        sent_cv, sent_lvl = get_sl(
            data, text_col=text_col, cv=cv, analyze=analyze, train=train)
    else:
        docs, sent_cv, pgram_matrix, sent_lvl = get_sl(data, text_col=text_col, cv=cv, analyze=analyze, train=train)
    
    # 1.1 Character-level (10 features)
    print('Character level features')
    char_lvl = get_cl(data, text_col=text_col, analyze=analyze) 
    
    
    # 1.2 Word-level
    print('Word level features')
    word_lvl = get_wl(data, text_col=text_col, analyze=analyze, docs=docs) 
    
    
    
        #sent_lvl = word_lvl.copy(deep=True)
    #return sent_lvl
    # if train == False:
    #    sent_cv, sent_lvl = get_sl(data, text_col=text_col, analyze=analyze)
    
    
    
    # 1.4 Document-level
    print('Document level features')
    doc_lvl = get_dl(data, text_col=text_col, analyze=analyze, docs=docs)
    
    #return doc_lvl
    
    # Return all features if extracting for feature analysis. LIWC is analyzed separately.
    if analyze == True:
        
        # Store in dict
        all_features_dict['character_level'] = char_lvl
        all_features_dict['word_level'] = word_lvl
        all_features_dict['sentence_level'] = sent_lvl # Maybe pop pgrams matrix into separate var/container?
        all_features_dict['document_level'] = doc_lvl
        
        
        return sent_cv, all_features_dict # pass sent_cv on to analyze_feats from here.
        
   
    
# Old approaches
    # Option 1 - extracting flat list (of n instances) (of dicts with n features) to vectorize in one go.
        # for feat_cat, feature_name in feature_pms['engineered'].items():
        #     if feat_cat in all_features.keys():
        #         selected_features[feat_cat] = all_features[feat_cat].values() 
        # return selected_features
        
        # TODO how to make sure that all features align? Pandas? hstack before fitting?
    
    # Option 2 - extract individual lists of [{'feature1' : feature_value}... {'feature2' : feature_value}] for each feauture?
        # Iterate through features to pass on, given parameters in parameter dict.
        # Get a flat list of all desired target features.
        #target_feats = list(itertools.chain.from_iterable([fn for fn in feature_pms['engineered'].values()]))
        
    # Lookup and retrieve each feature from all_features and store in selected_features
        
        # Works, but return that awkward df with individual dicts.
        # for feat_level, feat_name in all_features.items():# outer level {'feature_level': 'feature_name': [{'feature' : feature_val}]}
        #     for fn, fl in feat_name.items():
        #         if fn in target_feats:
        #             selected_features[fn] = fl
        
        # Return selected features
        
    
# 2. return selectively for classification
    if analyze == False:
        
        
        # Get a flat list of all desired target features.
        target_feats = list(itertools.chain.from_iterable([fn for fn in feature_pms['engineered'].values()]))
        
        #return char_lvl, word_lvl, sent_lvl, doc_lvl
        # Concatenate feature dfs for each level horizontally.
        #all_feats_df = pd.concat([char_lvl, word_lvl, sent_lvl, doc_lvl], axis=1, join='inner') # works.
        
        all_feats_df_list = [char_lvl, word_lvl, sent_lvl, doc_lvl]
        
        # Mitigating duplicate indeces in dfs..
        [df.reset_index(inplace=True, drop=True) for df in all_feats_df_list] 
        # 1.5 LIWC features
        # parsed_liwc is called in the main namespace.
        if feature_pms['liwc'] == True:
            liwc_feats = pd.DataFrame.from_records(
                liwc_match(parsed_liwc, data, extract=True))
            #selected_features['liwc_counts'] = liwc_feats # store LIWC straight in selected_feats dict.
            # index liwc_feats with data.index
            liwc_feats.set_index(data.index, inplace=True)
            
            all_feats_df_list.append(liwc_feats)
            
            #return liwc_feats
            #return sent_cv, all_features
            # concat liwc features to df selected features.
            
        # Concat all feature dfs.
        #try: 
        all_feats_df = pd.concat(all_feats_df_list, axis=1, join='inner')
        
        #print(all_feats_df)
        
        #except:
        #    return all_feats_df_list# , [len(df) for df in all_feats_df_list]
        # Select columns from all features df unless they are pos_ngrams. could add pos_freqs here. 
        # return all_feats_df 35+64=99 feats.
        
        selected_feats_df = all_feats_df[[fn for fn in target_feats if fn != 'pos_ngrams']]
    
        #return all_feats_df, target_feats
        
    
        
        return sent_cv, pgram_matrix, selected_feats_df    
            
    
#%% ANCHOR procedure for feature extraction.
# test_df = train.iloc[:50,:]
# #sent_cv, train_feats_df = extract_feats(test_df, feature_pms = feature_params, analyze=False, train=True)

# # Parse LIWC
# parsed_liwc = parse_liwc('../../../Data/LIWC2007dictionary poster.xls', text_col=text_col)
    
# # This is just a test of extraction with liwc.
# liwc_test = extract_feats(test_df, feature_pms = feature_params, analyze=False, train=True)



# # Dict_vectorize-fit_transform train.
# train_en_feat_vec = d_vectorize(train_selected_feats_df, train=True)

# # Combine feature matrices: # also use ngrams in model.py.
# train_feats_combined = conc_feat([train_pgram_matrix , train_en_feat_vec])

# # Extract test_feats
# sent_cv, test_pgram_matrix, test_selected_feats_df = extract_feats(val.iloc[:50,], feature_pms= feature_params, analyze=False, train=False, cv=sent_cv)

# # Dict_vectorize-transform test with train_dv.
# test_en_feat_vec = d_vectorize(test_selected_feats_df, train=False)

# -> concat pgram matrices and each selected feature df after dictvectorizing them. 


####
#analysis = analyze_feats(train_feats_dict) # analysis case
#feats_for_vec = extract_feats(test_df, feature_pms=feature_params, analyze=False, train=True) # the train case
# test = extract_feats(test_df, analyze=True, cv=train_cv, train=False) # test case

#%%
# analyze features TODO move to data_exploration

def analyze_feats(featuresDict, resultpath='./exploring/feature_analysis/', cv=None):
    
    # This function is called on the complete output of all the extract functions.
    # Put all extracted features into a single dict. You then call vectorize and concat on that based on lookup either manual or via function.

    # LIWC is handled separately..
    
    # 0. Append all lists of dicts (dictLists) to one flat list.
    featList = []
    
    
    posfreqList = [] 
    pgrams = None
    # Smarter solution : calculate stats directly on dict values in lists of dicts.
    
    statsDict = dict()
    
    #Loop through top level featDict
    for feat_level, feat_name in featuresDict.items():
        #featList.append(pd.DataFrame(feat_name))
        #print(feat_name.keys())
        
        #Second level - individual feature names : ['feature' : int/flaot].
        for feat, feat_value in feat_name.items():
            #print( feat, type(feat_value))
            
            # store pos features seperately.
            if feat == 'pos_freq':
                posfreqList.append(pd.DataFrame(feat_value))
                continue
            
            if feat == 'pos_ngrams':
                pgrams = feat_value
                continue
            
            featList.append(pd.DataFrame(feat_value))
    
    # Concat lists of extracted feature dataframes.
    featDF = pd.concat(featList, axis=1)
    
    #featDF = pd.DataFrame.from_records(featList)
    posfreqDF = pd.concat(posfreqList) # 
    #return posfreqDF.mean().to_dict()
    #return featDF
    
    #return featDF, posfreqDF
    
    # Split features into binary and frequency-based
    # Get series of bools columnwise where any value is not a float and greater than 1.
    filter_cols = featDF.select_dtypes(exclude=float).gt(1).any(0)
    
    # Filter the featuresDF based on a list of the above indeces (i.e. column names) in from the series above.
    #binDF = featDF.select_dtypes(exclude = float).between(0, 1)
    binDF = featDF.loc[ : , filter_cols[filter_cols == False].index.tolist()]
    absoDF = featDF.loc[ : , filter_cols[filter_cols == True].index.tolist()] # absolute counts
    freqDF = featDF.select_dtypes(float)
    
    #return binDF #, absoDF, freqDF
    # Multindex
    # reformDict = {}
    
    # for outerKey, innerDict in featuresDict.items():
    #     for innerKey, values in innerDict.items():
    #             reformDict[(outerKey,
    #                         innerKey)] = [value for value in]
    # return reformDict
    
    
    
    #return binDF, absoDF , freqDF, posfreqDF 
    
    # Write all the dfs to spreadsheet for easier visualization.
    writer = pd.ExcelWriter(resultpath, engine='xlsxwriter')
    binDF.to_excel(writer, sheet_name="binary_features")
    absoDF.to_excel(writer, sheet_name='absolute_features')
    freqDF.to_excel(writer, sheet_name="frequency_features")
    posfreqDF.to_excel(writer, sheet_name="pos_frequencies")
    #writer.save()
    
    # Get basic stats for each category of feature values.
    # Store results (binary, absolute and freq)
    
    # binary features - percentages of positives.
    bin_pcts =  (binDF.sum().divide(len(binDF)) * 100) 
    # Store in dict
    statsDict['binary_features (% positive)'] = bin_pcts.to_dict()
    # Write to sheet.
    bin_pcts.to_excel(writer, sheet_name = 'binary_percentages')
    
    # absolute count features - sum totals.
    abso_totals = absoDF.sum() # TODO change this to averages (e.g. average number of sentences etc..)
    statsDict['absolute_features (sum total)'] = abso_totals.to_dict() 
    abso_totals.to_excel(writer, sheet_name = 'absolute_totals')
    
    # frequency features - means.
    freq_means = freqDF.mean() #.round(3)
    statsDict['frequency_features (average)'] = freq_means.to_dict()  # mean of the normalized frequencies (rounded to 3 decimal points) - EXCLUDING POS frequencies..
    freq_means.to_excel(writer, sheet_name = 'frequencies_mean')
    
    # POS frequencies
    posfreq_means = posfreqDF.mean() #.round(3)
    statsDict['pos_frequncies'] = posfreq_means.to_dict()
    posfreq_means.to_excel(writer, sheet_name = 'posfreq_means')
    
    
 
    # Analyzing ngrams
    
    ### Adatped from this excample https://gist.github.com/xiaoyu7016/73a2836298cfaef8212fd20a94736d56 ###
    
    # # Here you just store the pgrams in a df.
    pgram_freqs = pd.DataFrame(pgrams.sum(axis=0).T,   
    index=cv.get_feature_names(), columns=['freq']).sort_values(by='freq', ascending=False)#.max(20).plot(kind='bar',title='posgrams')
    
    ### Acessed 19-05-2021 ###
    
    # Write pgrams to sheet in spreadsheet for inspection.
    pgram_freqs.to_excel(writer, sheet_name='pos_ngram_frequencies')
    
    # Save excel file.
    writer.save()
    
    # Return the results dictionary.
    return statsDict

#%%


 

#%% ANCHOR combining / concatenating features.

#draft of how to use DictVectorizer on LIWC



# Example of vectorizing dictionary counts.
# dv = DictVectorizer(sparse=False)  or sparse=True which yields sparse.csr.

# D = [{'cat1' : 3, 'cat2': 0 ...} {'cat1': 0, 'cat2': 55}...]
# X_train = dv.fit_transform(D) <- where D is the result of running the extract=True function on the training data.
    
    # This is where you would normalize either the dense numpy.ndarray or sparse.csr:
    # transformer = Normalizer().fit(X)
    # X_train = transformer.transform(X_train) 

# X_test = dv.fit(X_test) etc..

# Concatenate X (feature) arrays?


### https://stackoverflow.com/a/22710579 ###
# - "Don't forget to normalize this with sklearn.preprocessing.Normalizer, and be aware that even after normalization, those text_length features are bound to dominate the other features in terms of scale"
### Accessed 28-04-2021 ###


# Watch out for nan's in the resulting feature array? 
    # https://stackoverflow.com/q/39437687 reports on this as a side-effect of using DictVectorizer.
#%%

# NOTE You probably want to wrap all of these in one extract_features function...


#%%






#%%

if __name__ == "__main__":



#%%
    # ANCHOR Read some data just to develop with to test with.
    train = pd.read_pickle('../../../Data/train/train.pkl')
    
    
    val = pd.read_pickle('../../../Data/validation/validation.pkl')
    #%%
    
    train_text = train['text_clean'].tolist()
    y_train = pd.read_pickle('../../../Data/train/train_labels.pkl')
    
    val_text = val['text_clean'].tolist()
    y_val = pd.read_pickle('../../../Data/train/train_labels.pkl')
    
    
    qanon_classify = pd.read_pickle('../../../Data/qanon/preprocessed/classify/prep_qanon.pkl')
    
    nonqanon_classify = pd.read_pickle('../../../Data/nonqanon/preprocessed/classify/prep_nonqanon.pkl')
    qanon_same_size = qanon_classify.sample(n=len(nonqanon_classify))
    


#%%
    # Testing get_ngrams
    feature_params = {'ngrams': {'type': 'char',
                                 'size': (3, 3),
                                 'lowercase': False,
                                 'max_vocab': None},  # can be set to None. half or even 1/4 10k seem like good options.
                      'engineered':{ 
                          'character_level': ['char_punc_freq',
                                             'digit_freq', 
                                             'whitespace_freq',
                                             'linebreak_freq', 
                                             'uppercased_char_freq', 
                                             'special_char_freq',
                                             'repeated_questionmark', 
                                             'repeated_periods', 
                                             'repeated_exclamation',
                                             'contains_equals'],
                          
                          'word_level' : ['total_word_len',
                                          'avg_word_len',
                                          'avg_sentence_len_words',
                                          'avg_sentence_len_chars',
                                          
                                          'uppercased_words',
                                          'short_words',
                                          'elongated_words',
                                          'numberlike_tokens',
                                          
                                          'past_tense_words',
                                          'present_tense_words',
                                          'positive_adjectives',
                                          'comp_and_sup_adjectives',
                                          
                                          'emotes',
                                          'nonstandard_spelling',
                                          'quoted_words',
                                          
                                          'type_token_ratio',
                                          'hapax_legomena',
                                          'hapax_dislegomena',
                                          'syllable_freqs',
                                          
                                          'readability_flesch_kincaid'],
                          
                          'sentence_level': ['stopword_freq',
                                             'syn_punc_freq',
                                             'root_freq', 
                                             #'pos_freq', # Disable me!
                                             'pos_ngrams'],
                          
                          'document_level' : ['n_sents', 
                                              'n_mentions']
                                              
                        },
                        'liwc' : False # True
                      }
                      
                      
                      
    # Store feature_params in json.
    # with open('./feature_params.json', "w") as outfile:
        # json.dump(feature_params, outfile)
    
    # Load feature_params from json.
    # json_feat_params = json.load(open('./feature_params.json'))
    
#%% ANCHOR debugging individual extraction functions
    #test_df = train.iloc[:50,:]
    # test = get_wl(test_df, text_col='text_clean')
    # print(test)
    # exit() #ANCHOR EXITING PROGRAM
#%% ANCHOR test extraction. Pickle for convenience.
    #text_col = 'text_clean'
    parsed_liwc = parse_liwc('../../../Data/LIWC2007dictionary poster.xls')
    
    q_feats = extract_feats(qanon_same_size, text_col='text_clean', analyze=True, train=True)

    nq_feats = extract_feats(nonqanon_classify, text_col='text_clean', analyze=True, train=True) 
    
#     # Save
    pickle.dump(q_feats, open('./exploring/feature_analysis/q_feats.pickle', 'wb'))
    
    pickle.dump(nq_feats, open(
    './exploring/feature_analysis/nq_feats.pickle', 'wb'))

    
#     # Load
    q_feats = pickle.load(open('./exploring/feature_analysis/q_feats.pickle', 'rb'))
    
    nq_feats = pickle.load(open('./exploring/feature_analysis/nq_feats.pickle', 'rb'))
#%% ANCHOR testing feature analysis
#     test_df = train.iloc[ :50 , :]
#     test_cv, test_featuresDict = extract_feats(test_df, analyze=True, train=True)
#     analysis = analyze_feats(test_featuresDict , resultpath='./exploring/feature_analysis/train_sample.xlsx', cv=test_cv)
    sent_cv, train_pgram_matrix, train_selected_feats_df = extract_feats(train, feature_pms = feature_params, analyze=False, train=True)
    
    train_vecs = d_vectorize(train_selected_feats_df, train=True)
#%% ANCHOR quick feature inspection in corpora.

    # quick searches to get impression of features
    # qanon_same_size = qanon_sample(n=len(nonqanon_classify)) # sample qanon to same size as nonqanon.
    # Search chars.

    # total count of char in corpus as string (also in pct).
    #qanon_classify.text_clean.to_string().count(r'\n').sum() #/len(qanon_classify.text_clean.to_string())

    # number of tweets containing char (also in pct)
    #qanon_classify.text_clean.str.contains(r'\n').sum() #/len(qanon_classify.text_clean.str)

#%% ANCHOR Testing scipy calculation of chi-squared test on absolute counts.

    # df_matches = q_matches['results']['matches_df']
    
    
    # cats = df_matches.columns
    
    # x_sq_test = chisquare(df_matches)
    
    # df_x_sq_test = pd.DataFrame(x_sq_test).transpose()
    # df_x_sq_test.columns = ['chisquared_statistic', 'p-value']
    
    # df_x_sq_test.set_index(cats, inplace=True) 
    
    # df_x_sq_test
    # Try pd.crosstab()

#%%[markdown]
### Inspecting LIWC counts

#%%
#%%

