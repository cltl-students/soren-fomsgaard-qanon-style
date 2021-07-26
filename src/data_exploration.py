

#%% 

# imports


# ANCHOR IMPORTS
import pandas as pd
import swifter
from itertools import chain
from ast import literal_eval
import numpy as np


import re
from collections import Counter

from keywords import get_kws

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#from wordcloud import WordCloud as wc

#import nltk

# import spacy
# from sklearn.model_selection import train_test_split
# import collections
# from collections import Counter, defaultdict
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import en_core_web_sm
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import jaccard_score


from feature_extraction import *


#%%[markdown]

### Reading data
#%%


#%%

#ANCHOR  TWEETS STATS

def get_tweet_stats(data):  
    """Get basic statistics on tweets contained in a dataframe.

    Args:
        data (DataFrame): A Pandas DataFrame with tweets.
    
    Returns:
        dict: A dictionary with the counts etc. for tweets in the DF.
    """

    # tweet_stats = {'total tweets' : int,
    #                'total languages' : int,
    #                'most frequent languages' : dict,
    #                'most favs' : dict, # these have form index (in DF) : n of favs
    #                'top hashtag' : str
    #                } # consider concatening/aggregating some of these values with df_combined =  df.longest_tweet.map(str) + / + df.shortest_tweet.map(str) 
    
    # Dict for stats.
    tweet_stats = {}
    
    
    total_tweets = len(data)
    # Containers for counting tweet lengths.
    total_tokens = int
    total_chars = int
    unique_tokens = set()
    avg_tokens = float
    avg_char = float
    
    # Counts for the source data.
    
    # Average tweet length
    
    # Simple token count based on whitespace splits.
    whitespace_tokens = list(chain(*data['text'].str.split(" ")))
    total_tokens = len(whitespace_tokens)
    avg_tokens = total_tokens / total_tweets
    
    # Unique unigrams.
    unique_tokens.update(whitespace_tokens)
    
    # Lengths in characters.
    total_chars = data.text.str.len().sum()
    
    avg_char = round(data.text.str.len().mean(), 3)
    # equivalent to round(sum(map(len, data.text) ) / len(data.text))
    
    
    # Check whether data is normalized or not and count tweet lengths accordingly.
    if "tokens_norm" in data.columns:
        # Add logic for normalized data.
        
        # Count length in tokens
        all_tokens = list(chain(*data['tokens_norm']))
        avg_tokens = len(all_tokens) / len(data['tokens_norm'])
        
        # Count unique unigrams.
        #data['tokens_norm'].apply(unique_tokens.update)
        unique_tokens = data['tokens_norm'].explode().unique()
        
        # Join the tokens column to get length in chars.
        all_chars = pd.Series(data.tokens_norm.str.join(sep=" "))
        total_chars = all_chars.str.len().sum()
        #avg_char = len(all_chars) / len(data['tokens_norm'])
        
        avg_char = round(all_chars.str.len().mean() , 3)
        
    if "tokens_clean" in data.columns:
        # Counting lengths for non-aggressively normalized data.
            
        # Joining processed tokens and counting len.
        all_tokens = list(chain(*data['tokens_clean']))
        total_tokens = len(all_tokens)
        avg_tokens = total_tokens /  total_tweets
        
        # Count unique unigrams.
        # data['tokens_clean'].apply(unique_tokens.update)
        unique_tokens = data['tokens_clean'].explode().unique()
        
        total_chars = data.text_clean.str.len().sum()
        
        avg_char = round(data.text_clean.str.len().mean(), 3)
    
    

    
    # 
    # Longest tweet
    #longest_tweet = data['text'].str.len().nlargest(n=5) #.max() 
    
    # more optimal way:
    #print(data.text.map(lambda x: len(x)).max())
    #%%
    
    # Shortest tweet 
    #shortest_tweet = data['text'].str.len().nsmallest(n=5) 
    
    
    # Tweets per language
    
    #total_languages = data['language'].unique().dropna(inplace=True) # this counts nan (which are apparently there..)
    total_languages = data['language'].value_counts().index.tolist() 
    
    #%%
    # replace NaNs with "null"
    data['language'].fillna('unknown' , inplace=True) #NOTE NaNs tend to be just a url/mentions etc.
    
    # Pandas oneliner that works.
    #data.groupby('language').size()
    
    langs_by_freq = data['language'].value_counts()
    most_freq_langs = langs_by_freq[:10]
    # most_freq_langs = data.groupby(by='language').count()
    
    # check for discrepancy between number of raws with languages and those without (NaN)
    # len(data) - langs_by_freq.sum() # get difference between tweets with and without language detected.
    # check for nulls 
    # data['language'].isnull().values.any()
    # sum them & and store in series for concatenation.
    #num_no_lang = pd.Series(data['language'].isnull().sum())# NOTE luckily, they are equal.
    
    
    #%%
    # Most favs
    
    tweet_max_like = data['favs'].nlargest(n=5)
    
    # Least favs # obviously 0
    #tweet_min_like = data['favs'].nsmallest(n=5)
    
    
    
    #%%

    
    # Store Text stats.
    tweet_stats['total tweets'] = total_tweets
    
    tweet_stats['total size/length (tokens/chars)'] = {'tokens' : total_tokens , 'character' : total_chars}
    
    tweet_stats["avg. tweet length (tokens/chars)"] = {'tokens' : avg_tokens , 'characters' : avg_char} #avg_tweet_len char / tokens
    
    tweet_stats["unique tokens (unigram)"] = len(unique_tokens)
    tweet_stats['total languages'] = len(total_languages)
    tweet_stats['most frequent languages'] = most_freq_langs.to_dict()
    tweet_stats['most favs'] = tweet_max_like.to_dict()
    # Add average fav?
    
    tweet_stat_list = [result for result in tweet_stats.items()]
    
    # dict => df.
    #df_tweet_stats = pd.DataFrame.from_dict(tweet_stat_list)
    
    # df => LaTeX
    # df_tweet_stats.to_latex()
    
    
    # indeces_to_find = tweet_max_like.index.tolist() # all of these methods work!
    # data.query('index in @tweet_max_like.index.tolist()')
    # data.take(indeces_to_find) works as well.
    
    return tweet_stats
    

##################################

#%%
def get_user_stats(data):
    """Get basic statistics about users in a dataframe with tweets.

    Args:
        data (DataFrame): A Pandas DataFrame with tweets.

    Returns:
        dict: A dictionary with the resulting counts.
    """

    # ANCHOR USER STATS
    # basic_user_stats = {'Number of users': int,  
    #                     # 'Most frequent user' : int
    #                     'Users per language' : {str : int}, # use defaultdict? str=languge , int number of tweets
    #                     'Average tweet per user' : int
    #                     }
    
    # Dict for storing counts.
    user_stats = {}
    
    # Number of total users
    unique_users = data.user.unique()
    
    # Most frequent user name - top 10 - NOTE NOT RELEVANT
    most_freq_user = data['user'].value_counts()[:10].index.tolist()
     
    
    # Languages per user
    lang_per_user = data.language.groupby(data.user).unique()
    
    
    avg_lang_per_user = lang_per_user.map(len).mean() # Average n of languages (counting  'unknown')
    
    
    # Users per language
    user_per_lang = data.user.groupby(data.language).unique().map(len)
    # Sort counts.
    user_per_lang.sort_values(ascending = False, inplace=True)
    
    #avg_user_per_lang = user_per_lang.map(len).mean()
    avg_user_per_lang = user_per_lang.mean()
    
    # Seems like a lot of languages for some users, but it adds up:
    #len(lang_per_usr.get(key='Adam Cunningham'))
    
# User(s) with longest/shortest tweet #
    longest_users = data.loc[data['text'].str.len().groupby(data['user']).idxmax()]
    
    
# Average tweet per user 
    #avg_tweet_len = round(data.text.str.len().groupby(data.user).mean()) # length of tweet per user
    
    tweet_per_usr = data.text.groupby(data.user).count().sort_values(ascending=False)  # seems strange that some users have so many tweets.
    
    # But is adds up if we search for one usr:
    # data.loc[data['user'] =='Jennifer Kemp']
    
    avg_tweet_per_user = round(tweet_per_usr.mean(), 3) # mean tweets per n of tweets per user?
    
    
    #%%
# Most liked users
    
    # Most favs
    
    usr_max_like = data.groupby('user').agg({'favs':'max'}).nlargest(n=10, columns=['favs'])
    
    # Least favs # This is obviously 0.
    usr_min_like = data.groupby('user').agg({'favs': 'min'}).nsmallest(n=10, columns=['favs'])
    
    
# Store the user stats.
    user_stats['Number of users'] = len(unique_users)
    #user_stats['Most frequent user'] = most_freq_user # not relevant.
    user_stats['avg. tweet per user'] = round(avg_tweet_per_user, 3) 
    
    
    user_stats['n users per language'] = user_per_lang
    user_stats['avg. users per language'] = round(avg_user_per_lang, 3)
    user_stats['avg. language per user'] = round(avg_lang_per_user, 3) 
    
    
    return user_stats


#%% [markdown]
# Plot / visualize some of the above.




#%% [markdown]

# you can search for:# 
    # - soros
    # - new world order
    # - agenda21
    # - mark of the beast
    # - ashkenazi
    # - stopthesteal
    # - draconian
    
    # To get a sense of the language
    
    # %%
    
    # Let's try that.
    
    # test_terms = ["soros",
    #               "new world order",
    #               "agenda21",
    #               "mark of the beast",
    #               "ashkenazi",
    #               "stopthesteal",
    #               "draconian"]
    
    # test_terms = re.compile("soros",
    #                         "new world order",
    #                         "agenda21",
    #                         "mark of the beast",
    #                         "ashkenazi",
    #                         "stopthesteal|draconian")
    
    # could also use re.findall - series.str.findall()?



# %%[markdown]

##Visualizations

#%%

# Pie chart for language proportions (source data)

def lang_prop(data):
    
    stats = get_tweet_stats(data)    
    
    # This only counts the top 10 most frequent languages.
    #langs = sorted(stats['most frequent languages'].items())
    df_langs = data['language'].value_counts().reset_index()
    #df_langs = pd.DataFrame(langs , columns=['language' , 'frequency'])
    df_langs.columns = ['language' , 'frequency']
    # Add percentages.
    df_langs['%'] = df_langs['frequency'].apply(lambda x:\
                                        round(x/df_langs['frequency'].sum()*100 , 2))
    
    # Sort df.
    df_langs.sort_values(by='%', ascending=False, inplace=True)
    
    # Oneliner for percentages.
    #df['column'].value_counts(normalize=True) * 100
    
    # subset into other langs (less than 1%).
    threshold = 1
    
    # extract 'other' languages.
    #other_langs = df_langs[df_langs['%'] < threshold]
    other_langs = df_langs[df_langs['%'] < threshold]#['language']
    # rename 'other'languages - lower than threshold:
    df_langs.loc[(df_langs['%'] < threshold, 'language')] = 'other'
    
    ### Aggregation adapted from: https://stackoverflow.com/a/46827856 ###
    d = {'language': 'first', 'frequency': 'sum', '%': 'sum'}
    # trying to aggregate directly
    df_maj_oth = df_langs.groupby('language', as_index=False).aggregate(d).reindex(columns=df_langs.columns)
    
    ### Accessed 25-03-2021 ###
    
   
    #majority = df_langs[df_langs['%'] > threshold]
   
    #other_langs["language"] = "other"
    
    #https: // pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html
    # You could try using df.groupby('language').agg(['frequency' , '%']) # agg also takes functions like min and max.
    
    # subset into top five
    #majority = df_langs[:5].copy()
    #other_langs = pd.DataFrame( data= {"language" : ["other"],
    #                                "frequency" : [df_langs["frequency"][5:].sum()],
    #                                "%" : [df_langs["%"][5:].sum()]})
    
    
    
    
    
    
    
    # Concat other category to majority of languages.
    #df_maj_oth = pd.concat([majority, other_langs])  
    
    # Sort df.
    df_maj_oth.sort_values(by='%', ascending=False, inplace=True)
    
    # Plotting into pie(s)
    
    labels = df_maj_oth['language']
    values = df_maj_oth['frequency']
    
    
    #fig, ax = plt.subplots()
    
    # Without subplots()
    #pie = plt.pie(values,autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.1)
    
    ### lambda adapted from: https://stackoverflow.com/a/54087182 ###
    # Wedge percentages are rounded same as in df above, inside the format method in the lambda func.
    pie = plt.pie(values, autopct=lambda p: '{:.2f}% ({:.0f})'.format(round(p , 2), (p/100)*values.sum()), textprops={"color" : "w"})#, pctdistance=1.1, labeldistance=1.1)
    ### Accessed 26-03-2021 ###
    #ax.pie(values, autopct = '%1.1f%%' , pctdistance=1.1, labeldistance = 1.1)
    
    
    
    # Add legend
    plt.legend(pie[0], labels,title="language", loc="best", bbox_to_anchor=(1, 1))
    
    # Final laoyout adjustment.
    plt.tight_layout()
    plt.axis("equal")
    
    # Store chart in var
    #fig = plt.figure(figsize=(8, 6))
    plt.title("Distribution of Languages")
    # Show the pie chart.
    plt.show()
    
    
    
    # proportions of language (use a pie chart instead)
    #data['language'].value_counts()[:10].plot(kind='pie')
    
    # Note could return len(other_langs) for inspection of remaining languages in ther 'other' category.
    
    
    # https: // matplotlib.org/stable/gallery/pie_and_polar_charts/bar_of_pie.html breaks down a wedge. Could be used to break down the 'other' category if its members weren't so small.

    return pie, df_maj_oth, other_langs   
    


#%%[markdown]

### Emojis, emoticons, hashtags and keywords


def kw_graphs(data_norm, data_non):
    """Print pie chart visualization of hashtags, keywords and emotes in a dataset.

    Args:
        normalized_data (str): a filepath to a preprocessed dataset.
    """    
     
    # Read two data files at a time to get relevant counts.
     
    #data_norm = pd.read_pickle(normalized_data)
    
    #data_non_norm = pd.read_pickle(non_normalized_data)
    
    # 1. get keywords from normalized data (Qanon and Nonqanon)
    
    # Use scatter / cluster plot for these.
    keywords, hashtags = get_kws(data_norm, normalized=True)
    
    # Use pie chart for these.
    
    
    # 2. get emotes from non-aggressively norm. data 
    emojis, emoticons = get_kws(data_non, normalized=False)
    
    # Percentages of each emote.
    emoj_pct =  emojis.apply(lambda x: x/emojis.sum()*100)
    emot_pct = emoticons.apply(lambda x: x/emojis.sum()*100)
    
    threshold = 2
    most = emojis[emojis < threshold]
    other = emojis[emojis > threshold]
    
    
    # get percentage of each htag, keyword, emoji etc.?
    
    # Pie chart
    #emojis.plot(kind="pie")
    #plt.show()
    
    return  keywords, hashtags #emojis, emoticons

#%%
# #ANCHOR Function calls to kw_graphs()
#emoj, emot = kw_graphs(qanon_topic, qanon_classify)



#%%[markdown]
# Tweets over time.
#%%
# ANCHOR TIME

def tweets_over_time(data):
    """Get tweet distributions over time (wekk) and plot them.

    Args:
        data (pandas.DataFrame): A df with tweets.

    Returns:
        [type]: [description]
    """
    
    # TODO could have supplied parse=_date=["date"] to read_csv...
    
    ## NOTE: in the unconverted df they look like: "Fri Dec 11 00:41:21 +0000 2020"
    # 1. Convert timestamps to proper date time objects 
    
    data.date = data.date.swifter.apply(pd.to_datetime)
    
    ### Adapted from https://cvw.cac.cornell.edu/PyDataSci1/tweets_retweets ###
    df_time = data.groupby(pd.Grouper(
        key="date", freq='D', convention='start')).size()
    
    # ^ change freq to 'W-MON' to group by weeks. It might make more sense to tweets per week rather than day?
    
    ### Accessed 26-03-2021 ###
    
    
    #gmt_offset = +1 # offsetting Amsterdam time?
    
    #data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    
    #data['date'] = data.date + pd.to_timedelta(gmt_offset, unit='h')
    
    
    # Tweet/text per day
    #tweet_per_day = data[['text']].groupby(data['date'].dt.date).count()  
    
    # Tweet_per_week = pd.PeriodIndex(data['date'], freq='W')
    
    #tweet_per_week = data.groupby(data['date']).dt.week.unique().count() 
    
    #tweets_over_time = plt.bar(range(len(tweet_per_week)) , tweet_per_day.max())
    
    
    # Plotting
    # Matplotlib
    
    # Via pandas
    
    df_time.plot.line(figsize=(18 , 6), ylim=(0 , 10000))
    
    #plt.plot(data=df_weeks)
    
    plt.title('Tweets over time (per day)')
    plt.xlabel('Date')
    plt.ylabel('N Tweets (per day')
    plt.grid(True)
    #plt.show()
    
    #plt.close('all') # close the figure after editing.
    
    #%%
    # Try seaborn
    
    #sns.displot(df_time, kind='kde')
    # equicalent to:
    # data.date.plot()
    
    return df_time
    


# %%

# Inspect the post collection
# test = pd.read_csv('../../../Data/nonqanon/non-qanon-feb-mar - post collection.csv')
# test.columns = ['id', 'user', 'lang', 'text', 'date', 'favs']
# pat = 'qanon' #could also be regex 
# qs = test.query("text.str.contains(@pat)", engine='python') # could also just be df.filter()


#%% ANCHOR Main call

if __name__ == "__main__":


#%% 
    print(' ==== READING DATA ====\n\n')
#ANCHOR READ DATA
    # Load different version of the data (raw, topic preprocessed and? )
    
    # Source data 
    
    # Qanon
    qanon_src = pd.read_csv('../../../Data/qanon/qanon-dec-jan.csv', names=["id",
                                                                          "user",
                                                                          "language",
                                                                          "text",
                                                                          "date" ,
                                                                          "favs"])
    # Nonqanon
    nonqanon_src = pd.read_csv('../../../Data/nonqanon/non-qanon-feb-mar.csv' , names=["id",
                                                                            "user",
                                                                            "language",
                                                                            "text",
                                                                            "date",
                                                                            "favs"])
    #%%
    # Processed data
    
    # Topic
    qanon_topic = pd.read_pickle('../../../Data/qanon/preprocessed/topic/prep_qanon.pkl')
    
    nonqanon_topic = pd.read_pickle('../../../Data/nonqanon/preprocessed/topic/prep_nonqanon.pkl')
    
    
    # Classification
    
    qanon_classify = pd.read_pickle('../../../Data/qanon/preprocessed/classify/prep_qanon.pkl')
    
    nonqanon_classify = pd.read_pickle('../../../Data/nonqanon/preprocessed/classify/prep_nonqanon.pkl')
    qanon_same_size = qanon_classify.sample(n=len(nonqanon_classify))
    
    # Train / test / validation
    
    train = pd.read_pickle('../../../Data/train/train.pkl')
    test = pd.read_pickle('../../../Data/test/test.pkl') 
    validation = pd.read_pickle('../../../Data/validation/validation.pkl')
    
    # QaNonQ
    qanonq = pd.concat([train, test, validation])
    
    
    # Testing LIWC counts on the train proportions of each corpus.
    # q_train = pd.read_csv(
    #     '../../../Data/train/train_and_labels.csv').query('label == "Q"')
    #train_and_labels = pd.concat([pd.read_pickle('../../../Data/train/train.pkl'), pd.read_pickle('../../../Data/train/train_labels.pkl')], axis=1)
    
    
    # q_train = train_and_labels.query('cons == "Q"')
    # q_train.to_pickle('../../../Data/LIWC/for debugging/q_train.pickle')
    
                         
    # nq_train = train_and_labels.query('cons == "NONQ"')    
    # nq_train.to_pickle('../../../Data/LIWC/for debugging/nq_train.pickle')
#%% 
    print(' ==== RUNNING BASIC STAT FUNCTIONS ====\n\n')
#ANCHOR stat functions.

    #%% ANCHOR Tweet stat (text)
    print(' TEXT STATS...\n\n')
    # Qanon data
    print('QANON - TOPIC')
    get_tweet_stats(qanon_topic)
    print()
    
    print('QANON - CLASSIFY')
    get_tweet_stats(qanon_classify)
    q_clf_stat = get_tweet_stats(qanon_classify)  # assign to var for later use.
    print(q_clf_stat)
    print()
    
    print('QANON - SRC')
    get_tweet_stats(qanon_src)
    print()
    
    #%%
    # Nonqanon data
    print('NONQANON - TOPIC')
    get_tweet_stats(nonqanon_topic)
    print()
    
    print('NONQANON - CLASSIFY')
    get_tweet_stats(nonqanon_classify)
    nq_clf_stat = get_tweet_stats(nonqanon_classify) # assign to var for later use.
    print(nq_clf_stat)
    print()
    
    print('NONQANON - SRC')
    get_user_stat(nonqanon_src)
    print()
    #%%
    
    # Train data
    print('TRAIN DATA')
    get_tweet_stats(train)
    print()
    
    
    #q_train_stat = get_tweet_stats(q_train)
    #nq_train_stat = get_tweet_stats(nq_train)
    
    #%%
    # Validation data
    print('VALIDATION DATA')
    get_tweet_stats(validation)
    print()
    
    # Test data
    print('TEST DATA')
    get_tweet_stats(test)
    print()
    
    
    
    print('USER STATS...\n\n')
    #%% ANCHOR Tweetstat (users): call get_user_stats()
    
    # Qanon data
    print('QANON - TOPIC')
    get_user_stats(qanon_topic)
    print()
    
    print('QANON - CLASSIFY')
    get_user_stats(qanon_classify)
    print()
    
    print('QANON SRC')
    get_user_stats(qanon_src)
    
    #%%
    # Nonqanon data
    print('NONQANON - TOPIC')
    get_user_stats(nonqanon_topic)
    print()
    
    print('NONQANON - CLASSIFY')
    get_user_stats(nonqanon_classify)
    print()
    
    print('NONQANON - SRC')
    get_user_stats(nonqanon_src)
    print()
    
    #%%
    # Train data
    print('TRAIN DATA')
    get_user_stats(train)
    print()
    
    # Validation data
    print('VALIDATION DATA')
    get_user_stats(validation)
    print()
    
    # Test data
    print('TEST DATA')
    get_user_stats(test)
    print()
    
    ### Combined NonQanon
    print('COMBINED QANONQ DATA')
    get_tweet_stats(qanonq)
    get_user_stats(qanonq)
    print()

#%%
    print(' ==== PLOTTING LANGUAGE PROPORTIONS IN DATA ====\n\n')
#ANCHOR Getting and plotting language proportions with  lang_prop()

    qanon_src_langs = lang_prop(qanon_src)
    nonqanon_src_langs = lang_prop(nonqanon_src)

    langprop(nonqanon_src)


#%%
    print(' ==== VISUALIZATION OF TWEETS OVER TIME ====\n\n')
    print(' Note: cannot plot for Qanon and NonQanon simultaneously in separate graphs. Plotting Qanon..')
#ANCHOR Visualize tweets over time.
    # NOTE calling simultanously results in a single graph

    q_tweets_over_time = tweets_over_time(qanon_src)

#%%
    #nq_tweets_over_time = tweets_over_time(nonqanon_src)

#%%
    print(' ==== INSPECTING LIWC COUNTS IN QANON AND NONQANON SRC ====\n\n')
    print()
    
    print('PARSING LIWC')    
    terms = parse_liwc('../../../Data/LIWC2007dictionary poster.xls')
    terms_df = pd.DataFrame(terms)
    
    # ANCHOR Checking duplicates in the lexicon.
    #terms_df.replace('\*', '', regex=True)
    #terms_df.count().sum() - terms-df.nunique().sum() # doesnt work
    all_terms = terms_df.stack(dropna=True).to_list() # get a single list of all terms.
    
    term_counts = Counter(all_terms) # count them..
    all_terms_df = pd.DataFrame.from_dict(dict(Counter(all_terms)), orient='index').reset_index() # into DF
    
    all_terms_df.rename(columns = {'index' : 'term', 0 : 'count'}, inplace=True) # rename cols.
    dupes = all_terms_df[all_terms_df['count'] > 1]
    
    
    print('GETTING LIWC MATCHES IN DATA SETS...\n\n')

#%%    
#ANCHOR Inspection of total LIWC counts

    # Qanon LIWC matches for inspection
    # q_matches_df = q_matches['results']['matches_df']
    # nq_matches_df = nq_matches['results']['matches_df']
    
    print('QANON MATCHES')
    q_matches = liwc_match(
    terms, '../../../Data/qanon/preprocessed/classify/prep_qanon.pkl')
    print()
    
    # Pickle matches to save time later.
    with open('../../../Data/LIWC/for debugging/q_matches.pickle', 'wb') as outfile:
        pickle.dump(q_matches, outfile)
    
#%%
    # NonQanon matches.
    print('NONQANON MATCHES')
    nq_matches = liwc_match(
        terms, '../../../Data/nonqanon/preprocessed/classify/prep_nonqanon.pkl')
    print()
    
    
    # Pickle matches to save time
    with open('../../../Data/LIWC/for debugging/nq_matches.pickle', 'wb') as outfile:
        pickle.dump(nq_matches, outfile)


#%%
    # Load pickled LIWC results. 
    # Qanon
    with open('../../../Data/LIWC/for debugging/q_matches.pickle', 'rb') as infile:
        q_matches = pickle.load(infile)
    # NonQanon
    #%%
    with open('../../../Data/LIWC/for debugging/nq_matches.pickle', 'rb') as infile:
        nq_matches = pickle.load(infile)
    
        
        
#%% ANCHOR Analyzing LIWC match totals.
    
    q_totals = q_matches['results']['matches_total'].to_frame(name='totals')
    #q_totals
    
    #%%
    nq_totals = nq_matches['results']['matches_total'].to_frame(name='totals')
    #nq_totals


#%%
#%%
    # Add percentages to totals. 
    
    # Normalize categories by number of terms in categories?
    q_totals['cat_pct'] = round(q_totals['totals'] / q_totals['totals'].sum() * 100, 2)

    # Get frequencies relative to total corpus size.
    # Qanon
    q_totals['rel_freq'] = q_totals['totals'].apply(lambda x: x/q_clf_stat['total size/length (tokens/chars)']['tokens'])
     
    #Qanon train subset only
    #q_totals['rel_freq'] = q_totals['totals'].apply(lambda x: x/q_train_stat['total size/length (tokens/chars)']['tokens'])

     
    q_totals['rel_freq (%)'] = q_totals['rel_freq'].apply(lambda x: round(x * 100, 2))
    
#%%
    #NonQanon
    nq_totals['cat_pct'] = round(nq_totals['totals'] /
                             nq_totals['totals'].sum() * 100, 2)
                             
    nq_totals['rel_freq'] = nq_totals['totals'].apply(lambda x: x/nq_clf_stat['total size/length (tokens/chars)']['tokens'])
    
    # NonQanon train subset only.
    #nq_totals['rel_freq'] = nq_totals['totals'].apply(lambda x: x/nq_train_stat['total size/length (tokens/chars)']['tokens'])
    
    
    
    nq_totals['rel_freq (%)'] = nq_totals['rel_freq'].apply(lambda x: round(x * 100, 2))

#%%
    
    # Debugging LIWC matches again..
    #q_matches['regex_pats ']



#%% 
    #Concat totals into single overview
    compare_totals = pd.concat([q_totals, nq_totals],
                               axis=1, keys=('Qanon', 'NonQanon'))
                               
    # Add diff column Q vs. NonQ
    compare_totals['diff (Q vs. NQ)'] = compare_totals['Qanon']['rel_freq (%)'] - \
                                        compare_totals['NonQanon']['rel_freq (%)']
#%% 
    # Save totals comparison to file.
    compare_totals.to_csv('../../../Data/LIWC/totals_comparison.csv')
    #compare_totals.to_csv('../../../Data/LIWC/totals_comparison_train_only.csv')

    # Reload the comparison of LIWC matches.
    # total_comparison_full = pd.read_csv(
    #     '../../../Data/LIWC/totals_comparison.csv', index_col=[0], header=[0, 1])

    # total_comparison_full['Qanon']['rel_freq (%)'].sum()

#%% ANCHOR Testing scipy calculation of chi-squared test on absolute counts.

    # df_matches = q_matches['results']['matches_df']
    
    
    # cats = df_matches.columns
    
    # x_sq_test = chisquare(df_matches)
    
    # df_x_sq_test = pd.DataFrame(x_sq_test).transpose()
    # df_x_sq_test.columns = ['chisquared_statistic', 'p-value']
    
    
    # df_x_sq_test.set_index(cats, inplace=True)
    
    # df_x_sq_test
    # Try pd.crosstab()

#%% ANCHOR counting extracted features in the corpora.

#%% ANCHOR analyzing extracted stylistic features in the corpora

    # Extract features for analysis.
    print('EXTRACTING QANON FEATURES FOR ANALYSIS')
    q_pos_cv, q_feats = extract_feats(qanon_same_size, text_col='text_clean', analyze=True, train=True)
    print('EXTRACTING NONQANON FEATURES FOR ANALYSIS')
    nq_pos_cv, nq_feats = extract_feats(nonqanon_classify, text_col='text_clean', analyze=True, train=True) 
#%%
    print('ANALYZING QANON FEATURES')
    q_feat_analysis = analyze_feats(q_feats , resultpath='./exploring/feature_analysis/q_feature_analysis.xlsx', cv=q_pos_cv)
    print('ANALYZING NONQANON FEATURES')
    nq_feat_analysis = analyze_feats(nq_feats, resultpath='./exploring/feature_analysis/nq_feature_analysis.xlsx', cv=nq_pos_cv)
    print('DONE!')

# %%
