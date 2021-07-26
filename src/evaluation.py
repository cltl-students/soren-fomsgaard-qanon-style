
#%% #ANCHOR IMPORTS
import pickle
import pandas as pd
from collections import defaultdict

from topic_model import read_data, compute_corpus
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore

import pyLDAvis.gensim

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


#%%

# Function to load pickled model.


def load_clf(trained_mod):
    """Load a trained model from pickle file.
    
    Args:
        trained_mod (str): file path to pickle file.

    Returns:
        sklearn.classifier: A trained sklearn classifier.
    """

    # save model with open(wb) + pickle.dump.
    with open(trained_mod, 'rb') as file:
        model = pickle.load(file)

    return model


# %%[markdown]

def load_lda(model_path, dictionary):
    """Load a saved lda model and a dictionary.

    Args:
        model (str): path to a gensim gensim.LDAModel or LdaMulticore model.
        dictionary (str): path to a gensim.Dictionary dict used for training a model.

    Returns:
        tuple: the loaded (model, dictionary).
    """

    m = LdaMulticore.load(model_path)
    d = corpora.Dictionary.load(dictionary)
    return m, d

#%%
## Visualization #ANCHOR
def vis_topics(model, corpus, dictionary, name='ldamodel'):

    #model, dictionary = load_lda(model)
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary, R=20)

    # save visualization to html object.
    pyLDAvis.save_html(vis, f'./exploring/topic/visualization/{name}_visualization.html')

    #vis

    return vis




### Accessed 22-02-2021 ###

#%%
# Evaluate an LDA model
def eval_lda(lda_model, docs, corpus, dictionary):
    ### Adapted from tutorial at:  https://stackabuse.com/python-for-nlp-working-with-the-gensim-library-part-2/ ###
    # Compute perplexity - using model and corpus.
    perplexity = lda_model.log_perplexity(corpus)
    
    # NOTE takes too long to compute...
    # Compute coherence score via coherence model - using model, docs and dictionary.
    # coherence_model = CoherenceModel(model=lda_model,
    #                                  texts=docs,
    #                                  dictionary=dictionary,
    #                                  coherence='c_v')

    # coherence_score = coherence_model.get_coherence()
    
    return perplexity #, coherence_score
### accessed 05-03-2021 ###

#%%
#%%
# Compare LDA models NOTE Only works for models of equal shape, fitted to same corpus.
def compare_mods(model_1, model_2):
    pass

    mdiff, label = model_1.diff(model_2)
    topic_diff = mdiff  # get matrix with difference for each topic pair from `m1` and `m2`
    
    return topic_diff


#%%
# Load model, corpus, docs and dictionary for evaluation

 

def eval_topic_mod():
    
    print(f'\n\n Loading saved LDA models, corpora, docs and dicts ...')
    
    # Qanon
    qanon_model, qanon_dictionary = load_lda('./models/topic/lda/qanon/20_topics_multi_bigram.model' , './models/topic/lda/dictionary/qanon_bigram.dict')
    qanon_docs = read_data('../../../Data/qanon/preprocessed/topic/prep_qanon.pkl')
    #%%
    qanon_corpus, qanon_id2word = compute_corpus(qanon_docs)
    
    #%%
    
    # Nonqanon
    nonq_model, nonq_dictionary = load_lda('./models/topic/lda/nonqanon/20_topics_multi_bigram.model' , './models/topic/lda/dictionary/nonqanon_bigram.dict')
    nonq_docs = read_data('../../../Data/nonqanon/preprocessed/topic/prep_nonqanon.pkl')
    nonq_corpus, nonq_id2word = compute_corpus(nonq_docs)
    #%%
    
    vis_q = vis_topics(qanon_model, qanon_corpus, qanon_dictionary, name="qanon")
    # inspect visualization
    vis_q
    #%%
    vis_non_q = vis_topics(nonq_model, nonq_corpus, nonq_dictionary , name='nonqanon')
    vis_non_q
    
    
    #%% Call separate evaluation function
    
    qanon_eval = eval_lda(qanon_model, qanon_docs, qanon_corpus, qanon_dictionary)
    print(f'\n Perplexity score for Qanon LDA model: {qanon_eval}')
    #%%
    nonq_eval = eval_lda(nonq_model, nonq_docs, nonq_corpus, nonq_dictionary)
    print(f'\n Perplexity score for Nonqanon LDA model: {nonq_eval}')





#%% Compare qanon and nonqanon models.

#compare_mods(qanon_model , nonq_model)


#%%[markdown] #ANCHOR Classifier

### Evaluate classifiers.

# %%

def report_classifier(y_true, y_pred): #, labels):
    
    # add metrics.accuracy_score
    
    report_text = classification_report(y_true, y_pred)#, output_dict = True) #, labels) # use output_dict=True to return dict.
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    #print(report_text)
    
    # Draw confusion matrix , return some other scores?
    
    
    return report_dict , report_text

# %%

def get_scores(y_true, y_pred):
    pass # NOTE not necessary
    # precision
    #precision = precision_scor
    # recall
    
    # f1-score
    
    # Accuracy
    
    
#%%

# Manual calculation of accuracies

def eval_counts(goldlabel, machinelabel):
    '''
    This function compares the gold label to machine output
    
    :param goldlabel: the gold label
    :param machinelabel: the output label of the system in question
    :type goldlabel: the type of the object created in extract_label
    :type machinelabel: the type of the object created in extract_label
    
    :returns: a countainer providing the counts of true positives, false positives and false negatives for each class
    '''

    # Initiate default dicts for collecting counts.
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    class_counts = defaultdict(dict)

    # Zip label from both input objects.
    comparison = list(zip(goldlabel, machinelabel))

    # I got the idea of using a checklist like so from Nathan Vandermolden-Pater.
    # A list of labels to find false positives/negatives with.
    check_list = list(set(goldlabel))

    # Iterate through label pairs to compare.
    for label in check_list:
        for gold_label, machine_label in comparison:
            if gold_label == label and machine_label == label:
                true_positives[label] += 1
            if machine_label == label and gold_label != label:
                false_positives[label] += 1
            if gold_label == label and machine_label != label:
                false_negatives[label] += 1

        class_counts[label] = {"true positives": true_positives[label],
                               "false positives": false_positives[label],
                               "false negatives": false_negatives[label]}

    # An example of how to structure the dict in a DataFrame.
    #df_counts = pd.DataFrame.from_dict(class_counts, orient='index')
    #print(df_counts)

    return class_counts


#%%

def cal_acc(evaluation_counts):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary
    
    :param evaluation_counts: the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns the precision, recall and f-score of each class in a container
    '''

    metrics = defaultdict(int)

    for label, counts in evaluation_counts.items():
        precision = round(
            counts["true positives"] / (counts["true positives"] + counts["false positives"]), 2)
        recall = round(counts["true positives"] /
                       (counts["true positives"] + counts["false negatives"]), 2)
        fscore = round((2 * (precision * recall)) / (precision + recall), 2)

        metrics[label] = {"precision": precision,
                          "recall": recall,
                          "f-score": fscore}

    return metrics

#%%

def plot_c_mat(predictions, labels = ['NONQ', 'Q'], norm=None):
    
    # m = load_clf(clf)
    p = pd.read_csv(predictions)
    
    X_test = p['text']
    y_pred = p['predicted_label']
    y_true = p['true_label']
    
    c_mat = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm)
    
    disp = metrics.ConfusionMatrixDisplay(c_mat, display_labels = labels)
    
    disp.plot()
    
    #return

#%%

def get_errors(predictions, e_path='./models/evaluation/errors/', name = 'errors'):

    p = pd.read_csv(predictions)
    
    errors = p.query('predicted_label != true_label')
    
    errors.to_csv(f'{e_path}{name}.tsv', sep='\t', index=False)
    
    return errors
    
    

#%%
if __name__ == "__main__":
    #%% ANCHOR CALL to visualize topic models.

    #eval_topic_mod()
    pass
