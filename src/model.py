
#%%
#ANCHOR imports

import pandas as pd
import numpy as np
import pickle
import json

import mlflow

from sklearn import svm, naive_bayes

from sklearn.linear_model import LogisticRegression 

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.impute import SimpleImputer 

from sklearn.model_selection import GridSearchCV, SelectFromModel

from feature_extraction import *



# Evaluate elsewhere?
from evaluation import report_classifier, eval_counts, cal_acc, load_clf
#from sklearn.metrics import classification_report
import shap

from collections import defaultdict 

from matplotlib import pyplot as plt
import addcopyfighandler # for clipboard export of plots

#%%


# #%%[markdown]
# ## Load datasets 
# #%%
# # ANCHOR load data 


# train = pd.read_pickle('../../../Data/train/train.pkl')


# val = pd.read_pickle('../../../Data/validation/validation.pkl')
# #%%

# train_text = train['text_clean'].tolist()
# train_labels = pd.read_pickle('../../../Data/train/train_labels.pkl').tolist()

# val_text = val['text_clean'].tolist()
# val_labels = pd.read_pickle('../../../Data/validation/val_labels.pkl').tolist()

#%%


#%%


def load_data(train, train_labels, test, test_labels, text_col='text_clean', baseline=True):

    X_train = pd.read_pickle(train)
    #X_train = X_train[text_col]
    y_train = pd.read_pickle(train_labels).to_list()
    
    X_test = pd.read_pickle(test)
    #X_test = X_test[text_col]
    y_test = pd.read_pickle(test_labels).to_list()
    
    if baseline==True:
        #return ((X_train, y_train) , (X_test, y_test))
        return X_train[text_col], y_train, X_test[text_col], y_test
    else:
        return X_train, X_test

### Logging function taken from mlflow documentation: https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html ###
def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

### Accessed 02-04-2021 ###

#%%
# Train baseline svm with countvectorizer on character trigrams.

# LinearSVC versus SVC 

#svm.SVC(kernel='linear', n_jobs=-1, loss="hinge") # set loss function to hinge as per https://stackoverflow.com/a/35081862

# fit to input vectors

def bayesline(data=tuple,feat_params=dict(), scaling=False, experiment_name='Baseline'):
    
    #### Using mlflow to log, following this tutorial: https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html ###
    # Fit model.

    # Enable sklearn autologging.
    mlflow.sklearn.autolog(log_model_signatures=False) # disable signature logging since data is csr matrices.
        
    ### Adapted from https://towardsdatascience.com/managing-your-machine-learning-experiments-with-mlflow-1cd6ee21996e ###
    # set experiment name to organize runs
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # set path to log data, e.g., mlruns local folder
    mlflow.set_tracking_uri('./mlruns')
    
    ### Accessed 02-04-2021 ###
    
# Initialize model.
   
    model = naive_bayes.MultinomialNB()
            
    #                  }
    # # Unpack data
    train_text, y_train, test_text, y_test = data
    
    # Test performance on a quarter of the test data.
    # train_quarter = int(len(train_text)*0.25)
    # train_text = train_text.iloc[:train_quarter]
    # y_train = y_train[:train_quarter]
    # test_text = test_text.iloc[:train_quarter]
    # y_test = y_test[:train_quarter]

    
# Run experiment
    print('Running experiment')
    with mlflow.start_run(experiment_id = experiment.experiment_id) as run:    
        # Extract ngrams.
        print('Extracting features')
        # cv is the CountVectorizer.
        cv, X_train, X_test = ngrams(train_text , test_text, feat_params) 
        
        print(f'Train ({type(X_train)}) feature matrix has shape: {X_train.shape}')
        print(f'Test ({type(X_test)}) feature matrix has shape: {X_test.shape}')
        
        
        # y_train, y_test = labels
        
        
       # Scale input when using svm.SVC
        # print('scaling input matrices..')
        if scaling == True:
            scaler = preprocessing.StandardScaler(with_mean=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        
        # Train model.
        print('Training model')
        model.fit(X_train, y_train)
       
     
        
        # Test model.
        print('Making model predictions on test data')
        #y_pred = model.predict(X_test)
        
        
        # Get model predictions.
        y_pred = model.predict(X_test)
        
            
# Quick and dirty evaluation
        print('Evaluating prediction performance')
        evaluation_dict = report_classifier(y_pred, y_test)[0]
        evaluation_text = report_classifier(y_pred, y_test)[1]
        
        print(evaluation_text)
        
# Log feature parameters
        mlflow.log_params({'n train instances': X_train.shape[0],\
                           'n train features': X_train.shape[1],\
                           'n test instances': X_test.shape[0],\
                           'n test features' : X_test.shape[1],\
                           'scaling': scaling})
        mlflow.log_params(feat_params['ngrams'])
        
        # log evaluation of prediction to files
        mlflow.log_dict(evaluation_dict, 'pred_eval_scores.yaml')
        mlflow.log_text(evaluation_text, 'pred_eval_pretty.txt')
        
        mlflow.log_metrics(evaluation_dict['macro avg'])
        
        # Log confusion matrix for error analysis.
        
        
# Save predictions to csv.
        predictions = pd.DataFrame({'text' : test_text,\
                                    'predicted_label': y_pred,\
                                    'true_label': y_test})
                                                                    
        
        
        predictions.to_csv(f'./models/baseline/nb/results/baseline_predictions.csv', index=False)
        
        # also log predictions as an artifact file in mlflow.
        mlflow.log_artifact('./models/baseline/nb/results/baseline_predictions.csv')
        
        # fetch logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        
    # Save model
        model_path = f"models/baseline/nb"
        try:
            mlflow.sklearn.save_model(model, model_path)
        except: #MlflowException
            print(f'The path, {model_path} already exists. Skipping save.')
            pass
        
    # Save vectorizer
    pickle.dump(cv, open(f'{model_path}/vectorizer.pkl', 'wb'))
        
    ### Acessed 01-04-2021 ###
    
   
    
    #classification_report(y_test, y_pred)
    print('DONE!')
    return model, (X_train, X_test), (y_test, y_pred), cv



#%%
def baseline(m_type='svm', data=tuple,feat_params=dict(), scaling=False, experiment_name='Baseline', gs=False, m = None):
    
    #### Using mlflow to log, following this tutorial: https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html ###
    # Fit model.

    # Enable sklearn autologging.
    mlflow.sklearn.autolog(log_model_signatures=False) # disable signature logging since data is csr matrices.
        
    ### Adapted from https://towardsdatascience.com/managing-your-machine-learning-experiments-with-mlflow-1cd6ee21996e ###
    # set experiment name to organize runs
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # set path to log data, e.g., mlruns local folder
    mlflow.set_tracking_uri('./mlruns')
    
    ### Accessed 02-04-2021 ###
    
# Initialize model.
  
    # Too slow:
    # 'True' SVM
    #model = svm.SVC(kernel='linear', probablity=True) #, cache_size=7000)# , C=0.1, max_iter=10000)
    
    # linearSVC (highest model is most recent)
    # NOTE fails to converge with dual=True, despite advice.
    # NOTE Final version:
    if m_type == 'svm': # old setting (28-05-2021) scaling=false, sq_hinge, dual=True
        model = svm.LinearSVC(max_iter=50000, penalty='l2', dual=False, loss='squared_hinge', C=0.1, verbose=1) #, loss='hinge', C=0.1) # NOTE started with max_iter=10k. Used to converge with uncompressed feature matrices.
   
   
    
    # NOTE logreg seems to have higher performance out of the box at it does converge.
    # Old fashioned Logreg
    #model = LogisticRegression(max_iter=10000, n_jobs=-1, dual=True)
    # This probably also overfits.
    
    # Trying RandomForest.
    if m_type == 'randomforest':
        model = RandomForestClassifier(n_jobs = -1, verbose=1) # n_estimators defaults to 100.
                               
    
            
    # Grid search
    if gs == True:
        params = {'max_iter' : [5000], 'C' : [0.1]}
        model = GridSearchCV(model, params, n_jobs =-1, return_train_score=True, verbose=1)
    
    if m != None:
        model = m
    
    # # Unpack data
    train_text, y_train, test_text, y_test = data
    # Test performance on a quarter of the test data.
    # train_quarter = int(len(train_text)*0.25)
    # train_text = train_text.iloc[:train_quarter]
    # y_train = y_train[:train_quarter]
    # test_text = test_text.iloc[:train_quarter]
    # y_test = y_test[:train_quarter]
    
        
# Run experiment
    print('Running experiment')
    with mlflow.start_run(experiment_id = experiment.experiment_id) as run:    
        # Extract ngrams.
        print('Extracting features')
        # cv is the CountVectorizer.
        cv, X_train, X_test = ngrams(train_text , test_text, feat_params) 
        
        print(f'Train ({type(X_train)}) feature matrix has shape: {X_train.shape}')
        print(f'Test ({type(X_test)}) feature matrix has shape: {X_test.shape}')
        
        
        # y_train, y_test = labels
        
        
       # Scale input when using svm.SVC
        # print('scaling input matrices..')
        if scaling == True:
            scaler = preprocessing.StandardScaler(with_mean=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        
        # Train model.
        print('Training model')
        model.fit(X_train, y_train)
       
     
        
        # Test model.
        print('Making model predictions on test data')
        #y_pred = model.predict(X_test)
        
    # Try RandomForest, except LinearSVC.
        y_pred_conf = [np.nan for i in X_test] 
        y_pred_prob = [] #np.nan
        feature_imps = []
        
        try:
            feature_imps = model.feature_importances_ #.argsort() # get feature importances and sort high to low.    
            #y_pred_prob = model.predict_proba(X_test)
        
        except:
            y_pred_conf = model.decision_function(X_test)
            #y_pred_prob_uncal = model._predict_proba_lr(X_test) # this works but is less reliable.
            
            # feautre imps are model.coef_().ravel()?
        
        # Get model predictions.
        y_pred = model.predict(X_test)
        
        # Calibrate model.
        calibrated = CalibratedClassifierCV(model, cv='prefit')
        calibrated = calibrated.fit(X_train, y_train)
        
        # Get probabilities.
        y_pred_prob = calibrated.predict_proba(X_test)
            
# Quick and dirty evaluation
        print('Evaluating prediction performance')
        evaluation_dict = report_classifier(y_pred, y_test)[0]
        evaluation_text = report_classifier(y_pred, y_test)[1]
        
        print(evaluation_text)
        
# Log feature parameters
        mlflow.log_params({'n train instances': X_train.shape[0],\
                           'n train features': X_train.shape[1],\
                           'n test instances': X_test.shape[0],\
                           'n test features' : X_test.shape[1],\
                           'scaling': scaling})
        mlflow.log_params(feat_params['ngrams'])
        
        # log evaluation of prediction to files
        mlflow.log_dict(evaluation_dict, 'pred_eval_scores.yaml')
        mlflow.log_text(evaluation_text, 'pred_eval_pretty.txt')
        
        mlflow.log_metrics(evaluation_dict['macro avg'])
        
        # Log confusion matrix for error analysis.
        
        
# Save predictions to csv.
        predictions = pd.DataFrame({'text' : test_text,\
                                    'predicted_label': y_pred,\
                                    'pred_probability': list(y_pred_prob),\
                                    'pred_confidence': list(y_pred_conf),\
                                    'true_label': y_test})
                                    #,\
                                    #'pred_prob_uncal': list###(y_pred_prob_uncal)
        
      
        predictions.to_csv(f'./models/baseline/{m_type}/results/baseline_predictions.csv', index=False)
        
        # also log predictions as an artifact file in mlflow.
        mlflow.log_artifact('./models/baseline/svm/results/baseline_predictions.csv')
        
        # fetch logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        
    # Save model
        model_path = f"models/baseline/{m_type}/"
        try:
            mlflow.sklearn.save_model(model, model_path)
        except: #MlflowException
            print(f'The path, {model_path} already exists. Skipping save.')
            pass
        
    # Save vectorizer
    pickle.dump(cv, open(f'{model_path}/vectorizer.pkl', 'wb'))
        
    ### Acessed 01-04-2021 ###
    
   
    
    #classification_report(y_test, y_pred)
    print('DONE!')
    return (model, calibrated), (X_train, X_test), (y_test, y_pred, y_pred_prob, feature_imps), cv


#report_classifier(results[1][0], results[1][1])
# %%

# Test baseline on unseen, unrelated and nonconspiratorial data.

def pred_unrelated(logged_model, feature_pms=None):

    # 1. load baseline
    
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    # 2. extract ngrams (from real train data and 'fake' test data.)
    if feature_pms == None:
        feature_params = {'ngrams': {'type': 'char',
                                     'size': (3, 3),
                                     'lowercase': False,
                                     'max_vocab': None}  # can be set to None. half or even 1/4 10k seem like good options.
                                          \
                          }
        
    cv, X_train, X_alt_test = ngrams(train_text, alt_test_text, feature_pms)


    # 3. predict 
    # try:
    #loaded_model._predict_proba_lr(X_alt_test, alt_test_labels)
    # except:
    predictions = loaded_model.predict(X_alt_test)
    
    # 4. evaluate
    print('Evaluating prediction performance')
    evaluation_dict = report_classifier(predictions, alt_test_labels)[0]
    evaluation_text = report_classifier(predictions, alt_test_labels)[1]
    
    print(evaluation_text)
    
    return evaluation_dict, predictions



#%%


#%%
# ANCHOR extract most predictive features.

### adapted from https://aneesha.medium.com/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d ### 

 
def predictive_feats(classifier, feature_names, top_features=20):
     try:
        imps = classifier.coef_.ravel()
     except:
        imps = classifier.feature_importances_
     
     top_positive_imps = np.argsort(imps)[-top_features:]
     top_negative_imps = np.argsort(imps)[:top_features]
     top_imps = np.hstack([top_negative_imps, top_positive_imps])
     
     # create plot
     plt.figure(figsize=(15, 5))
     colors = ['red' if c < 0 else 'blue' for c in imps[top_imps]]
     plt.bar(np.arange(2 * top_features), imps[top_imps], color=colors)
     feature_names = np.array(feature_names)
     plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_imps], rotation=60, ha='right')
     plt.show()
     
     return plt

### Accessed 21-04-2021 ###
#%%
# ANCHOR
# Own take on getting predictive feats.

def imp_feats(model, vectorizer, top_features=20):

    
    
    #m = load_clf(model)
    # Assume that you get a numpy array with values for each feature.
    
    # Load vectorizer or rely on the global var here?
    
    #v = pickle.load(vectorizer, 'rb')
    
    # Get feature names
    feature_names = vectorizer.get_feature_names()
    
    # Or model
    try:
        imps = model.coef_.ravel() # svm coefficients.
    except:
        imps = model.feature_importances_ # randomforest
    
    ### Adapted from https://mljar.com/blog/feature-importance-in-random-forest/ ##
    df_feat_imps = pd.DataFrame(imps, index=feature_names, columns=[
                                'importance']).sort_values('importance', ascending=True)
    ### Accessed 22-04-2021 ###
    
    # Plot the 20 most important features - highest versus smallest values.
    # Are these for predictions on both classes? What do the negative values mean?
    
    smallest = df_feat_imps.nsmallest(top_features, columns='importance')
    
    # to get top coefficients, do the pandas equivalent of the np.hstack in the above functin.
    
    return df_feat_imps
    
#%%
def shap_vis(model, X_train, X_test, dv):
    
    feat_names = dv.get_feature_names() # call and append list of feat names in svm_clf as you vectorize features.
    #svm_explainer = shap.KernelExplainer(model.decision_function, X_train) # or do _predict_proba_lr if not using calibrated model.?
    #svm_shap_values = svm_explainer.shap_values(X_test, nsamples=100) 
    
    # How to set feature names to something interpretable?
    
    lin_explainer = shap.LinearExplainer(model, X_train)
    lin_shap_values = lin_explainer.shap_values(X_test)
    
    shap.summary_plot(lin_shap_values, X_test, feature_names = feat_names ) # summary plot - missing feature names.
    #shap.force_plot(lin_explainer.expected_value, lin_shap_values, X_test) # collective force plot
    return lin_explainer, lin_shap_values
    

#%%
# ANCHOR Classifier with engineered features.

def svm_clf(baseline_data, feature_data, feat_params, develop='development', experiment_name='', m=None, scaling=False, gs=False):
    
    # append engineered features both to X_train and? X_test
    
    assert len(experiment_name) > 0, print('Supply a name for the experiment!')
    # 0. Setup mlflow experiment
    # sklearn autologging.
    # disable signature logging since data is csr matrices.
    mlflow.sklearn.autolog(log_model_signatures=False)
    
    # set experiment name to organize runs
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # # set path to log data, e.g., mlruns local folder
    mlflow.set_tracking_uri('./mlruns')
    
        
    if m == None:
         model = svm.LinearSVC(max_iter=20000, penalty='l2',
                               dual=False, loss='squared_hinge', C=0.1, verbose=1)
        # model = lightning.classification.LinearSVC()
    else: 
        model = m
        
    # Grid search
    if gs == True:
        params = {'max_iter' : [5000], 'C' : [0.1]}
        model = GridSearchCV(model, params, n_jobs=-1,  return_train_score=True, verbose=1) #-> clf.fit(X_train, y_train)
        # inspect: sorted(clf.cv_results_.keys())
                        
    # Update params in an existing model like so: model.setparams(**{'param': value})
  
    
    #return experiment.name
# 1. Run mlflow experiment.
    print('Running experiment')
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    
# 2. Feature extraction. 
        print('Extracting features')
        # 2.2 Extract character ngrams
        # Unpack data (for char ngram extraction ang getting labels.)
        train_text, y_train, test_text, y_test = baseline_data 
        # Unpack data for feature extraction.
        train_feat_data, test_feat_data = feature_data
        
        
        # Sample all data and labels during development.
        # train_text = train_text.iloc[:50]
        # y_train = y_train[:50]
        # test_text = test_text.iloc[:50]
        # y_test = y_test[:50]
        
        # train_feat_data = train_feat_data.iloc[:50, :]
        # test_feat_data = test_feat_data.iloc[:50, :]
        
        
        
        # Sample all data and labels to check robustness of style features.
        # train_quarter = int(len(train_feat_data)*0.25)
        # train_text = train_text.iloc[:train_quarter]
        # y_train = y_train[:train_quarter]
        # test_text = test_text.iloc[:train_quarter]
        # y_test = y_test[:train_quarter]
        
        # train_feat_data = train_feat_data.iloc[:train_quarter, :]
        # test_feat_data = test_feat_data.iloc[:train_quarter, :]
        
        
        
        # char_cv is the char trigram CountVectorizer
        # Catch the key error when removing the 'ngrams' setting for some experiments. 
        if 'ngrams' in feat_params.keys():
            print('Extracting character trigrams')
            char_cv, train_char_trigrams, test_char_trigrams = ngrams(train_text, test_text, feat_params)
        
        
        # 2.3.1 Extract lexical, syntactic and structural (and LIWC) features.  also extract dictvectorizer for use with imp_feats()
        # Train data
        print('Extracting features from train data')
        posgram_cv, train_posgram_mat, train_en_feats = extract_feats(train_feat_data, feature_pms=feat_params, analyze=False, train=True)
            
        # Test data
        print('Extracting features from test data')
        posgram_cv, test_posgram_mat, test_en_feats = extract_feats(test_feat_data, feature_pms=feat_params, analyze=False, train=False, cv=posgram_cv)
        
        
        # 2.3.2 Vectorize engineered features.  this is where you return all of the DictVectorizers for each feature.
        print('Vectorizing extracted features.')
        # Train
        print('Vectorizing train')
        dv, X_train = d_vectorize(train_en_feats, train=True)
        # Test
        print('Vectorizing test')
        dv, X_test = d_vectorize(test_en_feats, train=False, dv = dv)
        
        
        # 2.3.4 concatenate feature vectors
        # Concat pgram matrix if pos_ngrams in feat params - otherwise just do chartrigrams and engineered features.
        # if 'ngrams' in feat_params.keys(): 
        #     X_train_combined = conc_features([X_train, train_en_feats_vec, train_posgram_mat])
        #     X_train_combined = conc_features([X_test, test_en_feats_vec, test_posgram_mat])
        
        # NOTE all the X_train / X_test below here (except inside conc_features calls) used to be X_train/test_combined.
        #print('Concatenating feature vectors.')
        if 'ngrams' in feat_params.keys():
            print('Concating character trigrams')
            # Include char trigrams - char trigrams + engineered features.
            X_train = conc_features([X_train, train_char_trigrams])
            X_test = conc_features([X_test, test_char_trigrams])
        
        # Including posgrams.
        # if 'pos_ngrams' in feat_params['engineered']['sentence_level']: #NOTE this assumes that you are using char trigrams.
        #     print('Concatenating pos_ngrams')
        #     X_train = conc_features([X_train, train_posgram_mat])
        #     X_test = conc_features([X_test, test_posgram_mat])
            
        if 'sentence_level' in feat_params['engineered'].keys():
            if 'pos_ngrams' in feat_params['engineered']['sentence_level']:
                    print('Concatenating pos_ngrams')
                    X_train = conc_features([X_train, train_posgram_mat])
                    X_test = conc_features([X_test, test_posgram_mat])

         
          
        print(f'Train ({type(X_train)}) feature matrix has shape: {X_train.shape}')
        print(f'Test ({type(X_test)}) feature matrix has shape: {X_test.shape}')
            
        #return model, X_train, X_test, y_train, y_test
        # Replace any nans.
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

        if np.isnan(X_train.data).any() == True:
            # print(f'Train feature matrix has {np.isnan(X_train.data).sum()} NaN values - compensating.') 
            #X_train.data = np.nan_to_num(X_train.data) 
            #X_train.eliminate_zeros() 
            X_train = imp_mean.fit_transform(X_train)
        if np.isnan(X_test.data).any() == True:
            # print(f'Test feature matrix has {np.isnan(X_test.data).sum()} NaN values - compensating.')
            #X_test.data = np.nan_to_num(X_test.data)
            #X_test.eliminate_zeros()
            X_test = imp_mean.transform(X_test) # fit_transform
        # Call X.shape / X.data.shape to inspect shapes
        # Call X.eliminate.zeros() to remove zeros inplace (which shrinks the matrix but is a bad idea.)
        
        # This is where you would scale the inputs.
        if scaling == True:
            scaler = StandardScaler(with_mean=False) 
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            
       
            
# 3. Train the model
        print('Training model')
        model.fit(X_train, y_train)
        
        
# 4. Make predictions
        print('Making predictions on test data')
        y_pred = model.predict(X_test)    
        
        # 4.1 add confidence scores.
        y_pred_conf = model.decision_function(X_test)
        
        # 4.2 Calibrate model to get prediction probabilities.
        calibrated = CalibratedClassifierCV(model, cv='prefit')
        calibrated = calibrated.fit(X_train, y_train)
        
        # Get probabilities
        y_pred_prob = calibrated.predict_proba(X_test)
        
        # Quick model evaluation.
        print('Evaluating prediction performance')
        evaluation_dict = report_classifier(y_pred, y_test)[0]
        evaluation_text = report_classifier(y_pred, y_test)[1]
            
        print(evaluation_text)
    
    # Visualize with shap?
        # svm_explainer = shap.KernelExplainer(calibrated.predict_proba, X_test) # or do X_train?
        # svm_shap_values = svm_explainer.shap_values(X_test) 
        # shap.summary_plot(svm_shap_values, X_test)
    
    # Log feature parameters #TODO uncomment
        mlflow.log_params({'n train instances': X_train.shape[0],\
                            'n train features': X_train.shape[1],\
                            'n test instances': X_test.shape[0],\
                            'n test features' : X_test.shape[1],\
                            'scaling': scaling})
        
        # Try to log feat_params dict
        #mlflow.log_params(feat_params.keys())
        #mlflow.log_params(feat_params['engineered'])
        mlflow.log_dict(feat_params , 'feature_parameters.json')
        
        # log evaluation of prediction to files
        mlflow.log_dict(evaluation_dict, 'pred_eval_scores.yaml')
        mlflow.log_text(evaluation_text, 'pred_eval_pretty.txt')
        
        mlflow.log_metrics(evaluation_dict['macro avg'])
        
        # Log confusion matrix for error analysis.
        # TODO add confusion matrix from evaluation.py
        
        # Save predictions to csv.
        predictions = pd.DataFrame({'text': test_text,
                                    'predicted_label': y_pred,
                                    'pred_probability': list(y_pred_prob),
                                    'pred_confidence': list(y_pred_conf),
                                    'true_label': y_test})
        
        #return predictions
        
        try:
            predictions.to_csv(f'./models/classifier/{develop}/{experiment.name}/predictions.csv', index=False)
        except:
            pass
        
        # Log predictions as an artifact file in mlflow.
        try:
            mlflow.log_artifact(f'./models/classifier/{develop}/{experiment.name}/predictions.csv')
        except:
            pass
        # fetch logged data
        #params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
            
        # Save model
        model_path = f"models/classifier/{develop}/{experiment.name}/model/"
        try:
            mlflow.sklearn.save_model(model, model_path)
        except:  # MlflowException
            print(f'The path, {model_path} already exists. Skipping save.')
            pass
    
        # Save vectorizer 
        try:
            pickle.dump(dv, open(f'{model_path}/dict_vectorizer.pkl', 'wb'))
        except:
            pass
# 5. Return
        print("DONE!")
        return model, (X_train, X_test), (y_train, y_test, y_pred, y_pred_prob, y_pred_conf), dv
#%%
#feature_parameters = json.load(open('./feature_params.json'))

    



#%%
# Feature selection
# sklearn.feature_selection._from_model
# sklearn.feature_selection.SelectKBest

if __name__ == "__main__":    

#%%
# Load data - development/validation case.
    # Baseline case
    dev_data_baseline = load_data('../../../Data/train/train.pkl',
                     '../../../Data/train/train_labels.pkl',
                     '../../../Data/validation/validation.pkl',
                     '../../../Data/validation/val_labels.pkl', baseline=True)
                     
    dev_data_clf = load_data('../../../Data/train/train.pkl',
                     '../../../Data/train/train_labels.pkl',
                     '../../../Data/validation/validation.pkl',
                     '../../../Data/validation/val_labels.pkl', baseline=False)


# Load data - final test case.
    data_baseline = load_data('../../../Data/train/train.pkl',
                     '../../../Data/train/train_labels.pkl',
                     '../../../Data/test/test.pkl',
                     '../../../Data/test/test_labels.pkl', baseline=True)

    data_clf = load_data('../../../Data/train/train.pkl',
                     '../../../Data/train/train_labels.pkl',
                     '../../../Data/test/test.pkl',
                     '../../../Data/test/test_labels.pkl', baseline=False)

# ANCHOR RUN THE MODEL.
#%%
    # 1. Baseline 
    # 1.1 org svm/randforest 'baseline'. ✅
    
    feature_parameters = json.load(open('./feature_params.json')) #
    model = svm.LinearSVC(max_iter=50000, penalty='l2', dual=True, loss='squared_hinge', C=0.1, verbose=1)
    
    #dev_results_cgrams_only = baseline(m_type='svm', data=dev_data_baseline, feat_params=feature_parameters, scaling=False, gs=False)#, m = model)
    # Final results. 
    results_cgrams_only = baseline(m_type='svm', data=data_baseline, feat_params=feature_parameters, scaling=False, gs=False)
    
    # Randomforest
    dev_results_cgrams_only = baseline(m_type='randomforest', data=dev_data_baseline, feat_params=feature_parameters, scaling=False, gs=False)
#%% 
    # # 1.2 Naive bayes - + word unigrams + new data splits. ✅
    feature_parameters = json.load(open('./feature_params_nb.json'))
    #dev_baseline_results = bayesline( data=dev_data_baseline, feat_params=feature_parameters, scaling=False)
    # Final results 
    baseline_results = bayesline(data=data_baseline, feat_params=feature_parameters, scaling=False)
   


#%%
    # 2. Feature experiments.
    
    # Parse LIWC if using it as features.
    if feature_parameters['liwc'] == True:
        parsed_liwc = parse_liwc('../../../Data/LIWC2007dictionary poster.xls')
    
    
    # Run experiments on development and test data.
    
    # 2.1.1.1 Full case no posgrams - best performance on dev. ✅
    feature_parameters = json.load(open('./feature_params_fc_no_pgrams.json'))
    model = svm.LinearSVC(max_iter=25000, penalty='l2', dual=False, loss='squared_hinge', C=0.1, verbose=1) # old version works with dual=True, loss=hinge, scaling=True
    
    
    #results = svm_clf(dev_data_baseline, dev_data_clf, feature_parameters, develop='development', experiment_name='classifier_full_case_no_posgrams', scaling=False, gs=False, m=model)
    # Final results
    results = svm_clf(data_baseline, data_clf, feature_parameters, develop='final', experiment_name='classifier_full_case_no_posgrams', scaling=False, gs=False, m=model)
    
    
    # 2.1.1.2 Full case no posgrams without own features ✅
    feature_parameters = json.load(open('./feature_params_fc_without_own_features.json'))
    model = svm.LinearSVC(max_iter=25000, penalty='l2', dual=False, loss='squared_hinge', C=0.1, verbose=1)
    # Final results
    results = svm_clf(data_baseline, data_clf, feature_parameters, develop='final', experiment_name='classifier_full_case_without_own_features', scaling=False, gs=False, m=model)
    
    
    # 2.1.2 Full case own features ✅
    feature_parameters = json.load(open('./feature_params_fc_own_features.json'))
    model = svm.LinearSVC(max_iter=25000, penalty='l2', dual=False, loss='squared_hinge', C=0.1, verbose=1)
    # Final results
    results = svm_clf(data_baseline, data_clf, feature_parameters, develop='final', experiment_name='classifier_full_case_own_features', scaling=False, gs=False, m=model)
    
    
    # 2.2 Full case with posgrams  ✅
    feature_parameters = json.load(open('./feature_params_fc_with_pgrams.json'))
    model = svm.LinearSVC(max_iter=25000, penalty='l2', dual=False, loss='squared_hinge', C=0.1, verbose=1) # old version works with scaling, dual=True loss=hinge
                              
    results_all_feats = svm_clf(dev_data_baseline, dev_data_clf, feature_parameters,develop='development', experiment_name='classifier_full_case_with_posgrams', m=model, scaling=False, gs=False) # scaling= False doesn't converge
    # Final results
    results_all_feats = svm_clf(data_baseline, data_clf, feature_parameters,develop='final', experiment_name='classifier_full_case_with_posgrams', m=model, scaling=False, gs=False)
    
    # 2.4 Base case with posgrams ✅
    feature_parameters = json.load(open('./feature_params_bc_with_pgrams.json'))
    model = svm.LinearSVC(max_iter=40000, penalty='l2', dual=False, loss='squared_hinge', C=0.1, verbose=1)
    results = svm_clf(dev_data_baseline, dev_data_clf, feature_parameters, develop ='development', experiment_name='classifier_base_case_with_posgrams', m=model, scaling=False, gs=False) # dual=True doesn't converge
    # Final results
    results = svm_clf(data_baseline, data_clf, feature_parameters, develop ='final', experiment_name='classifier_base_case_with_posgrams', m=model, scaling=False, gs=False) 
    
    # Test style features alone 
    results = svm_clf(data_baseline, data_clf, feature_parameters, develop ='final', experiment_name='classifier_base_case_with_posgrams', m=model, scaling=False, gs=False) 
    
    # 2.5.1 Base case no posgrams.   ✅
    feature_parameters = json.load(open('./feature_params_bc_no_pgrams.json'))
    model = svm.LinearSVC(max_iter=40000, penalty='l2',  dual=False, loss='squared_hinge', C=0.1, verbose=1) #old version : dual false, squared hinge, sclaing=True.
    results_feats_only = svm_clf(dev_data_baseline, dev_data_clf, feature_parameters, develop='development', experiment_name='classifier_base_case_no_posgrams', m = model, scaling=False) # dual=True doesn't converge?
    # Final results
    results_feats_only = svm_clf(data_baseline, data_clf, feature_parameters, develop='final', experiment_name='classifier_base_case_no_posgrams', m = model, scaling=False) # NOTE Use this for feature evaluation / selection.
    
    # 2.5.2 Base case own features ✅
    feature_parameters = json.load(open('./feature_params_bc_own_features.json'))
    model = svm.LinearSVC(max_iter=40000, penalty='l2',  dual=False, loss='squared_hinge', C=0.1, verbose=1) 
    
    # Final results
    results_feats_only = svm_clf(data_baseline, data_clf, feature_parameters, develop='final', experiment_name='classifier_base_case_own_features', m = model, scaling=False)
    
    # 2.5.3 Base case (with pos) without own features ✅
    feature_parameters = json.load(open('./feature_params_bc_without_own_features.json'))
    model = svm.LinearSVC(max_iter=40000, penalty='l2', dual=False, loss='squared_hinge', C=0.1, verbose=1)

    # Final results
    results_feats_only = svm_clf(data_baseline, data_clf, feature_parameters, develop='final', experiment_name='classifier_base_case_without_own_features', m=model, scaling=False)
    
    
    
#%%
#ANCHOR visualize result shap values
    
    # svm - all features
    #shap_vis(results_all_feats[0], results_all_feats[1][1])
    
    # svm - features only
    #shape_vis(results_feats_only[0] , results_all_feats[1][1])

#%%
# #ANCHOR Testing getting most  predictive features from experiment with style features alone.
    #model_fitted = results_feats_only[0] 
    #dv = results_feats_nly[-1]
    # this should works for both sklearn vectorizers
    #feature_names = dv.get_feature_names()
    
    # Using predictive_feats
    #pred_feats = predictive_feats(model_fitted, feature_names)
    
    # Using feature_imps
    #important_features = imp_feats(model_fitted, dv, top_features=40)
    
#%%
    #ANCHOR experiment with automatic feature selection
    # X_train = results_feats_only[1][1]
    # X_test = results_feats_only[1][2]
    
    # selector = SelectFromModel(model).fit_transform(X_train) #,n_features_to_select=35, verbose=1)
    
    
    


#%% ANCHOR Testing for overfitting on Alice in Wonderland.
    
        
    
    overfit_test = input('Test for overfitting on unrelated data? (y/n) ')
    
    if overfit_test == 'y':
        
        # # Uncomment to run the test.
        
        # Data to investigate trigram overfitting.
        alt_test = pd.read_csv('../../../Data/noncons/preprocessed/alice.csv', index_col=False)
        train_text = dev_data_baseline[0] # get the original training data
        
        # # Append a Q instance
        alt_test = alt_test.append({"text": "Alice is part of a secret elite ESTABLISHMENT!", "label": "Q"}, ignore_index=True)
        
        alt_test_text = alt_test['text'].tolist()
        alt_test_labels = alt_test['label'].tolist()
        
        # alt_fit_model, alt_predictions = pred_unrelated('./models/baseline/')
        feature_parameters = json.load(open('./feature_params.json'))
        alt_fit_model, alt_predictions = pred_unrelated('./models/baseline/svm/unscaled', feature_pms=feature_parameters)
        alt_fit_model, alt_predictions = pred_unrelated('./models/baseline/randomforest/unscaled/' , feature_pms=feature_parameters)
        
        feature_parameters = json.load(open('./feature_params_nb.json'))
        alt_fit_model, alt_predictions = pred_unrelated('./models/baseline/nb/unscaled/' , feature_pms=feature_parameters)
        
        alt_results = alt_test.copy(deep=True)
        alt_results['predicted_label'] = alt_predictions
        
        
        
        
        
        # #%%
        # # save predictions.
        alt_results.to_csv(
            './models/baseline/overfit/alt_predictions.csv', index=False)
        


#%% Manual evaluation for debugging.
    # e_counts = eval_counts(alt_test_labels, alt_predictions)
    
    # man_eval = cal_acc(e_counts)

# %%

    # Inspecting alt_predictions...
    #alt_results.query("label != predicted_label and predicted_label == 'Q'")
    #alt_results.query("label == predicted_label and predicted_label == 'Q'")
# %%

                    
# %%
