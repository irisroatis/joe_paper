import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.models import Model
from functools import partial
from category_encoders.glmm import GLMMEncoder
from category_encoders.ordinal import OrdinalEncoder
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
import pickle


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, brier_score_loss


def target_encoding(feature, target, df):
    dictionary_target_encoding = {}
    categories = df[feature].unique()
    changed_df = df.copy()
    for cat in categories:
        which_cat = df[df[feature] == cat]
        # For binary classification, this calculates the proportion of the positive class
        avg_value = np.sum(which_cat[target]) / len(which_cat)
        changed_df[feature] = changed_df[feature].replace([cat], avg_value)
        dictionary_target_encoding[cat] = avg_value
    return changed_df, dictionary_target_encoding

def split_inputs(x, y, categorical_variables):
    how_many_cat_variables = len(categorical_variables)
    dictionary = {}
    for index in range(how_many_cat_variables):
        dictionary[categorical_variables[index]] = [x[index], x[how_many_cat_variables+index]]
    dictionary['rest'] = x[2*how_many_cat_variables:]
    return (dictionary, y)

def split_inputs_encoder(x, y):
    return ({'all':x}, y)

def split_inputs_onehot(x, y, categorical_variables, how_many_cat_percolumn):
    how_many_cat_variables = len(categorical_variables)
    dictionary = {}
    how_many_so_far = 0
    for index in range(how_many_cat_variables):
        dictionary[categorical_variables[index]] = x[how_many_so_far : how_many_so_far + how_many_cat_percolumn[index]]
        how_many_so_far += how_many_cat_percolumn[index]
    dictionary['rest'] = x[how_many_so_far:]
    return (dictionary, y)

class MyLayer(Layer):
    def call(self, small_models_outputs, inputs):
        return tf.concat([*small_models_outputs.values(), inputs['rest']], axis=-1)

class SmallNetwork(Layer):
    def __init__(self, units, activation='sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.dense_layers = []
        for units in self.units:
            self.dense_layers.append(Dense(units, activation=self.activation, kernel_initializer=tf.keras.initializers.Zeros()))

    def call(self, inputs):
        h = inputs
        for dense_layer in self.dense_layers:
            h = dense_layer(h)
        return h

def calculate_relative_entropy(series):
  value_counts = series.value_counts(normalize=True)
  entropy = -np.sum(value_counts * np.log2(value_counts))
  cardinality = len(value_counts)
  if cardinality <= 1:
    return 0.0
  return entropy / np.log2(cardinality)

# --- Simulation Parameters ---
beta_1 = 0.1
beta_2 = -0.15
beta_3 = 0.2
beta_4 = 0.3
total_epochs = 250

which_open = '/home/ir318/trial_experiments/'

print('NEW ONE')
dictionary_results = {}
dictionary_results_tarreg = {}
dictionary_results_joechar1 = {}
dictionary_results_joechar2 = {}
dictionary_results_joeohe1 = {}
dictionary_results_joeohe2 = {}
dictionary_results_woe = {}
dictionary_results_glmm = {}
dictionary_results_onehot = {}
dictionary_results_ordinal = {}
dictionary_results_continuous = {'est_b1':[],'est_b2':[],'est_b3':[]}
dictionary_results_nocat = {'est_b1':[],'est_b2':[],'est_b3':[]}
dictionary_nr_obs_cat = {}
dictionary_relative_entropy = {}
dictionary_class_proportion = {}

dictionary_performance = {}
dictionary_performance_tarreg = {}
dictionary_performance_joechar1 = {}
dictionary_performance_joechar2 = {}
dictionary_performance_joeohe1 = {}
dictionary_performance_joeohe2 = {}
dictionary_performance_glmm = {}
dictionary_performance_onehot = {}
dictionary_performance_woe = {}
dictionary_performance_ordinal = {}
dictionary_performance_continuous = {'accuracy':[],'auc':[],'f1':[], 'logloss':[], 'brier': []} # Changed
dictionary_performance_nocat = {'accuracy':[],'auc':[],'f1':[], 'logloss':[], 'brier':[]} # Changed

# --- Data Simulation (Modified for Binary Target) ---
X = np.random.multivariate_normal([0,0,0], np.identity(3),20000)
x_1 = X[:,0]
x_2 = X[:,1]
x_3 = X[:,2]

# Generate a binary target variable
# Using a sigmoid to transform a linear combination into a probability, then binarize
linear_combination = beta_1 * x_1 + beta_2 * x_2 + beta_3 * x_3 + beta_4 * x_1 * x_2
# linear_combination = beta_1 * x_1 + beta_2 * x_2 + beta_3 * x_3
probabilities = 1 / (1 + np.exp(-linear_combination + np.random.normal(0, 0.3, 20000))) # Add some noise
y = np.random.binomial(1, probabilities).astype(int) # Bernoulli trials

target_variable = 'y'
categorical_variables = ['x_1']
continuous_variables = ['x_2', 'x_3']
binary_variables = []

indices_test = random.sample(range(20000), 10000)

list_nr_bins = [50, 100, 150, 200, 250, 300]
# list_nr_bins = [50, 150, 350]

for each_nr_bins in list_nr_bins:
    dictionary_results[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}
    dictionary_results_tarreg[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}
    dictionary_results_joechar1[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}
    dictionary_results_joechar2[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}
    dictionary_results_joeohe1[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}
    dictionary_results_joeohe2[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}
    dictionary_results_onehot[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}
    dictionary_results_glmm[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}
    dictionary_results_woe[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}
    dictionary_results_ordinal[each_nr_bins] = {'est_b1':[],'est_b2':[],'est_b3':[]}

    dictionary_performance[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_performance_tarreg[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_performance_joechar1[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_performance_joechar2[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_performance_glmm[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_performance_joeohe1[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_performance_joeohe2[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_performance_onehot[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_performance_woe[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_performance_ordinal[each_nr_bins] = {'accuracy':[],'auc':[],'f1':[], 'logloss':[],'correlation':[], 'brier':[]} # Changed
    dictionary_nr_obs_cat[each_nr_bins] = []
    dictionary_class_proportion[each_nr_bins] = []
    dictionary_relative_entropy[each_nr_bins] = []


for i in range(50):
    print('REPETITION NUMBER '+str(i+1)+' of 50')

    for nr_bins_x1 in list_nr_bins:
        print(nr_bins_x1)

        labels_x1 = [f'C_{{1,{i}}}' for i in range(nr_bins_x1)]

        x1_binned = pd.cut(x_1, bins=nr_bins_x1, labels=labels_x1).astype(str)

        df = pd.DataFrame({'x_1': x1_binned, 'x_2': x_2, 'x_3': x_3, 'y': y})
        df_test = df.iloc[indices_test]
        df_all_train = df.drop(indices_test)
        df_train = df_all_train.sample(n = 1000)

        train_sample_indices = df_train.index

        df_cont = pd.DataFrame({'x_1': x_1, 'x_2': x_2, 'x_3': x_3, 'y': y})
        df_test_cont = df_cont.iloc[indices_test]
        df_all_train_cont = df_cont.drop(indices_test)
        df_train_cont = df_all_train_cont.loc[train_sample_indices]

        nr_obs_cat =len(df_train['x_1'].unique())
        dictionary_nr_obs_cat[nr_bins_x1].append(nr_obs_cat)
        
        proportion_ones_train = df_train[target_variable].mean()
        dictionary_class_proportion[nr_bins_x1].append(proportion_ones_train)


        relative_entropy = calculate_relative_entropy(df_train['x_1'])
        dictionary_relative_entropy[nr_bins_x1].append(relative_entropy)

        #### CONTINUOUS
 
        from sklearn.linear_model import LogisticRegression
        model_cont = LogisticRegression(penalty = None) # Using liblinear for small datasets and binary problems
        model_cont.fit(df_train_cont.drop('y', axis=1), df_train_cont['y'])
        y_pred_proba_cont = model_cont.predict_proba(df_test_cont.drop('y', axis=1))[:, 1]
        y_pred_cont = model_cont.predict(df_test_cont.drop('y', axis=1)) # For accuracy/f1
        dictionary_results_continuous['est_b1'].append(model_cont.coef_[0][0])
        dictionary_results_continuous['est_b2'].append(model_cont.coef_[0][1])
        dictionary_results_continuous['est_b3'].append(model_cont.coef_[0][2])
        dictionary_performance_continuous['accuracy'].append(accuracy_score(df_test_cont['y'], y_pred_cont))
        dictionary_performance_continuous['auc'].append(roc_auc_score(df_test_cont['y'], y_pred_proba_cont))
        dictionary_performance_continuous['f1'].append(f1_score(df_test_cont['y'], y_pred_cont))
        dictionary_performance_continuous['logloss'].append(log_loss(df_test_cont['y'], y_pred_proba_cont))
        dictionary_performance_continuous['brier'].append(brier_score_loss(df_test_cont['y'], y_pred_proba_cont))


        ### no categorical features
        model_nocat = LogisticRegression(penalty = None)
        model_nocat.fit(df_train.drop(['x_1','y'], axis=1), df_train['y'])
        y_pred_proba_nocat = model_nocat.predict_proba(df_test.drop(['x_1','y'], axis=1))[:, 1]
        y_pred_nocat = model_nocat.predict(df_test.drop(['x_1','y'], axis=1))
        dictionary_results_nocat['est_b1'].append(np.nan) # x1 is dropped
        dictionary_results_nocat['est_b2'].append(model_nocat.coef_[0][0])
        dictionary_results_nocat['est_b3'].append(model_nocat.coef_[0][1])

        dictionary_performance_nocat['accuracy'].append(accuracy_score(df_test['y'], y_pred_nocat))
        dictionary_performance_nocat['auc'].append(roc_auc_score(df_test['y'], y_pred_proba_nocat))
        dictionary_performance_nocat['f1'].append(f1_score(df_test['y'], y_pred_nocat))
        dictionary_performance_nocat['logloss'].append(log_loss(df_test['y'], y_pred_proba_nocat))
        dictionary_performance_nocat['brier'].append(brier_score_loss(df_test['y'], y_pred_proba_nocat))

        ### SIMPLE TARGET
        target_df = df_train.copy()
        target_df_test = df_test.copy()

        prior = np.mean(df_train[target_variable])
        for col in categorical_variables:
            target_df, dict_target =  target_encoding(col, target_variable, target_df)
            target_df[col] = target_df[col].astype('float')
            target_df_test[col] = target_df_test[col].replace(list(dict_target.keys()), list(dict_target.values()))
            unique_test_no_train = list(set(df_test[col]) - set(df_train[col]))
            for uni in unique_test_no_train:
                target_df_test.loc[target_df_test[col] == uni, col] = prior
            target_df_test[col] = target_df_test[col].astype('float')
        X_train_target = target_df.drop('y', axis=1)
        y_train_target = target_df['y']
        X_test_target = target_df_test.drop('y', axis=1)
        y_test_target = target_df_test['y']


        model = LogisticRegression(penalty = None)
        model.fit(X_train_target, y_train_target)
        y_pred_proba = model.predict_proba(X_test_target)[:, 1]
        y_pred = model.predict(X_test_target)

        dictionary_results[nr_bins_x1]['est_b1'].append(model.coef_[0][0])
        dictionary_results[nr_bins_x1]['est_b2'].append(model.coef_[0][1])
        dictionary_results[nr_bins_x1]['est_b3'].append(model.coef_[0][2])
        dictionary_performance[nr_bins_x1]['accuracy'].append(accuracy_score(y_test_target, y_pred))
        dictionary_performance[nr_bins_x1]['auc'].append(roc_auc_score(y_test_target, y_pred_proba))
        dictionary_performance[nr_bins_x1]['f1'].append(f1_score(y_test_target, y_pred))
        dictionary_performance[nr_bins_x1]['logloss'].append(log_loss(y_test_target, y_pred_proba))
        dictionary_performance[nr_bins_x1]['correlation'].append(np.corrcoef(X_train_target[categorical_variables[0]], df_train_cont[categorical_variables[0]])[0,1])
        dictionary_performance[nr_bins_x1]['brier'].append(brier_score_loss(y_test_target, y_pred_proba))

        ### GLMM
        encoder_glmm = GLMMEncoder(cols=['x_1'])
        encoder_glmm.fit(df_train[['x_1']], df_train['y'])
        df_train_encoded_glmm = df_train.copy()
        df_test_encoded_glmm = df_test.copy()
        df_train_encoded_glmm[['x_1']] = encoder_glmm.transform(df_train[['x_1']])
        df_test_encoded_glmm[['x_1']] = encoder_glmm.transform(df_test[['x_1']])
        X_train_glmm = df_train_encoded_glmm.drop('y', axis=1)
        y_train_glmm = df_train_encoded_glmm['y']
        X_test_glmm = df_test_encoded_glmm.drop('y', axis=1)
        y_test_glmm = df_test_encoded_glmm['y']

        model_glmm = LogisticRegression(penalty = None)
        model_glmm.fit(X_train_glmm, y_train_glmm)
        y_pred_proba_glmm = model_glmm.predict_proba(X_test_glmm)[:, 1]
        y_pred_glmm = model_glmm.predict(X_test_glmm)

        dictionary_results_glmm[nr_bins_x1]['est_b1'].append(model_glmm.coef_[0][0])
        dictionary_results_glmm[nr_bins_x1]['est_b2'].append(model_glmm.coef_[0][1])
        dictionary_results_glmm[nr_bins_x1]['est_b3'].append(model_glmm.coef_[0][2])
        dictionary_performance_glmm[nr_bins_x1]['accuracy'].append(accuracy_score(y_test_glmm, y_pred_glmm))
        dictionary_performance_glmm[nr_bins_x1]['auc'].append(roc_auc_score(y_test_glmm, y_pred_proba_glmm))
        dictionary_performance_glmm[nr_bins_x1]['f1'].append(f1_score(y_test_glmm, y_pred_glmm))
        dictionary_performance_glmm[nr_bins_x1]['logloss'].append(log_loss(y_test_glmm, y_pred_proba_glmm))
        dictionary_performance_glmm[nr_bins_x1]['correlation'].append(np.corrcoef(X_train_glmm[categorical_variables[0]], df_train_cont[categorical_variables[0]])[0,1])
        dictionary_performance_glmm[nr_bins_x1]['brier'].append(brier_score_loss(y_test_glmm, y_pred_proba_glmm))


        ### TAR REGULARISATION
        target_encoder = TargetEncoder(target_type='continuous', smooth='auto', cv=5)
        target_encoder.fit(df_train[['x_1']], df_train['y'])
        df_train_encoded_tarreg = df_train.copy()
        df_test_encoded_tarreg = df_test.copy()
        df_train_encoded_tarreg[['x_1']] = target_encoder.transform(df_train[['x_1']])
        df_test_encoded_tarreg[['x_1']] = target_encoder.transform(df_test[['x_1']])
        X_train_tarreg = df_train_encoded_tarreg.drop('y', axis=1)
        y_train_tarreg = df_train_encoded_tarreg['y']
        X_test_tarreg = df_test_encoded_tarreg.drop('y', axis=1)
        y_test_tarreg = df_test_encoded_tarreg['y']

        model_tarreg = LogisticRegression(penalty = None)
        model_tarreg.fit(X_train_tarreg, y_train_tarreg)
        y_pred_proba_tarreg = model_tarreg.predict_proba(X_test_tarreg)[:, 1]
        y_pred_tarreg = model_tarreg.predict(X_test_tarreg)

        dictionary_results_tarreg[nr_bins_x1]['est_b1'].append(model_tarreg.coef_[0][0])
        dictionary_results_tarreg[nr_bins_x1]['est_b2'].append(model_tarreg.coef_[0][1])
        dictionary_results_tarreg[nr_bins_x1]['est_b3'].append(model_tarreg.coef_[0][2])
        dictionary_performance_tarreg[nr_bins_x1]['accuracy'].append(accuracy_score(y_test_tarreg, y_pred_tarreg))
        dictionary_performance_tarreg[nr_bins_x1]['auc'].append(roc_auc_score(y_test_tarreg, y_pred_proba_tarreg))
        dictionary_performance_tarreg[nr_bins_x1]['f1'].append(f1_score(y_test_tarreg, y_pred_tarreg))
        dictionary_performance_tarreg[nr_bins_x1]['logloss'].append(log_loss(y_test_tarreg, y_pred_proba_tarreg))
        dictionary_performance_tarreg[nr_bins_x1]['correlation'].append(np.corrcoef(X_train_tarreg[categorical_variables[0]], df_train_cont[categorical_variables[0]])[0,1])
        dictionary_performance_tarreg[nr_bins_x1]['brier'].append(brier_score_loss(y_test_tarreg, y_pred_proba_tarreg))

        ### ordinal encoding
        encoder_oe = OrdinalEncoder(cols=['x_1'])
        encoder_oe.fit(df_train[['x_1']], df_train['y'])
        df_train_encoded_oe = df_train.copy()
        df_test_encoded_oe = df_test.copy()
        df_train_encoded_oe[['x_1']] = encoder_oe.transform(df_train[['x_1']])
        df_test_encoded_oe[['x_1']] = encoder_oe.transform(df_test[['x_1']])
        X_train_oe = df_train_encoded_oe.drop('y', axis=1)
        y_train_oe = df_train_encoded_oe['y']
        X_test_oe = df_test_encoded_oe.drop('y', axis=1)
        y_test_oe = df_test_encoded_oe['y']

        model_oe = LogisticRegression(penalty = None)
        model_oe.fit(X_train_oe, y_train_oe)
        y_pred_proba_oe = model_oe.predict_proba(X_test_oe)[:, 1]
        y_pred_oe = model_oe.predict(X_test_oe)

        dictionary_results_ordinal[nr_bins_x1]['est_b1'].append(model_oe.coef_[0][0])
        dictionary_results_ordinal[nr_bins_x1]['est_b2'].append(model_oe.coef_[0][1])
        dictionary_results_ordinal[nr_bins_x1]['est_b3'].append(model_oe.coef_[0][2])
        dictionary_performance_ordinal[nr_bins_x1]['accuracy'].append(accuracy_score(y_test_oe, y_pred_oe))
        dictionary_performance_ordinal[nr_bins_x1]['auc'].append(roc_auc_score(y_test_oe, y_pred_proba_oe))
        dictionary_performance_ordinal[nr_bins_x1]['f1'].append(f1_score(y_test_oe, y_pred_oe))
        dictionary_performance_ordinal[nr_bins_x1]['logloss'].append(log_loss(y_test_oe, y_pred_proba_oe))
        dictionary_performance_ordinal[nr_bins_x1]['correlation'].append(np.corrcoef(X_train_oe[categorical_variables[0]], df_train_cont[categorical_variables[0]])[0,1])
        dictionary_performance_ordinal[nr_bins_x1]['brier'].append(brier_score_loss(y_test_oe, y_pred_proba_oe))

        ### one hot encoding
        encoder_onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder_onehot.fit(df_train[['x_1']])

        encoded_train_x1 = encoder_onehot.transform(df_train[['x_1']])
        encoded_test_x1 = encoder_onehot.transform(df_test[['x_1']])

        df_train_encoded_x1 = pd.DataFrame(encoded_train_x1, columns=encoder_onehot.get_feature_names_out(['x_1']), index=df_train.index)
        df_test_encoded_x1 = pd.DataFrame(encoded_test_x1, columns=encoder_onehot.get_feature_names_out(['x_1']), index=df_test.index)

        df_train_encoded_onehot = pd.concat([df_train.drop('x_1', axis=1), df_train_encoded_x1], axis=1)
        df_test_encoded_onehot = pd.concat([df_test.drop('x_1', axis=1), df_test_encoded_x1], axis=1)

        X_train_onehot = df_train_encoded_onehot.drop('y', axis=1)
        y_train_onehot = df_train_encoded_onehot['y']
        X_test_onehot = df_test_encoded_onehot.drop('y', axis=1)
        y_test_onehot = df_test_encoded_onehot['y']

        model_onehot = LogisticRegression(penalty = None)
        model_onehot.fit(X_train_onehot, y_train_onehot)
        y_pred_proba_onehot = model_onehot.predict_proba(X_test_onehot)[:, 1]
        y_pred_onehot = model_onehot.predict(X_test_onehot)

        dictionary_performance_onehot[nr_bins_x1]['accuracy'].append(accuracy_score(y_test_onehot, y_pred_onehot))
        dictionary_performance_onehot[nr_bins_x1]['auc'].append(roc_auc_score(y_test_onehot, y_pred_proba_onehot))
        dictionary_performance_onehot[nr_bins_x1]['f1'].append(f1_score(y_test_onehot, y_pred_onehot))
        dictionary_performance_onehot[nr_bins_x1]['logloss'].append(log_loss(y_test_onehot, y_pred_proba_onehot))
        dictionary_performance_onehot[nr_bins_x1]['brier'].append(brier_score_loss(y_test_onehot, y_pred_proba_onehot))

        list_columns = []
        list_datasets = {}
        dictionary_all_categorical_columns_positives = {}
        # dictionary_all_categorical_columns_negatives = {}
        dictionary_all_categorical_columns_overallmean = {}
        # dictionary_all_categorical_columns_overallstd = {}


        for which_column_to_categories in categorical_variables:
            categories = df[which_column_to_categories].unique().tolist()
            dict_whichcolumn_pos = {}
            # dict_whichcolumn_neg = {}
            dict_whichcolumn_overallmean = {}
            # dict_whichcolumn_overallstd = {}
            overall_mean = np.mean(df_train[target_variable])
            # overall_std = np.std(df_train[target_variable])
            for cat in categories:
                which_cat = df_train[df_train[which_column_to_categories] == cat]
                if which_cat.shape[0] >= 1:
                    dict_whichcolumn_pos[cat] = np.mean(which_cat[target_variable])
                    # dict_whichcolumn_neg[cat] = np.std(which_cat[target_variable])
                    dict_whichcolumn_overallmean[cat] = overall_mean
                    # dict_whichcolumn_overallstd[cat] = overall_std
                else:
                    dict_whichcolumn_pos[cat] = 0
                    # dict_whichcolumn_neg[cat] = 0
                    dict_whichcolumn_overallmean[cat] = overall_mean
                    # dict_whichcolumn_overallstd[cat] = overall_std

            dictionary_all_categorical_columns_positives[which_column_to_categories] = dict_whichcolumn_pos
            # dictionary_all_categorical_columns_negatives[which_column_to_categories] = dict_whichcolumn_neg
            dictionary_all_categorical_columns_overallmean[which_column_to_categories] = dict_whichcolumn_overallmean
            # dictionary_all_categorical_columns_overallstd[which_column_to_categories] = dict_whichcolumn_overallstd
            list_columns.append(which_column_to_categories+str('_P'))

        X_train, y_train =  df_train.drop('y', axis=1), df_train['y']
        X_test, y_test = df_test.drop('y', axis=1), df_test['y']
        X_train_mod_1 = X_train.copy()
        X_test_mod_1 = X_test.copy()

        for which_column_to_categories in categorical_variables:
            # X_train_mod_1[which_column_to_categories+str('_N')] = X_train_mod_1[which_column_to_categories].copy()
            X_train_mod_1[which_column_to_categories+str('_OM')] = X_train_mod_1[which_column_to_categories].copy()
            # X_train_mod_1[which_column_to_categories+str('_OS')] = X_train_mod_1[which_column_to_categories].copy()
            X_train_mod_1.rename(columns={which_column_to_categories: which_column_to_categories+str('_P')}, inplace=True)

            # X_test_mod_1[which_column_to_categories+str('_N')] = X_test_mod_1[which_column_to_categories].copy()
            X_test_mod_1[which_column_to_categories+str('_OM')] = X_test_mod_1[which_column_to_categories].copy()
            # X_test_mod_1[which_column_to_categories+str('_OS')] = X_test_mod_1[which_column_to_categories].copy()
            X_test_mod_1.rename(columns={which_column_to_categories: which_column_to_categories+str('_P')}, inplace=True)

            dic_pos = dictionary_all_categorical_columns_positives[which_column_to_categories]
            # dic_neg = dictionary_all_categorical_columns_negatives[which_column_to_categories]
            dic_overallmean = dictionary_all_categorical_columns_overallmean[which_column_to_categories]
            # dic_overallstd = dictionary_all_categorical_columns_overallstd[which_column_to_categories]

            test_this_column = pd.DataFrame(columns=['cat','mean','o_m'])

            for cat in dic_pos.keys():
                # n = dic_neg[cat]
                p = dic_pos[cat]
                o_mean = dic_overallmean[cat]
                # o_std = dic_overallstd[cat]
                # X_train_mod_1[which_column_to_categories+str('_N')] =    X_train_mod_1[which_column_to_categories+str('_N')].replace(cat, n)
                X_train_mod_1[which_column_to_categories+str('_P')] =    X_train_mod_1[which_column_to_categories+str('_P')].replace(cat, p)
                X_train_mod_1[which_column_to_categories+str('_OM')] =    X_train_mod_1[which_column_to_categories+str('_OM')].replace(cat, o_mean)
                # X_train_mod_1[which_column_to_categories+str('_OS')] =    X_train_mod_1[which_column_to_categories+str('_OS')].replace(cat, o_std)
                # X_test_mod_1[which_column_to_categories+str('_N')] = X_test_mod_1[which_column_to_categories+str('_N')].replace(cat, n)
                X_test_mod_1[which_column_to_categories+str('_P')] = X_test_mod_1[which_column_to_categories+str('_P')].replace(cat, p)
                X_test_mod_1[which_column_to_categories+str('_OM')] = X_test_mod_1[which_column_to_categories+str('_OM')].replace(cat, o_mean)
                # X_test_mod_1[which_column_to_categories+str('_OS')] = X_test_mod_1[which_column_to_categories+str('_OS')].replace(cat, o_std)
                test_this_column = test_this_column._append({'cat':cat,'mean': p, 'o_m': o_mean},ignore_index=True)

            list_datasets[which_column_to_categories] = test_this_column
            # list_columns.append(which_column_to_categories+str('_N'))

        for which_column_to_categories in categorical_variables:
            list_columns.append(which_column_to_categories+str('_OM'))
        # for which_column_to_categories in categorical_variables:
        #     list_columns.append(which_column_to_categories+str('_OS'))

        list_columns = list_columns+continuous_variables+binary_variables

        X_train_mod_1 = X_train_mod_1[list_columns]
        X_test_mod_1 = X_test_mod_1[list_columns]


        train_ds = tf.data.Dataset.from_tensor_slices((X_train_mod_1.values.astype(np.float32), y_train.values))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_mod_1.values.astype(np.float32),  y_test.values))
        f = partial(split_inputs, categorical_variables=categorical_variables)
        train_ds = train_ds.map(f)
        test_ds = test_ds.map(f)

        train_ds = train_ds.shuffle(500).batch(32)
        test_ds = test_ds.batch(32)

        for combination in [ [[3,1],'sigmoid'], [[1],'linear']]:
            hidden_layers = combination[0]
            encod_act = combination[1]

            small_models = {}
            inputs = {}
            for k in categorical_variables:
                small_models[k] = SmallNetwork(hidden_layers, encod_act)
                inputs[k] = Input(shape=(2,), name = str(k))

            inputs['rest'] = Input(shape=(len(binary_variables)+len(continuous_variables),), name = 'rest')
            small_models_outputs = {k: small_models[k](inputs[k]) for k in categorical_variables}
            h =  MyLayer()(small_models_outputs, inputs)

            initializer = tf.keras.initializers.GlorotUniform(seed=50)
            opt = 'adam'

            outputs = Dense(1, activation='sigmoid', kernel_initializer = initializer)(h)

            if combination == [ [3,1],'sigmoid']:
                name = 'JoeChar3Sig'
                model1 = Model(inputs=inputs, outputs=outputs)
                model1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
                history1 = model1.fit(train_ds, epochs = total_epochs, validation_data= test_ds,validation_freq=10, callbacks=[], verbose = 0)
                y_pred_proba_keras = model1.predict(test_ds).flatten()
                y_pred_keras = (y_pred_proba_keras > 0.5).astype(int)

                accuracy_joechar1 = accuracy_score(y_test.values, y_pred_keras)
                auc_joechar1 = roc_auc_score(y_test.values, y_pred_proba_keras)
                f1_joechar1 = f1_score(y_test.values, y_pred_keras)
                logloss_joechar1 = log_loss(y_test.values, y_pred_proba_keras)
                brier_joechar1 = brier_score_loss(y_test.values, y_pred_proba_keras)


                dictionary_performance_joechar1[nr_bins_x1]['accuracy'].append(accuracy_joechar1)
                dictionary_performance_joechar1[nr_bins_x1]['auc'].append(auc_joechar1)
                dictionary_performance_joechar1[nr_bins_x1]['f1'].append(f1_joechar1)
                dictionary_performance_joechar1[nr_bins_x1]['logloss'].append(logloss_joechar1)
                dictionary_performance_joechar1[nr_bins_x1]['brier'].append(brier_joechar1)

                
                dictionary_results_joechar1[nr_bins_x1]['est_b1'].append(model1.get_weights()[-2][0][0] if len(model1.get_weights()) >= 2 else np.nan)
                dictionary_results_joechar1[nr_bins_x1]['est_b2'].append(model1.get_weights()[-2][1][0] if len(model1.get_weights()) >= 2 else np.nan)
                dictionary_results_joechar1[nr_bins_x1]['est_b3'].append(model1.get_weights()[-2][2][0] if len(model1.get_weights()) >= 2 else np.nan)

            else:
                model2 = Model(inputs=inputs, outputs=outputs)
                name = 'JoeChar1Lin' 
                model2.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
                history2 = model2.fit(train_ds, epochs = total_epochs, validation_data= test_ds,validation_freq=10, callbacks=[], verbose = 0)
                y_pred_proba_keras = model2.predict(test_ds).flatten()
                y_pred_keras = (y_pred_proba_keras > 0.5).astype(int)

                accuracy_joechar2 = accuracy_score(y_test.values, y_pred_keras)
                auc_joechar2 = roc_auc_score(y_test.values, y_pred_proba_keras)
                f1_joechar2 = f1_score(y_test.values, y_pred_keras)
                logloss_joechar2 = log_loss(y_test.values, y_pred_proba_keras)
                brier_joechar2 = brier_score_loss(y_test.values, y_pred_proba_keras)


                dictionary_performance_joechar2[nr_bins_x1]['accuracy'].append(accuracy_joechar2)
                dictionary_performance_joechar2[nr_bins_x1]['auc'].append(auc_joechar2)
                dictionary_performance_joechar2[nr_bins_x1]['f1'].append(f1_joechar2)
                dictionary_performance_joechar2[nr_bins_x1]['logloss'].append(logloss_joechar2)
                dictionary_performance_joechar2[nr_bins_x1]['brier'].append(brier_joechar2)


                dictionary_results_joechar2[nr_bins_x1]['est_b1'].append(model2.get_weights()[-2][0][0] if len(model2.get_weights()) >= 2 else np.nan)
                dictionary_results_joechar2[nr_bins_x1]['est_b2'].append(model2.get_weights()[-2][1][0] if len(model2.get_weights()) >= 2 else np.nan)
                dictionary_results_joechar2[nr_bins_x1]['est_b3'].append(model2.get_weights()[-2][2][0] if len(model2.get_weights()) >= 2 else np.nan)

            for cat in categorical_variables:
                select_dataset = list_datasets[cat]
                select_dataset[name] = 0
                for idx in range(select_dataset.shape[0]):
                    mean = select_dataset['mean'].iloc[idx]
                    # std = select_dataset['std'].iloc[idx]
                    om = select_dataset['o_m'].iloc[idx]
                    # os = select_dataset['o_s'].iloc[idx]

                    tensor_to_use = tf.constant([[mean, om]])

                    select_dataset.loc[idx, name] = small_models[cat](tensor_to_use).numpy().flatten()[0]

                list_datasets[cat] = select_dataset

            X_train_encoded_usingM1 = X_train.copy()
            X_test_encoded_usingM1 = X_test.copy()
            for index_cat in range(len(categorical_variables)):
                X_train_encoded_usingM1[categorical_variables[index_cat]] = X_train_encoded_usingM1[categorical_variables[index_cat]].replace(list_datasets[categorical_variables[index_cat]]['cat'].values, list_datasets[categorical_variables[index_cat]][name].values)
                X_test_encoded_usingM1[categorical_variables[index_cat]] = X_test_encoded_usingM1[categorical_variables[index_cat]].replace(list_datasets[categorical_variables[index_cat]]['cat'].values, list_datasets[categorical_variables[index_cat]][name].values)

            if name == 'JoeChar3Sig':
                dictionary_performance_joechar1[nr_bins_x1]['correlation'].append(np.corrcoef(X_train_encoded_usingM1[categorical_variables[0]].tolist(), df_train_cont[categorical_variables[0]].tolist())[0,1])
            else:
                dictionary_performance_joechar2[nr_bins_x1]['correlation'].append(np.corrcoef(X_train_encoded_usingM1[categorical_variables[0]].tolist(), df_train_cont[categorical_variables[0]].tolist())[0,1])

        ###### one hot encoding
        encoded_onehot_categories = {}
        train_objs_num = X_train.shape[0]
        test_objs_num = X_test.shape[0]
        keys = []
        X_train_mod_one_hot = X_train[binary_variables + continuous_variables].copy()
        X_test_mod_one_hot = X_test[binary_variables + continuous_variables].copy()

        how_many_cat_percolumn = []
        how_many_cat_percolumn_everything = []


        for cat in categorical_variables:

            dataset = pd.concat(objs=[X_train[cat], X_test[cat], list_datasets[cat]['cat']], axis=0)

            dataset_preprocessed = pd.get_dummies(dataset, prefix=cat) # Add prefix for clarity

            train_preprocessed = dataset_preprocessed.iloc[:train_objs_num]
            test_preprocessed = dataset_preprocessed.iloc[train_objs_num:train_objs_num+test_objs_num]
            encoded_onehot_categories[cat] = dataset_preprocessed.iloc[train_objs_num+test_objs_num:]

            how_many_categories = train_preprocessed.shape[1]
            keys.extend(train_preprocessed.columns.tolist()) # Use actual column names

            X_train_mod_one_hot = pd.concat([X_train_mod_one_hot, train_preprocessed], axis = 1)
            X_test_mod_one_hot = pd.concat([X_test_mod_one_hot, test_preprocessed], axis = 1)

            how_many_cat_percolumn.append(how_many_categories)
            how_many_cat_percolumn_everything.append(how_many_categories + len(list_columns) - len(continuous_variables)) # Adjusted for other features

        X_train_mod_one_hot = X_train_mod_one_hot[keys + binary_variables + continuous_variables]
        X_test_mod_one_hot = X_test_mod_one_hot[keys + binary_variables + continuous_variables]

        result = list(set(list_columns) - set(continuous_variables))

        X_train_oheandchar = X_train_mod_one_hot.copy()
        X_train_oheandchar[result] = X_train_mod_1[result]
        X_train_oheandchar = X_train_oheandchar[keys + result + binary_variables + continuous_variables]

        X_test_oheandchar = X_test_mod_one_hot.copy()
        X_test_oheandchar[result] = X_test_mod_1[result]
        X_test_oheandchar = X_test_oheandchar[keys + result + binary_variables + continuous_variables]


        X_train_mod_one_hot = X_train_mod_one_hot.values
        X_test_mod_one_hot = X_test_mod_one_hot.values

        train_ds_onehot = tf.data.Dataset.from_tensor_slices((X_train_mod_one_hot.astype(np.float32), y_train))
        test_ds_onehot = tf.data.Dataset.from_tensor_slices((X_test_mod_one_hot.astype(np.float32), y_test))

        f = partial(split_inputs_onehot, categorical_variables=categorical_variables, how_many_cat_percolumn = how_many_cat_percolumn)
        train_ds_onehot = train_ds_onehot.map(f)
        test_ds_onehot = test_ds_onehot.map(f)
        train_ds_onehot = train_ds_onehot.batch(32)
        test_ds_onehot = test_ds_onehot.batch(32)


        #### bigger model joe ohe

        for combination in [[[3,1],'sigmoid'], [[1],'linear']]:
            hidden_layers = combination[0]
            encod_act = combination[1]
            small_models_onehot = {}
            inputs_onehot = {}
            for i in range(len(categorical_variables)):
                small_models_onehot[categorical_variables[i]] = SmallNetwork(hidden_layers, encod_act)
                inputs_onehot[categorical_variables[i]] = Input(shape=(how_many_cat_percolumn[i],), name = str(categorical_variables[i]))
            inputs_onehot['rest'] = Input(shape=(len(binary_variables)+len(continuous_variables),), name = 'rest')
            small_models_outputs_onehot = {k: small_models_onehot[k](inputs_onehot[k]) for k in categorical_variables}
            h = MyLayer()(small_models_outputs_onehot, inputs_onehot)

            initializer = tf.keras.initializers.GlorotUniform(seed=50)

            # Change activation to 'sigmoid' for binary classification output
            outputs_onehot = Dense(1, activation='sigmoid', kernel_initializer = initializer)(h)

            opt = 'adam'

            if combination == [ [3,1],'sigmoid']:
                name = 'JoeOhe3Sig'
                model_onehot1 = Model(inputs=inputs_onehot, outputs=outputs_onehot)
                # Change loss and metrics for binary classification
                model_onehot1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
                history_joeohe1 = model_onehot1.fit(train_ds_onehot, epochs = total_epochs, validation_data= test_ds_onehot,validation_freq=10,verbose = 0)

                y_pred_proba_keras = model_onehot1.predict(test_ds_onehot).flatten()
                y_pred_keras = (y_pred_proba_keras > 0.5).astype(int) # Binarize predictions

                # Update performance metrics for binary classification
                dictionary_performance_joeohe1[nr_bins_x1]['accuracy'].append(accuracy_score(y_test.values, y_pred_keras))
                dictionary_performance_joeohe1[nr_bins_x1]['auc'].append(roc_auc_score(y_test.values, y_pred_proba_keras))
                dictionary_performance_joeohe1[nr_bins_x1]['f1'].append(f1_score(y_test.values, y_pred_keras))
                dictionary_performance_joeohe1[nr_bins_x1]['logloss'].append(log_loss(y_test.values, y_pred_proba_keras))
                dictionary_performance_joeohe1[nr_bins_x1]['brier'].append(brier_score_loss(y_test.values, y_pred_proba_keras))



                if len(model_onehot1.get_weights()) > 2 and model_onehot1.get_weights()[-2].shape[0] >= 3:
                    dictionary_results_joeohe1[nr_bins_x1]['est_b1'].append(model_onehot1.get_weights()[-2][0][0])
                    dictionary_results_joeohe1[nr_bins_x1]['est_b2'].append(model_onehot1.get_weights()[-2][1][0])
                    dictionary_results_joeohe1[nr_bins_x1]['est_b3'].append(model_onehot1.get_weights()[-2][2][0])
                else:
                    dictionary_results_joeohe1[nr_bins_x1]['est_b1'].append(np.nan)
                    dictionary_results_joeohe1[nr_bins_x1]['est_b2'].append(np.nan)
                    dictionary_results_joeohe1[nr_bins_x1]['est_b3'].append(np.nan)


                for cat in categorical_variables:
                  select_dataset = encoded_onehot_categories[cat].copy()
                  select_dataset[name] = 0
                  for i in range(select_dataset.shape[0]):
                    see = list(select_dataset.iloc[i,:-1].values)
                    select_dataset[name].iloc[i] = small_models_onehot[cat](tf.constant([see])).numpy()
                  list_datasets[cat][name] = select_dataset[name]

                X_train_encoded_usingM6 = X_train.copy()
                X_test_encoded_usingM6 = X_test.copy()

                for index_cat in range(len(categorical_variables)):
                  X_train_encoded_usingM6[categorical_variables[index_cat]].replace(list_datasets[categorical_variables[index_cat]]['cat'].tolist(), list_datasets[categorical_variables[index_cat]][name].tolist(), inplace=True)
                  X_test_encoded_usingM6[categorical_variables[index_cat]].replace(list_datasets[categorical_variables[index_cat]]['cat'].tolist(), list_datasets[categorical_variables[index_cat]][name].tolist(), inplace=True)

                dictionary_performance_joeohe1[nr_bins_x1]['correlation'].append(np.corrcoef(X_train_encoded_usingM6[categorical_variables[0]].tolist(), df_train_cont[categorical_variables[0]].tolist())[0,1])





            else:
                model_onehot2 = Model(inputs=inputs_onehot, outputs=outputs_onehot)
                name = 'JoeOhe1Lin'
                # Change loss and metrics for binary classification
                model_onehot2.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

                history_joeohe2 = model_onehot2.fit(train_ds_onehot, epochs = total_epochs, validation_data= test_ds_onehot,validation_freq=10,verbose = 0)

                #### calculate accuracy, auc and f1 score
                y_pred_proba_keras = model_onehot2.predict(test_ds_onehot).flatten()
                y_pred_keras = (y_pred_proba_keras > 0.5).astype(int) # Binarize predictions

                # Update performance metrics for binary classification
                dictionary_performance_joeohe2[nr_bins_x1]['accuracy'].append(accuracy_score(y_test.values, y_pred_keras))
                dictionary_performance_joeohe2[nr_bins_x1]['auc'].append(roc_auc_score(y_test.values, y_pred_proba_keras))
                dictionary_performance_joeohe2[nr_bins_x1]['f1'].append(f1_score(y_test.values, y_pred_keras))
                dictionary_performance_joeohe2[nr_bins_x1]['logloss'].append(log_loss(y_test.values, y_pred_proba_keras))
                dictionary_performance_joeohe2[nr_bins_x1]['brier'].append(brier_score_loss(y_test.values, y_pred_proba_keras))

                # Note: Similar to above, coefficients are not directly interpretable as 'betas'.
                if len(model_onehot2.get_weights()) > 2 and model_onehot2.get_weights()[-2].shape[0] >= 3:
                    dictionary_results_joeohe2[nr_bins_x1]['est_b1'].append(model_onehot2.get_weights()[-2][0][0])
                    dictionary_results_joeohe2[nr_bins_x1]['est_b2'].append(model_onehot2.get_weights()[-2][1][0])
                    dictionary_results_joeohe2[nr_bins_x1]['est_b3'].append(model_onehot2.get_weights()[-2][2][0])
                else:
                    dictionary_results_joeohe2[nr_bins_x1]['est_b1'].append(np.nan)
                    dictionary_results_joeohe2[nr_bins_x1]['est_b2'].append(np.nan)
                    dictionary_results_joeohe2[nr_bins_x1]['est_b3'].append(np.nan)

                for cat in categorical_variables:
                  select_dataset = encoded_onehot_categories[cat].copy()
                  select_dataset[name] = 0
                  for i in range(select_dataset.shape[0]):
                    see = list(select_dataset.iloc[i,:-1].values)
                    select_dataset[name].iloc[i] = small_models_onehot[cat](tf.constant([see])).numpy()
                  list_datasets[cat][name] = select_dataset[name]

                X_train_encoded_usingM6 = X_train.copy()
                X_test_encoded_usingM6 = X_test.copy()

                for index_cat in range(len(categorical_variables)):
                  X_train_encoded_usingM6[categorical_variables[index_cat]].replace(list_datasets[categorical_variables[index_cat]]['cat'].tolist(), list_datasets[categorical_variables[index_cat]][name].tolist(), inplace=True)
                  X_test_encoded_usingM6[categorical_variables[index_cat]].replace(list_datasets[categorical_variables[index_cat]]['cat'].tolist(), list_datasets[categorical_variables[index_cat]][name].tolist(), inplace=True)

                dictionary_performance_joeohe2[nr_bins_x1]['correlation'].append(np.corrcoef(X_train_encoded_usingM6[categorical_variables[0]].tolist(), df_train_cont[categorical_variables[0]].tolist())[0,1])



import matplotlib.pyplot as plt
from matplotlib.patches import Patch

method_colors = {
    'Reg Target': 'blue',
    'Simple Target': 'orange',
    'GLMM': 'green',
    'Ordinal': 'red',
    'JoeChar Sigmoid': 'purple',
    'JoeChar Linear': 'brown',
    'One-Hot': 'cyan',
    'JoeOhe Sigmoid': 'magenta',
    'JoeOhe Linear': 'olive',
    'Continuous': 'gray',
    'No Cat': 'pink',
}



performance_data = {}
for nr_bins in list_nr_bins:
    performance_data[nr_bins] = {
        'Reg Target': dictionary_performance_tarreg[nr_bins],
        'Simple Target': dictionary_performance[nr_bins],
        'GLMM': dictionary_performance_glmm[nr_bins],
        'Ordinal': dictionary_performance_ordinal[nr_bins],
        'JoeChar Sigmoid': dictionary_performance_joechar1[nr_bins],
        'JoeChar Linear': dictionary_performance_joechar2[nr_bins],
        'One-Hot': dictionary_performance_onehot[nr_bins],
        'JoeOhe Sigmoid': dictionary_performance_joeohe1[nr_bins],
        'JoeOhe Linear': dictionary_performance_joeohe2[nr_bins]
    }


for performance_metric in ['accuracy', 'auc', 'f1', 'logloss', 'brier', 'correlation']:
    plt.figure(figsize=(20, 8))

    data_to_plot = []
    labels = []
    colors = []
    bin_boundaries = []
    bin_centers = []

    count = 0
    methods_list = ['Reg Target', 'Simple Target', 'GLMM', 'Ordinal', 'JoeChar Sigmoid', 'JoeChar Linear', 'One-Hot', 'JoeOhe Sigmoid', 'JoeOhe Linear']

    for nr_bins in list_nr_bins:
        for method in methods_list:
            data_to_plot.append(performance_data[nr_bins][method][performance_metric])
            labels.append(f'Bins: {nr_bins}\n{method}')
            colors.append(method_colors[method])

        count += len(methods_list)
        bin_boundaries.append(count)
        bin_centers.append(count - len(methods_list) / 2)

    if performance_metric in dictionary_performance_continuous: 
        data_to_plot.append(dictionary_performance_continuous[performance_metric])
        labels.append('Continuous')
        colors.append('slategray')  
        bin_boundaries.append(count)
        bin_centers.append(count + 0.5)
  
    if performance_metric in dictionary_performance_nocat:
        data_to_plot.append(dictionary_performance_nocat[performance_metric])
        labels.append('NO CAT')
        colors.append('darkcyan') 
        bin_boundaries.append(count + (1 if performance_metric in dictionary_performance_continuous else 0))
        bin_centers.append(count + (1 if performance_metric in dictionary_performance_continuous else 0) + 0.5)
  


    box = plt.boxplot(data_to_plot, labels=labels, vert=True, patch_artist=True)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    for boundary in bin_boundaries[:-1]:
        plt.axvline(boundary + 0.5, color='gray', linestyle=':', linewidth=1)

    y_max = max([max(group) if len(group) > 0 else 0 for group in data_to_plot])

    for center, nr_bins in zip(bin_centers[:-2], list_nr_bins):
        plt.text(center + 0.5, y_max * 0.95, f'{nr_bins} cat', ha='center', va='bottom', fontsize=10, weight='bold') # Adjusted y offset for logloss

    plt.text(bin_centers[-1], y_max * 0.95, 'Extra', ha='center', va='bottom', fontsize=10, weight='bold') # Adjusted y offset

    legend_patches = [Patch(facecolor=color, label=method) for method, color in method_colors.items()]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('Combination of Number of Bins and Method')
    plt.ylabel(performance_metric.upper())
    plt.title(f'Distribution of {performance_metric.upper()} Across Methods for Different Binning Strategies')
    plt.xticks([])
    plt.tight_layout()
    plt.grid(True, axis='y')
    
    ### change where to save
    plt.savefig(which_open+'simulated_binary_classification_interaction_'+performance_metric+'.png')
    plt.show()

results_to_save = {
    'dictionary_results': dictionary_results,
    'dictionary_results_tarreg': dictionary_results_tarreg,
    'dictionary_results_joechar1': dictionary_results_joechar1,
    'dictionary_results_joechar2': dictionary_results_joechar2,
    'dictionary_results_joeohe1': dictionary_results_joeohe1,
    'dictionary_results_joeohe2': dictionary_results_joeohe2,
    'dictionary_results_woe': dictionary_results_woe,
    'dictionary_results_glmm': dictionary_results_glmm,
    'dictionary_results_onehot': dictionary_results_onehot,
    'dictionary_results_ordinal': dictionary_results_ordinal,
    'dictionary_results_continuous': dictionary_results_continuous,
    'dictionary_results_nocat': dictionary_results_nocat,
    'dictionary_nr_obs_cat': dictionary_nr_obs_cat,
    'dictionary_relative_entropy': dictionary_relative_entropy,
    'dictionary_class_proportion':dictionary_class_proportion,
    'dictionary_performance': dictionary_performance,
    'dictionary_performance_tarreg': dictionary_performance_tarreg,
    'dictionary_performance_joechar1': dictionary_performance_joechar1,
    'dictionary_performance_joechar2': dictionary_performance_joechar2,
    'dictionary_performance_joeohe1': dictionary_performance_joeohe1,
    'dictionary_performance_joeohe2': dictionary_performance_joeohe2,
    'dictionary_performance_glmm': dictionary_performance_glmm,
    'dictionary_performance_onehot': dictionary_performance_onehot,
    'dictionary_performance_woe': dictionary_performance_woe,
    'dictionary_performance_ordinal': dictionary_performance_ordinal,
    'dictionary_performance_continuous': dictionary_performance_continuous,
    'dictionary_performance_nocat': dictionary_performance_nocat,
    'list_nr_bins': list_nr_bins
}


#### saving results (for later plotting)
with open(which_open+'results_binary_classification_synthetic_int.pkl', 'wb') as f:
    pickle.dump(results_to_save, f)