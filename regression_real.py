import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import pickle
import openml
from openml.datasets import get_dataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense
from functools import partial
import matplotlib.pyplot as plt
from category_encoders.glmm import GLMMEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.ordinal import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from scipy.stats import spearmanr, kendalltau

print('Real_regression')

def calculate_entropy(series):
  """Calculates normalised entropy for a pandas Series (categorical feature)."""
  value_counts = series.value_counts(normalize=True)
  entropy = -np.sum(value_counts * np.log2(value_counts))
  cardinality = len(value_counts)
  if cardinality <= 1:
    return 0.0
  return entropy / np.log2(cardinality)


def split_inputs(x, y, categorical_variables):
  how_many_cat_variables = len(categorical_variables)
  dictionary = {}
  for index in range(how_many_cat_variables):
    dictionary[categorical_variables[index]] = [x[index], x[how_many_cat_variables+index], x[2*how_many_cat_variables+index], x[3*how_many_cat_variables+index]]
  dictionary['rest'] = x[4*how_many_cat_variables:]
  return (dictionary, y)

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
            self.dense_layers.append(Dense(units, activation=self.activation, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=50)))
            # self.dense_layers.append(Dropout(0.2))

    def call(self, inputs):
        h = inputs
        for dense_layer in self.dense_layers:
            h = dense_layer(h)
        return h

def compare_encoding_ranks(
    original_categories: pd.Series,
    encoded_values_1: pd.Series,
    encoded_values_2: pd.Series,
    encoder_name_1: str = "Encoder A",
    encoder_name_2: str = "Encoder B"
):
    """
    Compares the ranks of two different numerical encodings for the same
    categorical feature using Spearman's Rank Correlation and Kendall's Tau.
    """
    # Create a DataFrame for easy handling
    df = pd.DataFrame({
        'original_category': original_categories,
        f'{encoder_name_1}_values': encoded_values_1,
        f'{encoder_name_2}_values': encoded_values_2
    })

    # Group by original category and take the mean of the encoded values
    df_agg = df.groupby('original_category').agg(
        mean_val_1=(f'{encoder_name_1}_values', 'mean'),
        mean_val_2=(f'{encoder_name_2}_values', 'mean')
    ).reset_index()

    # Sort by one of the encoded values to visualize rank consistency
    df_agg_sorted = df_agg.sort_values(by='mean_val_1').reset_index(drop=True)

    print(f"--- Rank Comparison of {encoder_name_1} vs. {encoder_name_2} ---")
    print("\nAggregated Encoded Values (first 10 unique categories, sorted by first encoder):")
    print(df_agg_sorted.head(10).to_string())

    # Calculate Spearman's Rank Correlation
    spearman_corr, spearman_pvalue = spearmanr(
        df_agg['mean_val_1'], df_agg['mean_val_2']
    )
    print(f"\nSpearman's Rank Correlation: {spearman_corr:.4f} (p-value: {spearman_pvalue:.4f})")

    # Calculate Kendall's Tau
    kendall_tau, kendall_pvalue = kendalltau(
        df_agg['mean_val_1'], df_agg['mean_val_2']
    )
    print(f"Kendall's Tau: {kendall_tau:.4f} (p-value: {kendall_pvalue:.4f})")



    results = {
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_pvalue,
        'kendall_tau': kendall_tau,
        'kendall_p_value': kendall_pvalue
    }
    return results

def characteristics_about_dataset(X, categorical_indicator, want_plots = False, X_total = None):

  ### look at cardinalities of categorical features
  categorical_cardinalities = {}
  categorical_entropies = {}
  for i, is_categorical in enumerate(categorical_indicator):
    if is_categorical:
      col_name = attribute_names[i] if attribute_names else f'Column_{i}'
      cardinality = X.iloc[:, i].nunique()
      categorical_cardinalities[col_name] = cardinality
      categorical_entropies[col_name] = calculate_entropy(X.iloc[:, i])
  summary_df = pd.DataFrame(list(categorical_cardinalities.items()), columns=['Feature', 'Cardinality'])
  summary_df['Entropy'] = summary_df['Feature'].map(categorical_entropies)



  if want_plots:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Feature', y='Cardinality', data=summary_df.sort_values(by='Cardinality', ascending=False))
    plt.title('Cardinality of Categorical Features')
    plt.xlabel('Feature Name')
    plt.ylabel('Cardinality')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    ### Visualise entropies
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Feature', y='Entropy', data=summary_df.sort_values(by='Entropy', ascending=False))
    plt.title('Entropy of Categorical Features')
    plt.xlabel('Feature Name')
    plt.ylabel('Entropy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

  if X_total is not None:
    ### look at orioortion of total categories from X_total that are seen in X_train
    list_proportion = []
    for i, is_categorical in enumerate(categorical_indicator):
      if is_categorical:
        col_name = attribute_names[i] if attribute_names else f'Column_{i}'
        cardinality = X_total.iloc[:, i].nunique()
        cardinality_train = X.iloc[:, i].nunique()
        proportion = cardinality_train / cardinality
        list_proportion.append(proportion)
    summary_df['Proportion'] = list_proportion
  return summary_df








def run_iteration(i, df_train_all, X_test, y_test, categorical_indicator, categorical_features, continuous_features, target_variable, total_epochs, df, which_methods):
    df_train = df_train_all.sample(n=1000) # Use a different random state for each iteration
        
    y_train = df_train[target_variable]
    X_train = df_train.drop(target_variable, axis=1)

    summary_df = characteristics_about_dataset(df_train, categorical_indicator, want_plots=False, X_total = df)


    if 'model_nocat' in which_methods:
      model_nocat = LinearRegression() # Changed to LinearRegression
      model_nocat.fit(X_train.drop(categorical_features, axis=1), y_train)
      y_pred_nocat = model_nocat.predict(X_test.drop(categorical_features, axis=1))
      print('model_nocat is done')


    if 'model_tarreg' in which_methods:
      target_encoder = TargetEncoder(smooth='auto', cv=5, target_type = 'continuous') 
      target_encoder.fit(X_train[categorical_features], y_train)
      X_train_tarreg = X_train.copy()
      X_test_tarreg = X_test.copy()
      X_train_tarreg[categorical_features] = target_encoder.transform(X_train[categorical_features])
      X_test_tarreg[categorical_features] = target_encoder.transform(X_test[categorical_features])
      model_tarreg = LinearRegression()  # Changed to LinearRegression
      model_tarreg.fit(X_train_tarreg, y_train)
      y_pred_tarreg = model_tarreg.predict(X_test_tarreg)
      print('model_tarreg is done')




    ### GLMM
    if 'GLMM' in which_methods:
      encoder_glmm = GLMMEncoder(cols=categorical_features, drop_invariant=False, verbose=0, binomial_target = False) # Added drop_invariant and verbose for cleaner output
      encoder_glmm.fit(X_train[categorical_features], y_train)
      X_train_encoded_glmm = X_train.copy()
      X_test_encoded_glmm = X_test.copy()
      X_train_encoded_glmm[categorical_features] = encoder_glmm.transform(X_train[categorical_features])
      X_test_encoded_glmm[categorical_features] = encoder_glmm.transform(X_test[categorical_features])
      model_glmm = LinearRegression() 
      model_glmm.fit(X_train_encoded_glmm, y_train)
      y_pred_glmm = model_glmm.predict(X_test_encoded_glmm)
      print('GLMM is done')
      
    if 'ORD' in which_methods:
        ### ordinal encoding
        encoder_oe = OrdinalEncoder(cols=categorical_features)
        encoder_oe.fit(X_train[categorical_features], y_train)
        X_train_encoded_ord = X_train.copy()
        X_test_encoded_ord = X_test.copy()
        X_train_encoded_ord[categorical_features] = encoder_oe.transform(X_train[categorical_features])
        X_test_encoded_ord[categorical_features] = encoder_oe.transform(X_test[categorical_features])
     
    
        model_oe = LinearRegression()
        model_oe.fit(X_train_encoded_ord, y_train)
        y_pred_ord = model_oe.predict(X_test_encoded_ord)
        print('ORD is done')



  ### One hot encoding
    if 'OneHot' in which_methods:
        encoder_onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # Fit the encoder only on the training categorical features
        encoder_onehot.fit(X_train[categorical_features])

        # Transform both training and test categorical features
        encoded_train_x1 = encoder_onehot.transform(X_train[categorical_features])
        encoded_test_x1 = encoder_onehot.transform(X_test[categorical_features])

        df_train_encoded_x1 = pd.DataFrame(encoded_train_x1,
                                           columns=encoder_onehot.get_feature_names_out(categorical_features),
                                           index=X_train.index) # Use X_train.index as it's the features part of df_train

        df_test_encoded_x1 = pd.DataFrame(encoded_test_x1,
                                          columns=encoder_onehot.get_feature_names_out(categorical_features),
                                          index=X_test.index)

      
        X_train_onehot = pd.concat([X_train.drop(columns=categorical_features, axis=1), df_train_encoded_x1], axis=1)

        X_test_onehot = pd.concat([X_test.drop(columns=categorical_features, axis=1), df_test_encoded_x1], axis=1)

       

        model_onehot = LinearRegression() 
        model_onehot.fit(X_train_onehot, y_train)
        y_pred_onehot = model_onehot.predict(X_test_onehot)
        print('OneHot is done')






    if 'JoeChar' in which_methods:

      list_columns = []
      list_datasets_iter = {}
      dictionary_all_categorical_columns_positives = {}
      dictionary_all_categorical_columns_negatives = {}
      dictionary_all_categorical_columns_overallmean = {}
      dictionary_all_categorical_columns_overallstd = {}


      for which_column_to_categories in categorical_features:
          categories = df[which_column_to_categories].unique().tolist()
          dict_whichcolumn_pos = {}
          dict_whichcolumn_neg = {}
          dict_whichcolumn_overallmean = {}
          dict_whichcolumn_overallstd = {}
          overall_mean = np.mean(df_train[target_variable])
          overall_std = np.std(df_train[target_variable])
          for cat in categories:
              which_cat = df_train[df_train[which_column_to_categories] == cat]
              if which_cat.shape[0] >= 1:
                  dict_whichcolumn_pos[cat] = np.mean(which_cat[target_variable])
                  dict_whichcolumn_neg[cat] = np.std(which_cat[target_variable])
                  dict_whichcolumn_overallmean[cat] = overall_mean
                  dict_whichcolumn_overallstd[cat] = overall_std
              else:
                  dict_whichcolumn_pos[cat] = 0
                  dict_whichcolumn_neg[cat] = 0
                  dict_whichcolumn_overallmean[cat] = overall_mean
                  dict_whichcolumn_overallstd[cat] = overall_std

          dictionary_all_categorical_columns_positives[which_column_to_categories] = dict_whichcolumn_pos
          dictionary_all_categorical_columns_negatives[which_column_to_categories] = dict_whichcolumn_neg
          dictionary_all_categorical_columns_overallmean[which_column_to_categories] = dict_whichcolumn_overallmean
          dictionary_all_categorical_columns_overallstd[which_column_to_categories] = dict_whichcolumn_overallstd
          list_columns.append(which_column_to_categories+str('_P'))

      X_train, y_train =  df_train.drop(target_variable, axis=1), df_train[target_variable]
      X_test_copy = X_test.copy() # Use a copy for modification
      X_train_mod_1 = X_train.copy()
      X_test_mod_1 = X_test_copy.copy()


      for which_column_to_categories in categorical_features:
          X_train_mod_1[which_column_to_categories+str('_N')] = X_train_mod_1[which_column_to_categories].copy()
          X_train_mod_1[which_column_to_categories+str('_OM')] = X_train_mod_1[which_column_to_categories].copy()
          X_train_mod_1[which_column_to_categories+str('_OS')] = X_train_mod_1[which_column_to_categories].copy()
          X_train_mod_1.rename(columns={which_column_to_categories: which_column_to_categories+str('_P')}, inplace=True)

          X_test_mod_1[which_column_to_categories+str('_N')] = X_test_mod_1[which_column_to_categories].copy()
          X_test_mod_1[which_column_to_categories+str('_OM')] = X_test_mod_1[which_column_to_categories].copy()
          X_test_mod_1[which_column_to_categories+str('_OS')] = X_test_mod_1[which_column_to_categories].copy()
          X_test_mod_1.rename(columns={which_column_to_categories: which_column_to_categories+str('_P')}, inplace=True)

          dic_pos = dictionary_all_categorical_columns_positives[which_column_to_categories]
          dic_neg = dictionary_all_categorical_columns_negatives[which_column_to_categories]
          dic_overallmean = dictionary_all_categorical_columns_overallmean[which_column_to_categories]
          dic_overallstd = dictionary_all_categorical_columns_overallstd[which_column_to_categories]


          test_this_column = pd.DataFrame(columns=['cat','mean','std','o_m','o_s'])


          for cat in dic_pos.keys():
              n = dic_neg[cat]
              p = dic_pos[cat]
              o_mean = dic_overallmean[cat]
              o_std = dic_overallstd[cat]
              X_train_mod_1[which_column_to_categories+str('_N')] =   X_train_mod_1[which_column_to_categories+str('_N')].replace(cat, n)
              X_train_mod_1[which_column_to_categories+str('_P')] =   X_train_mod_1[which_column_to_categories+str('_P')].replace(cat, p)
              X_train_mod_1[which_column_to_categories+str('_OM')] =   X_train_mod_1[which_column_to_categories+str('_OM')].replace(cat, o_mean)
              X_train_mod_1[which_column_to_categories+str('_OS')] =   X_train_mod_1[which_column_to_categories+str('_OS')].replace(cat, o_std)

              X_test_mod_1[which_column_to_categories+str('_N')] = X_test_mod_1[which_column_to_categories+str('_N')].replace(cat, n)
              X_test_mod_1[which_column_to_categories+str('_P')] = X_test_mod_1[which_column_to_categories+str('_P')].replace(cat, p)
              X_test_mod_1[which_column_to_categories+str('_OM')] = X_test_mod_1[which_column_to_categories+str('_OM')].replace(cat, o_mean)
              X_test_mod_1[which_column_to_categories+str('_OS')] = X_test_mod_1[which_column_to_categories+str('_OS')].replace(cat, o_std)

              test_this_column = pd.concat([test_this_column, pd.DataFrame([{'cat':cat,'mean': p, 'std': n, 'o_m': o_mean, 'o_s':o_std}])], ignore_index=True)


          list_datasets_iter[which_column_to_categories] = test_this_column
          list_columns.append(which_column_to_categories+str('_N'))

      for which_column_to_categories in categorical_features:
          list_columns.append(which_column_to_categories+str('_OM'))
      for which_column_to_categories in categorical_features:
          list_columns.append(which_column_to_categories+str('_OS'))

      list_columns = list_columns+continuous_features

      X_train_mod_1 = X_train_mod_1[list_columns]
      X_test_mod_1 = X_test_mod_1[list_columns]


      train_ds = tf.data.Dataset.from_tensor_slices((X_train_mod_1.values.astype(np.float32), y_train.values.astype(np.float32)))
      test_ds = tf.data.Dataset.from_tensor_slices((X_test_mod_1.values.astype(np.float32),  y_test.values.astype(np.float32)))
      f = partial(split_inputs, categorical_variables=categorical_features)
      train_ds = train_ds.map(f)
      test_ds = test_ds.map(f)

      train_ds = train_ds.shuffle(500).batch(32)
      test_ds = test_ds.batch(32)


      # Define early stopping callback
      early_stopping = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',  # Monitor validation loss
          patience=10,         # Number of epochs with no improvement after which training will be stopped.
          restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
      )



      for combination in [ [[3,1],'sigmoid'], [[1],'linear']]:
          hidden_layers = combination[0]
          encod_act = combination[1]

          small_models = {}
          inputs = {}
          for cat_feat in categorical_features:
              small_models[cat_feat] = SmallNetwork(hidden_layers, encod_act)
              inputs[cat_feat] = Input(shape=(4,), name = str(cat_feat))

          inputs['rest'] = Input(shape=(len(continuous_features),), name = 'rest')
          small_models_outputs = {k: small_models[k](inputs[k]) for k in categorical_features}
          h =  MyLayer()(small_models_outputs, inputs)

          initializer = tf.keras.initializers.GlorotUniform(seed=50)

          opt = 'adam'

          outputs = Dense(1, activation='linear', kernel_initializer = initializer)(h) # Changed activation to linear

          if combination == [ [3,1],'sigmoid']:
              name = 'JoeChar3Sig'
              model1 = Model(inputs=inputs, outputs=outputs)
              model1.compile(loss='mse', optimizer=opt) # Changed loss and metrics
              history1 = model1.fit(train_ds, epochs = total_epochs, validation_data= test_ds,validation_freq=1, callbacks=[early_stopping], verbose = 0) # Add early stopping here
              y_pred_keras1 = model1.predict(test_ds).flatten() # Predict continuous values
              # y_pred_keras1 = (y_prob_keras1 > 0.5).astype(int) # Removed classification prediction


          else:
              model2 = Model(inputs=inputs, outputs=outputs)
              name = 'JoeChar1Lin'

              model2.compile(loss='mse', optimizer=opt) # Changed loss and metrics
              history2 = model2.fit(train_ds, epochs = total_epochs, validation_data= test_ds,validation_freq=1, callbacks=[early_stopping], verbose = 0) # Add early stopping here
              y_pred_keras2 = model2.predict(test_ds).flatten() # Predict continuous values
              # y_pred_keras2 = (y_prob_keras2 > 0.5).astype(int) # Removed classification prediction


          for cat in categorical_features:
              select_dataset = list_datasets_iter[cat]



              select_dataset[name] = 0
              for row_index in range(select_dataset.shape[0]):
                  mean = select_dataset['mean'].iloc[row_index]
                  std = select_dataset['std'].iloc[row_index]
                  om = select_dataset['o_m'].iloc[row_index]
                  os = select_dataset['o_s'].iloc[row_index]

                  tensor_to_use = tf.constant([[mean,std, om, os]])

                  select_dataset.loc[row_index, name] = small_models[cat](tensor_to_use).numpy()


              list_datasets_iter[cat] = select_dataset

          X_train_encoded_usingM1 = X_train.copy()
          X_test_encoded_usingM1 = X_test_copy.copy()
          for index_cat in range(len(categorical_features)):
              replacement_values = list_datasets_iter[categorical_features[index_cat]][name].values
              X_train_encoded_usingM1[categorical_features[index_cat]] = X_train_encoded_usingM1[categorical_features[index_cat]].replace(list_datasets_iter[categorical_features[index_cat]]['cat'].values, replacement_values)
              X_test_encoded_usingM1[categorical_features[index_cat]] = X_test_encoded_usingM1[categorical_features[index_cat]].replace(list_datasets_iter[categorical_features[index_cat]]['cat'].values, replacement_values)
          # Store the encoded X_test for this JoeChar model
          if name == 'JoeChar3Sig':
              X_test_encoded_joechar3sig = X_test_encoded_usingM1
          elif name == 'JoeChar1Lin':
              X_test_encoded_joechar1lin = X_test_encoded_usingM1

      print('JoeChar is done')


    ###### one hot encoding

    if 'JoeOhe' in which_methods:

      encoded_onehot_categories = {}
      train_objs_num = X_train.shape[0]
      test_objs_num = X_test.shape[0]
      keys = []
      X_train_mod_one_hot = X_train[continuous_features].copy()
      X_test_mod_one_hot = X_test[continuous_features].copy()



      how_many_cat_percolumn = []
      how_many_cat_percolumn_everything = []
      for cat in categorical_features:
        dataset = pd.concat(objs=[X_train[cat], X_test[cat],list_datasets_iter[cat]['cat']], axis=0)
        # dataset = pd.concat(objs=[X_train[cat], X_test[cat]], axis=0)
        dataset_preprocessed = pd.get_dummies(dataset)
        train_preprocessed = dataset_preprocessed[:train_objs_num]
        test_preprocessed = dataset_preprocessed[train_objs_num:train_objs_num+test_objs_num]
        encoded_onehot_categories[cat] = dataset_preprocessed[train_objs_num+test_objs_num:]
        how_many_categories = train_preprocessed.shape[1]
        keys += [cat+'_one_hot_'+str(i) for i in range(1,how_many_categories+1)]
        X_train_mod_one_hot = pd.concat([X_train_mod_one_hot, train_preprocessed], axis = 1)
        X_train_mod_one_hot.columns = continuous_features + keys
        X_test_mod_one_hot = pd.concat([X_test_mod_one_hot, test_preprocessed], axis = 1)
        X_test_mod_one_hot.columns = continuous_features + keys
        how_many_cat_percolumn.append(how_many_categories)
        how_many_cat_percolumn_everything.append(how_many_categories+2)



      X_train_mod_one_hot = X_train_mod_one_hot[keys + continuous_features]
      X_test_mod_one_hot = X_test_mod_one_hot[keys +  continuous_features]

      X_train_mod_one_hot= X_train_mod_one_hot.values
      X_test_mod_one_hot = X_test_mod_one_hot.values

      train_ds_onehot = tf.data.Dataset.from_tensor_slices((X_train_mod_one_hot.astype(np.float32), y_train.values))
      test_ds_onehot = tf.data.Dataset.from_tensor_slices((X_test_mod_one_hot.astype(np.float32), y_test.values))

      f = partial(split_inputs_onehot, categorical_variables=categorical_features, how_many_cat_percolumn = how_many_cat_percolumn)
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
        for i in range(len(categorical_features)):
          small_models_onehot[categorical_features[i]] = SmallNetwork(hidden_layers, encod_act)
          inputs_onehot[categorical_features[i]] = Input(shape=(how_many_cat_percolumn[i],), name = str(categorical_features[i]))
        inputs_onehot['rest'] = Input(shape=(len(continuous_features),), name = 'rest')
        small_models_outputs_onehot = {k: small_models_onehot[k](inputs_onehot[k]) for k in categorical_features}
        h = MyLayer()(small_models_outputs_onehot, inputs_onehot)

        initializer = tf.keras.initializers.GlorotUniform(seed=50)

        outputs_onehot = Dense(1, activation='linear', kernel_initializer = initializer)(h) # Changed activation to linear

        opt = 'adam'


        if combination == [ [3,1],'sigmoid']:
          model_onehot1 = Model(inputs=inputs_onehot, outputs=outputs_onehot)
          model_onehot1.compile(loss='mse', optimizer=opt) # Changed loss and metrics
          history_joeohe1 = model_onehot1.fit(train_ds_onehot, epochs = total_epochs, validation_data= test_ds_onehot,validation_freq=1,verbose = 0, callbacks=[early_stopping])

          y_pred_keras_ohe1 = model_onehot1.predict(test_ds_onehot).flatten() # Predict continuous values
          # y_pred_keras_ohe1 = (y_pred_keras_ohe1 > 0.5).astype(int) # Removed classification prediction






        else:
          model_onehot2 = Model(inputs=inputs_onehot, outputs=outputs_onehot)
          model_onehot2.compile(loss='mse', optimizer=opt) # Changed loss and metrics

          history_joeohe2 = model_onehot2.fit(train_ds_onehot, epochs = total_epochs, validation_data= test_ds_onehot,validation_freq=1,verbose = 0, callbacks=[early_stopping])
          y_pred_keras_ohe2 = model_onehot2.predict(test_ds_onehot).flatten() # Predict continuous values
          # y_pred_keras_ohe2 = (y_pred_keras_ohe2 > 0.5).astype(int) # Removed classification prediction

      print('JoeOhe is done')
    
    ###plot history for all models
    plt.plot(history1.history['loss'], label='JoeChar3Sig')
    plt.plot(history2.history['loss'], label='JoeChar1Lin')
    plt.plot(history_joeohe1.history['loss'], label='JoeOhe3Sig')
    plt.plot(history_joeohe2.history['loss'], label='JoeOhe1Lin')
    plt.savefig('loss_plots.png')
    plt.show()


    results_for_this_iter = {}

    if 'model_nocat' in which_methods:
      results_for_this_iter['model_nocat'] = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred_nocat), 'R-squared': r2_score(y_test, y_pred_nocat) # Changed metrics
        }
    if 'model_tarreg' in which_methods:
      results_for_this_iter['model_tarreg'] = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred_tarreg), 'R-squared': r2_score(y_test, y_pred_tarreg) # Changed metrics
        }

    if 'GLMM' in which_methods:
      results_for_this_iter['GLMM'] = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred_glmm), 'R-squared': r2_score(y_test, y_pred_glmm) # Changed metrics
        }
      
    if 'ORD' in which_methods:
      results_for_this_iter['ORD'] = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred_ord), 'R-squared': r2_score(y_test, y_pred_ord) 
        }

    if 'JoeChar' in which_methods:
      results_for_this_iter['JoeChar3Sig'] = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred_keras1), 'R-squared': r2_score(y_test, y_pred_keras1) # Changed metrics
        }
      results_for_this_iter['JoeChar1Lin'] = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred_keras2), 'R-squared': r2_score(y_test, y_pred_keras2) # Changed metrics
        }

    if 'JoeOhe' in which_methods:
      results_for_this_iter['JoeOhe3Sig'] = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred_keras_ohe1), 'R-squared': r2_score(y_test, y_pred_keras_ohe1) # Changed metrics
        }
      results_for_this_iter['JoeOhe1Lin'] = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred_keras_ohe2), 'R-squared': r2_score(y_test, y_pred_keras_ohe2) # Changed metrics
        }
    if 'OneHot' in which_methods:
      results_for_this_iter['OneHot'] = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred_onehot), 'R-squared': r2_score(y_test, y_pred_onehot) # Changed metrics
        }

    results_for_this_iter['summaries'] = summary_df
    return results_for_this_iter




how_many_iterations = 10
total_epochs = 500
results = {}


for dataset_id in [41211, 41445, 41210, 41267]:


  results[dataset_id] = []

  dataset = get_dataset(dataset_id)

  X, y, categorical_indicator, attribute_names = dataset.get_data(
      dataset_format="dataframe", target=dataset.default_target_attribute
  )

  if 'lpep_pickup_datetime' in attribute_names:
      idx = attribute_names.index('lpep_pickup_datetime')
      if idx < len(categorical_indicator): 
          categorical_indicator.pop(idx)
      attribute_names.pop(idx) 
      X.drop('lpep_pickup_datetime', axis=1, inplace=True)
      
    
  if 'datetime' in attribute_names:
    idx = attribute_names.index('datetime')
    if idx < len(categorical_indicator): 
        categorical_indicator.pop(idx)
    attribute_names.pop(idx) 
    X.drop('datetime', axis=1, inplace=True)
    
  if 'FL_DATE' in attribute_names:
    idx = attribute_names.index('FL_DATE')
    if idx < len(categorical_indicator): 
        categorical_indicator.pop(idx)
    attribute_names.pop(idx) 
    X.drop('FL_DATE', axis=1, inplace=True)




  
  if 'lpep_dropoff_datetime' in attribute_names:
      idx = attribute_names.index('lpep_dropoff_datetime')
      if idx < len(categorical_indicator):
          categorical_indicator.pop(idx)
      attribute_names.pop(idx) 
      X.drop('lpep_dropoff_datetime', axis=1, inplace=True)


  ### create df
  df = pd.concat([X, y], axis=1)



  categorical_features = list(df.select_dtypes(include='category').columns)
  continuous_features = list(df.select_dtypes(include='number').columns)


  target_variable = y.name
  continuous_features.remove(target_variable)

  df[target_variable] = (df[target_variable] - df[target_variable].min()) / (df[target_variable].max() - df[target_variable].min())

  df[continuous_features] = df[continuous_features].apply(pd.to_numeric, errors='coerce')
  df[categorical_features] = df[categorical_features].astype(str)

  df = df.dropna(subset=continuous_features) ## removing obs with missing valyes in continuous
  for col in continuous_features:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

  summary_df = characteristics_about_dataset(X, categorical_indicator)


  df_test = df.sample(n=df.shape[0]//2)
  df_train_all = df.drop(df_test.index)
  y_test = df_test[target_variable]
  X_test = df_test.drop(target_variable, axis=1)

  which_methods = ['model_nocat','model_tarreg','GLMM','OneHot','JoeChar','JoeOhe','ORD']

  if len(continuous_features) == 0:
    which_methods.remove('model_nocat')


  for i in range(how_many_iterations):
      print(f"Running iteration {i+1}/{how_many_iterations}...")
      # Pass a placeholder for summary_df as it's modified within run_iteration
      iteration_result = run_iteration(i, df_train_all, X_test, y_test, categorical_indicator, categorical_features, continuous_features, target_variable, total_epochs, df, which_methods)
      results[dataset_id].append(iteration_result)

  # Save the results after processing each dataset ID
  which_open = '/home/ir318/trial_experiments/results_real.pkl'
  with open(which_open, 'wb') as f:
      pickle.dump(results, f)
  print(f"Results for dataset {dataset_id} saved.")
  
  
  