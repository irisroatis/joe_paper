#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:24:46 2025

@author: roatisiris
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from category_encoders import TargetEncoder, OneHotEncoder, GLMMEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon
import itertools
import matplotlib.patches as mpatches



def generate_data(N, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # cat1 generation remains the same
    how_many_categories = 250
    cat1_levels = [f"cat1_{i}" for i in range(how_many_categories)]
    cat1 = np.random.choice(cat1_levels, N, p=[1/how_many_categories]*how_many_categories)
    cat1_effects = {lvl: np.random.normal(0, 0.5) for lvl in cat1_levels}
    
    # Generate continuous features first
    cont1 = np.random.normal(0, 1, N)
    cont2 = np.random.normal(0, 1, N)
    
    categories = ['A', 'B', 'C', 'D']
    # Define probabilities to create imbalanced categories for cat2
    probabilities = [0.60, 0.15, 0.20, 0.05]
    cat2 = np.random.choice(categories, N, p=probabilities)
    
    cat2_effects = {'A': 1.5, 'B': 0.5, 'C': -0.5, 'D': -1.5}
    cat2_term = np.array([cat2_effects[c] for i, c in enumerate(cat2)])
    
    # Calculate logits using the new interaction term
    logits = np.array([cat1_effects[c] for c in cat1]) + cat2_term + 0.1 * cont1 + np.random.normal(0, 0.5, N)
    
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)
    return pd.DataFrame({'cat1': cat1, 'cat2': cat2, 'cont1': cont1, 'cont2': cont2, 'y': y})

# Fixed test set for consistent evaluation
df_test = generate_data(10000, seed=999)
X_test = df_test.drop('y', axis=1)
y_test = df_test['y']


def evaluate_logistic_model(X_train_enc, X_test_enc, y_train, y_test):

    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_train_enc, y_train)
    preds = clf.predict_proba(X_test_enc)[:, 1]
    auc = roc_auc_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    return auc, brier


def get_encoded_data(encoder_cat1, encoder_cat2, X_train, y_train, X_test):

    # Create copies of the encoders to ensure independent fitting
    enc1 = encoder_cat1.__class__()
    enc2 = encoder_cat2.__class__()
    
    # Encode cat1
    X_train_cat1 = enc1.fit_transform(X_train[['cat1']], y_train)
    X_test_cat1 = enc1.transform(X_test[['cat1']])
    
    # Encode cat2
    X_train_cat2 = enc2.fit_transform(X_train[['cat2']], y_train)
    X_test_cat2 = enc2.transform(X_test[['cat2']])
    
    # Reset indices for safe concatenation
    X_train_cat1.index = X_train.index
    X_test_cat1.index = X_test.index
    X_train_cat2.index = X_train.index
    X_test_cat2.index = X_test.index

    # Combine encoded categorical features with continuous features
    X_train_cont = X_train[['cont1', 'cont2']]
    X_test_cont = X_test[['cont1', 'cont2']]
    
    X_train_enc = pd.concat([X_train_cat1, X_train_cat2, X_train_cont], axis=1)
    X_test_enc = pd.concat([X_test_cat1, X_test_cat2, X_test_cont], axis=1)
    
    return X_train_enc, X_test_enc


# Run a single training experiment with all 9 combinations
def run_single_experiment(seed=None):

    df_train = generate_data(1000, seed=seed)
    X_train = df_train.drop('y', axis=1)
    y_train = df_train['y']

    # Define the encoders to test
    encoders = {
        'Target': TargetEncoder(), 
        'OneHot': OneHotEncoder(use_cat_names=True), 
        'GLMM': GLMMEncoder()
    }
    
    # Create all possible combinations
    strategies = list(itertools.product(encoders.keys(), repeat=2))
    run_results = {}
    
    for enc_cat1_name, enc_cat2_name in strategies:
        strategy_name = f'cat1={enc_cat1_name}, cat2={enc_cat2_name}'
        
        # Get the encoded data for this strategy
        X_train_enc, X_test_enc = get_encoded_data(
            encoders[enc_cat1_name], 
            encoders[enc_cat2_name], 
            X_train, y_train, X_test
        )
        
        # Evaluate the model
        auc, brier = evaluate_logistic_model(X_train_enc, X_test_enc, y_train, y_test)
        run_results[strategy_name] = {'auc': auc, 'brier': brier}
        
    return run_results


n_runs = 100
all_results = {}


for seed in range(n_runs):
    run_results = run_single_experiment(seed=seed)
    
    for strategy_name, metrics in run_results.items():
        if strategy_name not in all_results:
            all_results[strategy_name] = {'AUC': [], 'Brier': []}
        all_results[strategy_name]['AUC'].append(metrics['auc'])
        all_results[strategy_name]['Brier'].append(metrics['brier'])


df_auc_results = pd.DataFrame({
    name: metrics['AUC'] for name, metrics in all_results.items()
})
df_brier_results = pd.DataFrame({
    name: metrics['Brier'] for name, metrics in all_results.items()
})

# Define base colors in RGB tuples (0-1 range)
base_colors = {
    'Target':  (0.984313725490196, 0.5019607843137255, 0.4470588235294118),  # Red for TAR
    'OneHot': (1.0, 1.0, 0.7019607843137254),  # Yellow for OHE
    'GLMM': (0.5019607843137255, 0.6941176470588235, 0.8274509803921568),    # Blue for GLMM
}

# Define the blended colors 
blended_colors_ordered = {
    ('Target', 'OneHot'): (1.0, 0.6, 0.4),      
    ('OneHot', 'Target'): (0.8, 0.45, 0.3),      

    ('Target', 'GLMM'): (0.7, 0.6, 0.8),        
    ('GLMM', 'Target'): (0.55, 0.45, 0.7),      

    ('OneHot', 'GLMM'): (0.6, 0.9, 0.8),        
    ('GLMM', 'OneHot'): (0.4, 0.7, 0.6),         
}


two_line_labels = []
color_dict = {}
for name in df_auc_results.columns:
    parts = name.split(', ')
    cat1_part_full = parts[0].replace('cat1=', '')
    cat2_part_full = parts[1].replace('cat2=', '')

    cat1_abbr = cat1_part_full.replace('Target', 'TAR').replace('OneHot', 'OH').replace('GLMM', 'GLMM')
    cat2_abbr = cat2_part_full.replace('Target', 'TAR').replace('OneHot', 'OH').replace('GLMM', 'GLMM')
    two_line_labels.append(f"cat1 = {cat1_abbr}\ncat2 = {cat2_abbr}")

    if cat1_part_full == cat2_part_full:
        color_dict[name] = base_colors[cat1_part_full]
    else:
        key = (cat1_part_full, cat2_part_full)
        color_dict[name] = blended_colors_ordered[key]


df_auc_melt = df_auc_results.melt(var_name='Encoding Strategy', value_name='ROC AUC')
df_brier_melt = df_brier_results.melt(var_name='Encoding Strategy', value_name='Brier Score')

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.boxplot(data=df_auc_melt, x='Encoding Strategy', y='ROC AUC', ax=axes[0],
            palette=color_dict)
# axes[0].set_title('ROC AUC over 50 Runs', fontsize=16)
axes[0].set_xlabel('')
axes[0].set_ylabel('AUC', fontsize=25, labelpad=20)
axes[0].set_xticks([])
axes[0].tick_params(axis='y', labelsize=20)

sns.boxplot(data=df_brier_melt, x='Encoding Strategy', y='Brier Score', ax=axes[1],
            palette=color_dict)
axes[1].set_xlabel('')
axes[1].set_ylabel('BS', fontsize=25, labelpad=20)
axes[1].set_xticks([])
axes[1].tick_params(axis='y', labelsize=25)


legend_patches = [mpatches.Patch(color=color_dict[strategy], label=label)
                  for strategy, label in zip(df_auc_results.columns, two_line_labels)]

fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=25, frameon=True,
           bbox_to_anchor=(0.5, -0.2))

plt.tight_layout(rect=[0, 0.05, 1, 0.95], w_pad=7.0)  

##### save where wanted
plt.savefig('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/example_plot.pdf', bbox_inches='tight')

plt.show()





strategy_names = list(all_results.keys())
for name1, name2 in itertools.combinations(strategy_names, 2):
    # Calculate the difference in AUC and Brier scores
    auc_diffs = np.array(all_results[name1]['AUC']) - np.array(all_results[name2]['AUC'])
    brier_diffs = np.array(all_results[name1]['Brier']) - np.array(all_results[name2]['Brier'])

    # Calculate mean and standard deviation of the differences
    auc_diff_mean = np.mean(auc_diffs)
    auc_diff_std = np.std(auc_diffs)
    brier_diff_mean = np.mean(brier_diffs)
    brier_diff_std = np.std(brier_diffs)

    print(f"\nComparing '{name1}' vs '{name2}':")
    print(f"  AUC Difference = {auc_diff_mean:.4f} ± {auc_diff_std:.4f}")
    print(f"  Brier Difference = {brier_diff_mean:.4f} ± {brier_diff_std:.4f}")
    

results_data = []

strategy_names = list(all_results.keys())
for name1, name2 in itertools.combinations(strategy_names, 2):
    # Calculate the difference in AUC and Brier scores
    auc_diffs = np.array(all_results[name1]['AUC']) - np.array(all_results[name2]['AUC'])
    brier_diffs = np.array(all_results[name1]['Brier']) - np.array(all_results[name2]['Brier'])

    # Calculate mean and standard deviation of the differences
    auc_diff_mean = np.mean(auc_diffs)
    auc_diff_std = np.std(auc_diffs)
    brier_diff_mean = np.mean(brier_diffs)
    brier_diff_std = np.std(brier_diffs)

    # # Perform statistical tests
    # t_test_p = ttest_rel(all_results[name1]['AUC'], all_results[name2]['AUC']).pvalue
    # wilcoxon_p = wilcoxon(all_results[name1]['AUC'], all_results[name2]['AUC']).pvalue

    # Append the results to the list
    results_data.append({
        'Comparison': f'{name1} vs {name2}',
        'ROC AUC Diff (A-B)': f'{auc_diff_mean:.4f} ± {auc_diff_std:.4f}',
        'Brier Score Diff (A-B)': f'{brier_diff_mean:.4f} ± {brier_diff_std:.4f}'
        # 't-test p-value': f'{t_test_p:.4f}',
        # 'Wilcoxon p-value': f'{wilcoxon_p:.4f}'
    })



