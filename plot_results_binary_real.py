#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 09:14:50 2025

@author: roatisiris
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
with open("method_colors.pkl", "rb") as f:
    method_colors = pickle.load(f)

model_name_mapping = {
    'model_nocat': 'NO CAT',
    'model_tarreg': 'TAR_REG',
    'JoeChar3Sig': 'JOECHAR_3SIG',
    'JoeChar1Lin': 'JOECHAR_1LIN',
    'GLMM': 'GLMM',
    'OneHot': 'OH',
    'model_woe': 'WOE',
    'ORD': 'ORD',
    'model_continuous': 'Continuous',
    'JoeOhe3Sig': 'JOEOH_3SIG',
    'JoeOhe1Lin': 'ENT',
}

methods_list = [
    'ORD',          # Ordinal
    'OH',          # One-Hot
    'TAR_REG',      # Reg Target
    'GLMM',         # GLMM
    'ENT',  # JoeOhe Linear
    'JOEOH_3SIG',  # JoeOhe Sigmoid
    'JOECHAR_1LIN', # JoeChar Linear
    'JOECHAR_3SIG',  # JoeChar Sigmoid,
    'NO CAT'
]

processed_results = {}
dictionary_proportions = {}
dictionary_minentropy = {}
dictionary_maxentropy = {}
dictionary_meanentropy = {}
dictionary_varentropy = {}
dictionary_mincardinality = {}
dictionary_maxcardinality = {}
dictionary_meancardinality = {}
dictionary_varcardinality = {}
dictionary_minproportion = {}
dictionary_maxproportion = {}
dictionary_meanproportion = {}
dictionary_varproportion = {}

with open('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/trial_experiments/results.pkl', 'rb') as f:
    results = pickle.load(f)
    
    
dataset_whichones = [41283, 41434, 981]



for dataset_id, dataset_results in results.items():
    if dataset_id in dataset_whichones:
        processed_results[dataset_id] = {}
        model_metrics = {}
    
        for iteration_results in dataset_results:
            for model_name, metrics in iteration_results.items():
                if model_name not in ['summaries', 'proportion_ones_train']:
                    if model_name not in model_metrics:
                        model_metrics[model_name] = {}
                    for metric_name, value in metrics.items():
                        if metric_name not in model_metrics[model_name]:
                            model_metrics[model_name][metric_name] = []
                        model_metrics[model_name][metric_name].append(value)
    
        for model_name, metrics_data in model_metrics.items():
            processed_results[dataset_id][model_name] = {}
            for metric_name, values in metrics_data.items():
                processed_results[dataset_id][model_name][metric_name] = {
                    'values': values,
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
metric_names_to_plot = ['ROC AUC Score', 'Brier Score']
n_datasets = len(processed_results)
n_metrics = len(metric_names_to_plot)

fig, axes = plt.subplots(
    n_datasets, n_metrics,
    figsize=(8 * n_metrics, 6 * n_datasets + 2),
    squeeze=False
)

dataset_ids = list(processed_results.keys())
reverse_map = {v: k for k, v in model_name_mapping.items()}




for row_index, dataset_id in enumerate(dataset_ids):
    print(dataset_id)
    models_data = processed_results[dataset_id]
    
    models_data.pop('model_tarsimple', None)

    for col_index, metric_name in enumerate(metric_names_to_plot):
        ax = axes[row_index, col_index]

        model_names = []
        boxplot_data = []
        colors = []

        for pretty_name in methods_list:
            if pretty_name in reverse_map:
                original_model_name = reverse_map[pretty_name]
                if (original_model_name in models_data 
                    and metric_name in models_data[original_model_name]):
                    
                    metric_dict = models_data[original_model_name][metric_name]
                    values = metric_dict['values']
                    boxplot_data.append(values)
                    model_names.append(original_model_name)

                    explicit_model_name = model_name_mapping.get(original_model_name, original_model_name)
                    colors.append(method_colors.get(explicit_model_name, "red"))

        if model_names:
            bp = ax.boxplot(
                boxplot_data,
                positions=np.arange(len(model_names)),
                widths=0.6,
                patch_artist=True            )

            # Apply colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                patch.set_linewidth(2)

            for median in bp['medians']:
                median.set(color="black", linewidth=2)

            ax.set_xticks([])
            if 'AUC' in metric_name:
                ax.set_ylabel('Dataset '+str(dataset_id)+'\n  \n AUC', fontsize=25)
            else:
                ax.set_ylabel('BS', fontsize=25)
            ax.tick_params(axis='y', labelsize=25)



legend_patches = []
legend_labels = []
for pretty_name in methods_list:
    if pretty_name in model_name_mapping.values(): 
        color = method_colors.get(pretty_name, "red")
        legend_patches.append(Patch(facecolor=color, edgecolor="black", label=pretty_name))
        legend_labels.append(pretty_name)


fig.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=5,
    bbox_to_anchor=(0.5, -0.07),
    fontsize=23
)

plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.savefig('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/binary_real_results.pdf',
            bbox_inches='tight')
plt.show()





entropy_values = {}
cardinality_values = {}
proportion_values = {}


dataset_ids = list(processed_results.keys())
print(dataset_ids)
for dataset_id in dataset_ids:
    dataset_results = results[int(dataset_id)]
    entropy_values[dataset_id] = []
    cardinality_values[dataset_id] = []
    proportion_values[dataset_id] = []
    
    for iter_item in dataset_results:
        entropy_values[dataset_id].extend(iter_item['summaries']['Entropy'].tolist())
        cardinality_values[dataset_id].extend(iter_item['summaries']['Cardinality'].tolist())
        proportion_values[dataset_id].extend(iter_item['summaries']['Proportion'].tolist())

def dict_to_long_df(data_dict, metric_name):
    return pd.DataFrame({
        "Dataset": [d for d in data_dict for _ in data_dict[d]],
        "Value": [v for d in data_dict for v in data_dict[d]]
    })

df_entropy = dict_to_long_df(entropy_values, "Entropy")
df_cardinality = dict_to_long_df(cardinality_values, "Cardinality")
df_proportion = dict_to_long_df(proportion_values, "Proportion")

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

sns.boxplot(ax=axes[0], data=df_entropy, x="Value", y="Dataset",
            orient="h", color = 'gray',  order=dataset_whichones)
axes[0].set_xlabel("Normalised Entropy", fontsize=25)
axes[0].set_ylabel("Dataset ID", fontsize=25)
axes[0].tick_params(axis='y', labelsize=25)
axes[0].tick_params(axis='x', labelsize=25)

sns.boxplot(ax=axes[1], data=df_cardinality, x="Value", y="Dataset",
            orient="h", color = 'gray',  order=dataset_whichones)
axes[1].set_xlabel("Cardinality", fontsize=25)
axes[1].set_ylabel("Dataset ID", fontsize=25)
axes[1].tick_params(axis='y', labelsize=25)
axes[1].tick_params(axis='x', labelsize=25)

sns.boxplot(ax=axes[2], data=df_proportion, x="Value", y="Dataset",
            orient="h", color = 'gray',  order=dataset_whichones)
axes[2].set_xlabel("Proportion", fontsize=25)
axes[2].set_ylabel("Dataset ID", fontsize=25)
axes[2].tick_params(axis='y', labelsize=25)
axes[2].tick_params(axis='x', labelsize=25)

plt.tight_layout()
plt.savefig('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/binary_real_details.pdf', bbox_inches='tight')
plt.show()


