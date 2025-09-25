#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary classification results visualization
Adjusted so CONT and NO CAT are close to the bins (no extra spacing)
"""

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --- Load results ---
intnoint = 'noint'
which_one = 'results_binary_classification_synthetic_'+intnoint+'.pkl'
with open("/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/trial_experiments/"+which_one, "rb") as f:
    results = pickle.load(f)

# Extract dictionaries
performance_results = results['dictionary_performance']
performance_results_tarreg = results['dictionary_performance_tarreg']
performance_results_joechar1 = results['dictionary_performance_joechar1']
performance_results_joechar2 = results['dictionary_performance_joechar2']
performance_results_glmm = results['dictionary_performance_glmm']
performance_results_onehot = results['dictionary_performance_onehot']
performance_results_woe = results['dictionary_performance_woe']
performance_results_ordinal = results['dictionary_performance_ordinal']
performance_results_continuous = results['dictionary_performance_continuous']
performance_results_nocat = results['dictionary_performance_nocat']
performance_results_joeohe1 = results['dictionary_performance_joeohe1']
performance_results_joeohe2 = results['dictionary_performance_joeohe2']

all_performance_data = []
desired_metrics = ['auc', 'brier']

def append_data(data_dict, method_name):
    for nr_bins, metric_data in data_dict.items():
        try:
            nr_bins = int(nr_bins)
        except Exception:
            pass
        for metric in desired_metrics:
            if metric in metric_data:
                for value in metric_data[metric]:
                    all_performance_data.append({
                        'Metric': metric,
                        'Nr_bins': nr_bins,
                        'Method': method_name,
                        'Value': value
                    })

# Append data for binned methods
append_data(performance_results_tarreg, 'TAR_REG')
# append_data(performance_results, 'TAR_SIMPLE')  # Excluded 'TAR_SIMPLE'
append_data(performance_results_joechar1, 'JOECHAR_3SIG')
append_data(performance_results_joechar2, 'JOECHAR_1LIN')
append_data(performance_results_glmm, 'GLMM')
append_data(performance_results_onehot, 'OH')
append_data(performance_results_woe, 'WOE')
append_data(performance_results_ordinal, 'ORD') # 'ORD' is included
append_data(performance_results_joeohe1, 'JOEOH_3SIG')
append_data(performance_results_joeohe2, 'ENT')

# Append continuous and no-cat as special categories
for metric in desired_metrics:
    if metric in performance_results_continuous:
        for value in performance_results_continuous[metric]:
            all_performance_data.append({
                'Metric': metric,
                'Nr_bins': 'CONT',
                'Method': 'CONT',
                'Value': value
            })
    if metric in performance_results_nocat:
        for value in performance_results_nocat[metric]:
            all_performance_data.append({
                'Metric': metric,
                'Nr_bins': 'NO CAT',
                'Method': 'NO CAT',
                'Value': value
            })

df_performance = pd.DataFrame(all_performance_data)

unique_methods = df_performance['Method'].unique()
colors = sns.color_palette("bright", len(unique_methods))
method_colors = dict(zip(unique_methods, colors))




list_nr_bins = results['dictionary_performance'].keys()

if 'ENT' in method_colors and 'NO CAT' in method_colors:
    # Get the original colors
    ent_color = method_colors['ENT']
    nocat_color = method_colors['NO CAT']
    
    # Swap the colors
    method_colors['ENT'] = nocat_color
    method_colors['NO CAT'] = ent_color
    



method_aliases = {
    'Ordinal': 'ORD',
    'One-Hot': 'OH',
    'Reg Target': 'TAR_REG',
    'GLMM': 'GLMM',
    'JoeOhe Linear': 'ENT',
    'JoeOhe Sigmoid': 'JOEOH_3SIG',
    'JoeChar Linear': 'JOECHAR_1LIN',
    'JoeChar Sigmoid': 'JOECHAR_3SIG'
    
}

# Collect performance data
performance_data = {}
for nr_bins in list_nr_bins:
    performance_data[nr_bins] = {
        'Reg Target': performance_results_tarreg[nr_bins],
        'GLMM': performance_results_glmm[nr_bins],
        'Ordinal': performance_results_ordinal[nr_bins],
        'JoeChar Sigmoid': performance_results_joechar1[nr_bins],
        'JoeChar Linear': performance_results_joechar2[nr_bins],
        'One-Hot': performance_results_onehot[nr_bins],
        'JoeOhe Sigmoid': performance_results_joeohe1[nr_bins],
        'JoeOhe Linear': performance_results_joeohe2[nr_bins]
    }

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 15))

for i, performance_metric in enumerate(['auc', 'brier']):
    ax = axes[i]
    data_to_plot = []
    labels = []
    colors = []
    bin_boundaries = []
    bin_centers = []
    
    count = 0

    methods_list = [
        'Ordinal', 
        'One-Hot',
        'Reg Target',
        'GLMM',
        'JoeOhe Linear',
        'JoeOhe Sigmoid', 
    'JoeChar Linear','JoeChar Sigmoid']
    plotted_methods = set()

    for nr_bins in list_nr_bins:
        for method in methods_list:
            if performance_metric in performance_data[nr_bins][method]:
                values = performance_data[nr_bins][method][performance_metric]
                data_to_plot.append(values)
                
                labels.append(f'{nr_bins} cat\n{method}')
                color_key = method_aliases[method]
                colors.append(method_colors[color_key])
                plotted_methods.add(color_key)
        count += len(methods_list)
        bin_boundaries.append(count)
        bin_centers.append(count - len(methods_list) / 2)

    # Continuous
    if performance_metric in performance_results_continuous:
        values = performance_results_continuous[performance_metric]
        data_to_plot.append(values)
        labels.append('Continuous')
        colors.append(method_colors['CONT'])
        plotted_methods.add('CONT')
        bin_boundaries.append(count)
        bin_centers.append(count + 0.5)

    # No-cat
    if performance_metric in performance_results_nocat:
        values = performance_results_nocat[performance_metric]
        data_to_plot.append(values)
        labels.append('NO CAT')
        colors.append(method_colors['NO CAT'])
        plotted_methods.add('NO CAT')
        bin_boundaries.append(count + 1)
        bin_centers.append(count + 1.5)

    # Boxplot
    box = ax.boxplot(data_to_plot, labels=labels, vert=True, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Group separators
    for boundary in bin_boundaries[:-1]:
        ax.axvline(boundary + 0.5, color='gray', linestyle=':', linewidth=1)

    # Custom x labels
    xtick_positions = []
    xtick_labels = []

    for idx, nr_bins in enumerate(list_nr_bins):
        start_index = idx * len(methods_list)
        end_index = start_index + len(methods_list)
        if data_to_plot[start_index:end_index]:
            xtick_positions.append(np.mean(range(start_index + 1, end_index + 1)))
            xtick_labels.append(f"{nr_bins} cat")

    # Baselines
    baseline_positions = []
    baseline_labels = []
    if 'CONT' in plotted_methods:
        cont_index = len(list_nr_bins) * len(methods_list)
        baseline_positions.append(cont_index + 1)
        baseline_labels.append('CONT')
    if 'NO CAT' in plotted_methods:
        nocat_index = len(list_nr_bins) * len(methods_list) + (1 if 'CONT' in plotted_methods else 0)
        baseline_positions.append(nocat_index + 1)
        baseline_labels.append('NO CAT')
    
    # Combine and sort for correct display order
    all_xtick_positions = xtick_positions + baseline_positions
    all_xtick_labels = xtick_labels + baseline_labels
    
    all_xtick_positions[-1] += 1
    all_xtick_positions[-2] += 1
    
    ax.set_xticks(all_xtick_positions,fontsize = 23)
    ax.set_xticklabels(all_xtick_labels, rotation=60, ha='right',fontsize = 23)
    # ax.set_xlabel('Number of Categories / Baseline', fontsize=12)
    if performance_metric == 'auc':
        ax.set_ylabel(performance_metric.upper(), fontsize=23)
    else:
        ax.set_ylabel('BS', fontsize=23)
    ax.grid(True, axis='y')
    ax.tick_params(axis='y', labelsize=23)

    
    ax.legend([],[], frameon=False) # Hide individual subplot legends

# Shared legend
legend_patches = []
for key, val in method_aliases.items():
    if val in plotted_methods:
        legend_patches.append(Patch(facecolor=method_colors[val], label=val))

if 'CONT' in plotted_methods:
    legend_patches.append(Patch(facecolor=method_colors['CONT'], label='CONT'))
if 'NO CAT' in plotted_methods:
    legend_patches.append(Patch(facecolor=method_colors['NO CAT'], label='NO CAT'))

fig.legend(handles=legend_patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05),fontsize = 23)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# Save
if 'noint' in which_one:
    plt.savefig('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/binary_noint.pdf', bbox_inches='tight')
else:
    plt.savefig('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/binary_int.pdf', bbox_inches='tight')

plt.show()
plt.close()

with open("method_colors.pkl", "wb") as f:
    pickle.dump(method_colors, f)