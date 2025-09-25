#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


which_one = 'results_regression_int.pkl'
# which_one = 'results_regression_noint.pkl'

with open("/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/trial_experiments/"+which_one, "rb") as f:
    results = pickle.load(f)

# Extract dictionaries
performance_results = results['dictionary_performance']
performance_results_tarreg = results['dictionary_performance_tarreg']
performance_results_joechar1 = results['dictionary_performance_joechar1']
performance_results_joechar2 = results['dictionary_performance_joechar2']
performance_results_glmm = results['dictionary_performance_glmm']
performance_results_onehot = results['dictionary_performance_onehot']
performance_results_ordinal = results['dictionary_performance_ordinal']
performance_results_continuous = results['dictionary_performance_continuous']
performance_results_nocat = results['dictionary_performance_nocat']
performance_results_joeohe1 = results['dictionary_performance_joeohe1']
performance_results_joeohe2 = results['dictionary_performance_joeohe2']

all_performance_data = []
desired_metrics = ['rmse', 'rsq']

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

append_data(performance_results_tarreg, 'TAR_REG')
append_data(performance_results_joechar1, 'JOECHAR_3SIG')
append_data(performance_results_joechar2, 'JOECHAR_1LIN')
append_data(performance_results_glmm, 'GLMM')
append_data(performance_results_onehot, 'OH')
append_data(performance_results_ordinal, 'ORD')
append_data(performance_results_joeohe1, 'JOEOH_3SIG')
append_data(performance_results_joeohe2, 'ENT')

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

if 'ENT' in method_colors and 'NO CAT' in method_colors:
    ent_color = method_colors['ENT']
    nocat_color = method_colors['NO CAT']
    
    method_colors['ENT'] = nocat_color
    method_colors['NO CAT'] = ent_color
    


list_nr_bins = results['dictionary_performance'].keys()

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
        'Simple Target': performance_results[nr_bins],
        'GLMM': performance_results_glmm[nr_bins],
        'Ordinal': performance_results_ordinal[nr_bins],
        'JoeChar Sigmoid': performance_results_joechar1[nr_bins],
        'JoeChar Linear': performance_results_joechar2[nr_bins],
        'One-Hot': performance_results_onehot[nr_bins],
        'JoeOhe Sigmoid': performance_results_joeohe1[nr_bins],
        'JoeOhe Linear': performance_results_joeohe2[nr_bins]
    }

fig, axes = plt.subplots(2, 1, figsize=(24, 15))

specific_bins_to_plot = [50, 100, 150, 200, 250, 300]

metrics = ['rmse', 'rsq']
for i, performance_metric in enumerate(metrics):
    ax = axes[i]
    
    data_to_plot = []
    labels = []
    colors = []
    bin_boundaries = []
    bin_centers = []

    count = 0
    methods_list = list(method_aliases.keys())

    for nr_bins in specific_bins_to_plot:
        for method in methods_list:
            if performance_metric in performance_data[nr_bins][method]:
                data_to_plot.append(performance_data[nr_bins][method][performance_metric])
                labels.append(f'{nr_bins} cat\n{method}')
                color_key = method_aliases[method]
                colors.append(method_colors[color_key])
        count += len(methods_list)
        bin_boundaries.append(count)
        bin_centers.append(count - len(methods_list) / 2)

    # Continuous
    if performance_metric in performance_results_continuous:
        data_to_plot.append(performance_results_continuous[performance_metric])
        labels.append('CONT')
        colors.append(method_colors['CONT'])
        bin_boundaries.append(count)
        bin_centers.append(count + 0.5)

    # No-cat
    if performance_metric in performance_results_nocat:
        data_to_plot.append(performance_results_nocat[performance_metric])
        labels.append('NO CAT')
        colors.append(method_colors['NO CAT'])
        bin_boundaries.append(count+1)
        bin_centers.append(count + 1.5)

    box = ax.boxplot(data_to_plot, labels=labels, vert=True, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    for boundary in bin_boundaries[:-1]:
        ax.axvline(boundary + 0.5, color='gray', linestyle=':', linewidth=1)

    xtick_positions = bin_centers[:-2]
    xtick_labels = [f"{nr_bins} cat" for nr_bins in specific_bins_to_plot]
    xtick_positions.extend(bin_centers[-2:])
    xtick_labels.extend(["CONT", "NO CAT"])
    
    xtick_positions[-1] += 1
    xtick_positions[-2] += 1
    ax.set_xticks(xtick_positions,fontsize = 23)
    ax.set_xticklabels(xtick_labels, rotation=60, ha='right',fontsize = 23)
    ax.tick_params(axis='y', labelsize=23)
    

    if performance_metric == 'rmse':
        ax.set_ylabel(performance_metric.upper(), fontsize=23)
    else:
        ax.set_ylabel('$R^2$', fontsize=23)
    ax.grid(True, axis='y')
    
legend_patches = [Patch(facecolor=method_colors[val], label=val) for key, val in method_aliases.items()]
legend_patches.append(Patch(facecolor=method_colors['CONT'], label='CONT'))
legend_patches.append(Patch(facecolor=method_colors['NO CAT'], label='NO CAT'))

fig.legend(handles=legend_patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0),fontsize = 23)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# Save
if 'noint' in which_one:
    plt.savefig('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/regression_noint.pdf', bbox_inches='tight')
else:
    plt.savefig('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/regression_int.pdf', bbox_inches='tight')

plt.show()
plt.close()

