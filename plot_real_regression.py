

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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
    'model_simpletarget': 'TAR_SIMPLE',
}

methods_list = [
    'ORD',
    'OH',
    'TAR_REG',
    'GLMM',
    'ENT',
    'JOEOH_3SIG',
    'JOECHAR_1LIN',
    'JOECHAR_3SIG',
    'NO CAT'
]


with open('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/trial_experiments/results_real.pkl', 'rb') as f:
    results = pickle.load(f)


dataset_whichones = [41211, 41445, 41210]
processed_results = {}


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
    
        # keep raw values + summary stats
        for model_name, metrics_data in model_metrics.items():
            processed_results[dataset_id][model_name] = {}
            for metric_name, values in metrics_data.items():
                processed_results[dataset_id][model_name][metric_name] = {
                    'values': values,
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }


metric_names_to_plot = ['Mean Squared Error', 'R-squared']
n_datasets = len(processed_results)
n_metrics = len(metric_names_to_plot)

fig, axes = plt.subplots(
    n_datasets, n_metrics,
    figsize=(8 * n_metrics, 6 * n_datasets),
    squeeze=False
)

dataset_ids = list(processed_results.keys())
reverse_map = {v: k for k, v in model_name_mapping.items()}

for row_index, dataset_id in enumerate(dataset_ids):
    models_data = processed_results[dataset_id]
    models_data.pop('model_tarsimple', None)  # drop unwanted model

    for col_index, metric_name in enumerate(metric_names_to_plot):
        ax = axes[row_index, col_index]

        boxplot_data = []
        colors = []
        model_names = []

        # enforce method order
        for pretty_name in methods_list:
            if pretty_name in reverse_map:
                original_model_name = reverse_map[pretty_name]
                if original_model_name in models_data and metric_name in models_data[original_model_name]:
                    values = models_data[original_model_name][metric_name]['values']
                    
                    if metric_name == 'Mean Squared Error':
                        values = np.sqrt(values)
                    boxplot_data.append(values)
                    model_names.append(original_model_name)
                    

                    colors.append(method_colors.get(pretty_name, "red"))

        if model_names:
            bp = ax.boxplot(
                boxplot_data,
                positions=np.arange(len(model_names)),
                widths=0.6,
                patch_artist=True
            )

            # Apply colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                patch.set_linewidth(2)

            for median in bp['medians']:
                median.set(color="black", linewidth=2)

            for mean in bp['means']:
                mean.set(marker="o", markerfacecolor="white",
                         markeredgecolor="black", markersize=8)

            ax.set_xticks([])
            if 'squared' in metric_name:
                ax.set_ylabel('$R^2$', fontsize=25)
            else:
                ax.set_ylabel('Dataset '+str(dataset_id)+'\n  \n RMSE', fontsize=25)
            ax.tick_params(axis='y', labelsize=25)



legend_patches = [
    Patch(facecolor=method_colors.get(pretty_name, "red"),
          edgecolor="black", label=pretty_name)
    for pretty_name in methods_list
]

fig.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=5,
    bbox_to_anchor=(0.5, -0.07),
    fontsize=23
)

plt.tight_layout(rect=[0.05, 0, 0.9, 1])
plt.savefig('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/regression_real_results_boxplots.pdf',
            bbox_inches='tight')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


entropy_values = {}
cardinality_values = {}
proportion_values = {}

for dataset_id, dataset_results in results.items():
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
        "Value": [v for d in data_dict for v in data_dict[d]],
        "Metric": metric_name
    })

df_entropy = dict_to_long_df(entropy_values, "Entropy")
df_cardinality = dict_to_long_df(cardinality_values, "Cardinality")
df_proportion = dict_to_long_df(proportion_values, "Proportion")


df_summary = pd.concat([df_entropy, df_cardinality, df_proportion], axis=0)


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

metrics = ["Entropy", "Cardinality", "Proportion"]
xlabels = ["Normalised Entropy", "Cardinality", "Proportion"]


for i, (metric, xlabel) in enumerate(zip(metrics, xlabels)):
    sns.boxplot(
        ax=axes[i],
        data=df_summary[df_summary["Metric"] == metric],
        x="Value",
        y="Dataset",
        orient="h",
        color = 'gray',
        order = dataset_whichones 
    )
    axes[i].set_xlabel(xlabel, fontsize=25)
    axes[i].set_ylabel("Dataset ID", fontsize=25)
    axes[i].tick_params(axis='y', labelsize=25)
    axes[i].tick_params(axis='x', labelsize=25)

plt.tight_layout()
plt.savefig('/Users/roatisiris/Desktop/for_cluster/final_experiments/new_again/regression_real_details_boxplots.pdf',
            bbox_inches='tight')
plt.show()
