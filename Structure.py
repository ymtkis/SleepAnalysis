import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_path = '/mnt/d/Q project'
if not os.path.exists(base_path):
    base_path = '/mnt/e/Q project'
conditions = {'QIH(48h)': 4}
epoch_per_1d = 3600
# 'QIH(24h)': 3, 'QIH(48h)': 4, 'QIH(72h)': 5, 'Ctrl_Stim(24h)': 3, 'SD': 2.5, 'SD+QIH(1h)': 2.5, , 'SD+QIH(48h)': 4

def stage_indexing(base_path, condition, coeff):
    # Dataframe for label
    stage_file = f'{base_path}/EEGEMG/Compile/{condition}/{condition}_staging.xlsx'
    stage_dict = pd.read_excel(stage_file, usecols=[3], skiprows=2, sheet_name=None, header=None)
    stage_df = pd.DataFrame()
    for sheet_name, stage_data in stage_dict.items():
        stage_df[sheet_name] = stage_data
    
    label_indices = pd.DataFrame()
    for sheet_name in stage_df.columns:
        stage_data = stage_df[sheet_name]
        label_data = pd.Series(index=stage_data.index)
        label_data[stage_data == "W"] = 1
        label_data[stage_data == "NR"] = 2
        label_data[stage_data == "R"] = 3
        label_data.fillna(4, inplace=True) 
        label_indices[sheet_name] = label_data
    label_indices = label_indices.astype(int)

    if condition in ['SD', 'SD+QIH(1h)', 'SD+QIH(24h)']:
        bsl_first_L = label_indices.iloc[:int(epoch_per_1d * 0.5)]
        bsl_first_D = label_indices.iloc[int(epoch_per_1d * 0.5) : epoch_per_1d]
        bsl_second_L = label_indices.iloc[epoch_per_1d : int(epoch_per_1d * 1.5)]
        bsl_second_D = label_indices.iloc[int(epoch_per_1d * 1.5) : epoch_per_1d * 2]
        bsl_label_indices = pd.concat([bsl_first_D.reset_index(drop=True), bsl_first_L.reset_index(drop=True),
                                       bsl_second_D.reset_index(drop=True), bsl_second_L.reset_index(drop=True)], ignore_index=True)
    else:    
        bsl_label_indices = label_indices[:epoch_per_1d * 2]
    post_label_indices = label_indices[int(epoch_per_1d * coeff) : int(epoch_per_1d * (coeff + 2))]
    label_indices = {'BSL': bsl_label_indices, 'Post': post_label_indices}

    return label_indices


def SleepAnalysis_barplot(condition, data, time, epoch_per_1d):
    
    colors = {1:(1, 0.5882, 0.5882), 2:(0.5333, 0.7412, 0.9882), 3:(0.4392, 0.6784, 0.2784), 4:(1, 1, 1)}
    stages = ['', 'W', 'NR', 'R']
    fig, ax = plt.subplots(figsize=(10, 6), sharex=True)
    num_samples = len(data.columns)
    bar_height = 1 / num_samples

    for i, column in enumerate(data.columns):
        indiv_data = data[column]
        for j, value in enumerate(indiv_data):
            ax.barh(i * bar_height, 1, left=j, height=bar_height, color=colors[value])
        
        xticks = np.arange(0, data.shape[0] + epoch_per_1d, epoch_per_1d)
        xticklabels = np.arange(0, len(xticks) * 24, 24)
        ax.set_yticks([])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Time (h)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 
        ax.spines['left'].set_visible(False)
        
    fig.suptitle(time, fontsize=24)
    legend_elements = [plt.Rectangle((0,0),1,1, color=colors[ep], label=stages[ep]) for ep in list(colors.keys())[:3]]
    fig.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    output_path = f'{base_path}/SleepAnalysis/Structure/{condition}/{condition}_{time}.tif'
    plt.savefig(output_path, format='tiff', dpi=350)


for condition, coeff in conditions.items():
    label_indices = stage_indexing(base_path, condition, coeff)
    for time, data in label_indices.items():
        SleepAnalysis_barplot(condition, data, time, epoch_per_1d)

    print(f'{condition}_Structure  <Done>')




