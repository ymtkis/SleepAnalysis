import os
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import math

# Load
base_path = '/mnt/d/Q project'
if not os.path.exists(base_path):
    base_path = '/mnt/e/Q project'
conditions = {'QIH(48h)': 12}
stages = ['W', 'NR', 'R']
# 'QIH(24h)': 10, 'QIH(48h)': 12, 'QIH(72h)': 14, 'Ctrl_Stim(24h)': 10, 'SD': 9, 'SD+QIH(48h)': 12 


def calculate_episode_duration(data):
    counts = {}
    current_length = 0
    current_stage = data.iloc[0]
    bin = 5 if current_stage != 'R' else 3

    for stage in range(1, len(data)):
        next_stage = data.iloc[stage]
        if next_stage == current_stage:
            current_length += 1
        else:
            binned_length = (current_length - 1) // bin * bin + bin / 2
            if binned_length > 0:
                counts.setdefault(current_stage, {}).setdefault(binned_length, 0)
                counts[current_stage][binned_length] += 1
                current_stage = next_stage
                current_length = 1
                bin = 5 if current_stage != 'R' else 3

    # 最後のステージの処理
    binned_length = (current_length - 1) // bin * bin + bin / 2
    if binned_length > 0:
        counts.setdefault(current_stage, {}).setdefault(binned_length, 0)
        counts[current_stage][binned_length] += 1
        counts = {stage: dict(sorted(lengths.items())) for stage, lengths in counts.items()}
        stage_order = ['W', 'NR', 'R']
        counts = {stage: counts.get(stage, {}) for stage in stage_order}

    return counts


def unify_counts_keys(counts_list):
    # 各ステージごとに存在する長さのキーのリストを保持
    stage_lengths = {}
    for counts in counts_list:
        for stage, lengths in counts.items():
            stage_lengths.setdefault(stage, set()).update(lengths.keys())

    # add episode length
    unified_counts_list = []
    for counts in counts_list:
        unified_counts = {}
        for stage, lengths in counts.items():
            existing_lengths = sorted(stage_lengths[stage])
            unified_lengths = {length: lengths.get(length, 0) for length in existing_lengths}
            unified_counts[stage] = unified_lengths
        unified_counts_list.append(unified_counts)

    return unified_counts_list


for condition, div_num in conditions.items():
    staging_file = f'{base_path}/EEGEMG/Compile/{condition}/{condition}_staging.xlsx'        
    output_path = f'{base_path}/SleepAnalysis/Episode duration/{condition}/{condition}_Episode duration.xlsx'
    xls = pd.ExcelFile(staging_file)
    
    staging_data = []
    for sheet_name in xls.sheet_names:
        staging_data_indiv = pd.read_excel(staging_file, sheet_name=sheet_name).iloc[1:, 2]
        staging_data.append(staging_data_indiv)
    staging_data = pd.DataFrame(staging_data).transpose()

    # Data division per 12h
    division_size = len(staging_data) // div_num
    if condition == 'SD':
        subsets = {
            'BSL_D': staging_data[division_size:2 * division_size],
            'BSL_L': staging_data[:division_size],
            'Post_D': staging_data[(div_num - 4) * division_size:(div_num - 3) * division_size],
            'Post_L': staging_data[(div_num - 3) * division_size:(div_num - 2) * division_size]
        }
    elif condition == 'QIH(12h)':
        subsets = {
            'BSL_D': staging_data[division_size:2 * division_size],
            'BSL_L': staging_data[:division_size],
            'Post_D': staging_data[(div_num - 4) * division_size:(div_num - 3) * division_size],
            'Post_L': staging_data[(div_num - 5) * division_size:(div_num - 4) * division_size]
        }
    else:
        subsets = {
            'BSL_D': staging_data[:division_size],
            'BSL_L': staging_data[division_size:2 * division_size],
            'Post_D': staging_data[(div_num - 4) * division_size:(div_num - 3) * division_size],
            'Post_L': staging_data[(div_num - 3) * division_size:(div_num - 2) * division_size]
        }   

    episode_duration_per_timeseg = {timeseg: [] for timeseg in subsets.keys()}
    for timeseg, subset in subsets.items():
        subset.columns = range(subset.shape[1])
        for column in subset.columns:
            episode_duration_col = calculate_episode_duration(subset[column].dropna().reset_index(drop=True))
            episode_duration_per_timeseg[timeseg].append(episode_duration_col)

    episode_duration_per_timeseg = {timeseg: unify_counts_keys(episode_duration_list) 
                                    for timeseg, episode_duration_list in episode_duration_per_timeseg.items()}

   
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for timeseg, episode_duration_all_mouse in episode_duration_per_timeseg.items():
            for stage in stages: 
                stage_dfs = []
                for mouse_index, episode_duration_all_stage in enumerate(episode_duration_all_mouse):
                    if stage in episode_duration_all_stage:
                        episode_durations = episode_duration_all_stage[stage]
                        stage_df = pd.DataFrame(episode_durations, index = [f'Individual {mouse_index + 1}']).T 
                        stage_dfs.append(stage_df)
                stage_dfs = pd.concat(stage_dfs, axis=1)
                stage_dfs.to_excel(writer, sheet_name=f'{timeseg}_{stage}')
    
    print(f'{condition}_data compile  <Done>')



def plot_episode_duration(data, stage):   

    plt.rcParams.update({'font.size': 20})

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    plt.suptitle(stage)
    #colors = {'W': (1, 0.5882, 0.5882), 'NR': (0.5333, 0.7412, 0.9882), 'R': (0.4392, 0.6784, 0.2784)}
    x_max =  {'W': 60, 'NR': 40, 'R': 4}

    LD = ['D', 'L']
    for i, ld in enumerate(LD):
        BSL_data = pd.read_excel(data, sheet_name=f'BSL_{ld}_{stage}')
        post_data = pd.read_excel(data, sheet_name=f'Post_{ld}_{stage}')

        BSL_means = BSL_data.iloc[1:, 1:].mean(axis=1) 
        BSL_sems = BSL_data.iloc[1:, 1:].sem(axis=1) 
        post_means = post_data.iloc[1:, 1:].mean(axis=1) 
        post_sems = post_data.iloc[1:, 1:].sem(axis=1) 

        epoch_to_min = 7.5
        BSL_x = BSL_data.iloc[1:, 0] / epoch_to_min
        post_x = post_data.iloc[1:, 0] / epoch_to_min

        axs[i].plot(BSL_x, BSL_means, marker='None', linestyle='-', color='gray', label='BSL', linewidth=5)
        axs[i].fill_between(BSL_x, BSL_means - BSL_sems, BSL_means + BSL_sems, color='gray', alpha=0.5)
        axs[i].plot(post_x, post_means, marker='None', linestyle='-', color='red', label='Post', linewidth=5)
        axs[i].fill_between(post_x, post_means - post_sems, post_means + post_sems, color='red', alpha=0.5)
        
        axs[i].set_xticks(np.arange(0, x_max[stage] + 1, x_max[stage] / 4))
        axs[i].set_xlim([0, x_max[stage]])
        axs[i].set_yticks([0, 5, 10, 15])
        axs[i].set_yticklabels(['0', '5', '10', '15'])
        axs[i].set_ylim([0, 15])
        axs[i].set_xlabel('Episode Duration (min)')
        axs[i].set_ylabel('Count')

        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

        # Statistics
        significant_x = []
        p_values = []  # p値を保持するリスト
        x_labels = []  # p値に対応するラベルを保持するリスト

        post_keys = set(post_data.iloc[:, 0].unique())
        for idx in range(len(BSL_data.index)):
            if BSL_data.iloc[idx, 0] in post_keys:
                corresponding_rows = post_data[post_data.iloc[:, 0] == BSL_data.iloc[idx, 0]]
                for _, post_row in corresponding_rows.iterrows():
                    u_stat, p_val = mannwhitneyu(BSL_data.iloc[idx, 1:], post_row[1:], alternative='two-sided')
                    p_values.append(p_val)
                    x_labels.append(BSL_data.iloc[idx, 0])

        # Holm法によるp値の補正
        p_adjusted = multipletests(p_values, alpha=0.05, method='holm')[1]

        # 補正後のp値が0.05未満のものをsignificant_xに追加
        significant_x = [x for x, p in zip(x_labels, p_adjusted) if p < 0.05]
        for x in significant_x:
            axs[i].plot(x, 15*0.98, marker='s', markersize=8, color='red', alpha=0.5)

           
    axs[0].axvspan(0, x_max[stage], color='gray', alpha=0.2)
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

    return fig



for condition in conditions:
    output_path = f'{base_path}/SleepAnalysis/Episode duration/{condition}/{condition}_Episode duration.xlsx'
    reload = pd.ExcelFile(output_path)
    for stage in stages:
        fig = plot_episode_duration(reload, stage)
        fig.savefig(f'{base_path}/SleepAnalysis/Episode duration/{condition}/{condition}_{stage}.tif', format='tif', dpi=300)

    print(f'{condition}_figure  <Done>')