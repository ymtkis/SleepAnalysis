import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import itertools

base_path = '/mnt/d/Q_project/EEGEMG'
if not os.path.exists(base_path):
    base_path = '/mnt/e/Q_project/EEGEMG'
output_path = '/mnt/d/Q_project/SleepAnalysis/Total amount'
if not os.path.exists(output_path):
    output_path = '/mnt/e/Q_project/SleepAnalysis/Total amount'
conditions = {'QIH(24h)': 3} # Values represent the date on which "Post" begins.
stages = ['W', 'NR', 'R']
# 'QIH(24h)': 3, 'QIH(48h)': 4, 'QIH(72h)': 5, 'Ctrl_Stim(24h)': 3, 'SD': 3, 'SD+QIH(1h)': 3, 'SD+QIH(2h)': 3, 'SD+QIH(24h)': 4, 'SD+QIH(48h)': 4,  


def total_amount_df(conditions, stages, base_path, output_path):

    per_12h_results = {}
    per_1h_results = {}
    test_results_dict = {}

    for condition, post_date in conditions.items():

        BSL_dfs = []
        post1_dfs = []
        post2_dfs = []
        staging_file = f'{base_path}/Compile/{condition}/{condition}_staging.xlsx'
        xls = pd.ExcelFile(staging_file)

        for sheet_name in xls.sheet_names:

            df = pd.read_excel(staging_file, sheet_name=sheet_name)
            stage_data = df.iloc[2:, 2]

            hourly_counts = []
            
            for i in range(0, len(stage_data), 450):
                hourly_div = stage_data[i:i + 450]
                counts = hourly_div.value_counts()
                hourly_counts.append(counts / 7.5) # Conversion to minutes
            hourly_df = pd.DataFrame(hourly_counts)
            hourly_df.fillna(0, inplace=True)  # NaN to 0
            hourly_df.index = range(1, len(hourly_df) + 1)  # indexing

            # BSL, Post1 and Post2
            if (condition == 'SD' and sheet_name in ['220105BD2', '1', '12', '14', '17', '18', '20', '22']) or (condition == 'SD+QIH(1h)' and sheet_name in ['66', '74', '77']) :
                post_date = 2.5

            if (condition == 'SD+QIH(24h)' and sheet_name in ['220105BD2', '220105BD4', '1', '5', '14', '18', '20', '22']):
                post_date = 3.5

            if (condition == 'SD+QIH(24h)' and sheet_name in ['220105BD2', '220105BD4', '1', '5', '14', '18', '20', '22']) or (condition == 'SD' and sheet_name in ['220105BD2', '1', '12', '14', '17', '18', '20', '22']) or (condition == 'SD+QIH(1h)' and sheet_name in ['66', '74', '77']):
                BSL1_L_df = hourly_df.iloc[0:12]
                BSL1_D_df = hourly_df.iloc[12:24]
                BSL2_L_df = hourly_df.iloc[24:36]
                BSL2_D_df = hourly_df.iloc[36:48]
                BSL1_df = pd.concat([BSL1_D_df.reset_index(drop=True), BSL1_L_df.reset_index(drop=True)], ignore_index=True)
                BSL2_df = pd.concat([BSL2_D_df.reset_index(drop=True), BSL2_L_df.reset_index(drop=True)], ignore_index=True)
                BSL_df = (BSL1_df.reset_index(drop=True) + BSL2_df.reset_index(drop=True)) / 2
            
            elif condition == 'SD+QIH(48h)':
                BSL_df = hourly_df.iloc[0:24]
                BSL_df = BSL_df.reset_index(drop=True)

            else:
                BSL1_df = hourly_df.iloc[0:24]
                BSL2_df = hourly_df.iloc[24:48]
                BSL_df = (BSL1_df.reset_index(drop=True) + BSL2_df.reset_index(drop=True)) / 2


            if condition == 'QIH(12h)':
                post1_df = hourly_df.iloc[int(24 * post_date) : int(24 * (post_date + 1))].reset_index(drop=True)
                post2_df = hourly_df.iloc[int(24 * (post_date + 1)) : int(24 * (post_date + 2))].reset_index(drop=True)

            else:
                post1_df = hourly_df.iloc[int(24 * post_date) : int(24 * (post_date + 1))].reset_index(drop=True)
                post2_df = hourly_df.iloc[int(24 * (post_date + 1)) : int(24 * (post_date + 2))].reset_index(drop=True)

            BSL_dfs.append(BSL_df)
            post1_dfs.append(post1_df)
            post2_dfs.append(post2_df)

            valid_indices = [i for i, df in enumerate(post2_dfs) if not df.empty]
            BSL_dfs_filtered = [BSL_dfs[i] for i in valid_indices]
            post2_dfs_filtered = [post2_dfs[i] for i in valid_indices]
        


        def per_12h(dfs):

            per_12h_result = {'W': [], 'NR': [], 'R': []}
            total = 720

            for df in dfs:
                per_12h_df = df[:12]
                per_12h_sum = per_12h_df.sum()  # 各列の合計時間

                for key in per_12h_result:
                    ratio = per_12h_sum.get(key, np.nan) / total * 100
                    per_12h_result[key].append(ratio)

            return per_12h_result



        # Mean and SEM
        def per_1h(dfs):

            per_1h_mean_df = pd.concat(dfs).groupby(level=0).mean()
            per_1h_sem_df = pd.concat(dfs).groupby(level=0).sem()
            
            return per_1h_mean_df, per_1h_sem_df
        


        per_12h_BSL = per_12h(BSL_dfs)
        per_12h_post1 = per_12h(post1_dfs)
        per_12h_post2 = per_12h(post2_dfs)

        keep_indices = [
            i for i in range(len(per_12h_BSL['W']))
            if per_12h_BSL['W'][i] != 0.0 and
            per_12h_post1['W'][i] != 0.0 and
            per_12h_post2['W'][i] != 0.0
        ]

        per_12h_BSL = {k: [v[i] for i in keep_indices] for k, v in per_12h_BSL.items()}
        per_12h_post1 = {k: [v[i] for i in keep_indices] for k, v in per_12h_post1.items()}
        per_12h_post2 = {k: [v[i] for i in keep_indices] for k, v in per_12h_post2.items()}

        per_1h_mean_BSL, per_1h_sem_BSL = per_1h(BSL_dfs)
        per_1h_mean_post1, per_1h_sem_post1 = per_1h(post1_dfs)
        per_1h_mean_post2, per_1h_sem_post2 = per_1h(post2_dfs)

        per_12h_results[condition] = {'per_12h_BSL': per_12h_BSL, 'per_12h_post1': per_12h_post1, 'per_12h_post2': per_12h_post2}

        per_1h_results[condition] = {
            'per_1h_mean_BSL': per_1h_mean_BSL, 'per_1h_sem_BSL': per_1h_sem_BSL,
            'per_1h_mean_post1': per_1h_mean_post1, 'per_1h_sem_post1': per_1h_sem_post1,
            'per_1h_mean_post2': per_1h_mean_post2, 'per_1h_sem_post2': per_1h_sem_post2
        }


        # Statistics
        with pd.ExcelWriter(f'{output_path}/{condition}/{condition}_anova_result.xlsx') as writer:

            anovas = [(BSL_dfs, post1_dfs, 'ANOVA_Post(0-24h)'), (BSL_dfs_filtered, post2_dfs_filtered, 'ANOVA_Post(24-48h)')]
            for anova_BSL_dfs, anova_post_dfs, anova_sheet_name in anovas:
                for stage in stages:
                    anova_BSL = []
                    anova_post = []
                    for anova_BSL_df, anova_post_df in zip(anova_BSL_dfs, anova_post_dfs):
                        for hour in range(24):                                
                            anova_BSL.append({'Condition': 'BSL', 'Hour': hour, 'Stage': stage, 'Value': anova_BSL_df.loc[hour, stage]})
                            anova_post.append({'Condition': 'Post', 'Hour': hour, 'Stage': stage, 'Value': anova_post_df.loc[hour, stage]})                
                    anova_BSL = pd.DataFrame(anova_BSL)
                    anova_post = pd.DataFrame(anova_post)
                    anova_data = pd.concat([anova_BSL, anova_post])

                    model = ols('Value ~ C(Condition) * C(Hour)', data=anova_data).fit()
                    anova_summary = pd.DataFrame(model.summary().tables[0])
                    anova_results = pd.DataFrame(model.summary().tables[1])
                    anova_results = pd.concat([anova_summary, anova_results])                
                    anova_results.to_excel(writer, sheet_name=f'{anova_sheet_name}_{stage}', index=False)

        

        with pd.ExcelWriter(f'{output_path}/{condition}/{condition}_test_result.xlsx') as writer:            
            comparisons = [(BSL_dfs, post1_dfs, 'Post(0-24h)'), (BSL_dfs_filtered, post2_dfs_filtered, 'Post(24-48h)')]
            for BSL_dfs, post_dfs, sheet_name in comparisons:
                test_results = pd.DataFrame(index=range(24), columns=[f'{stage}_p' for stage in stages] + [f'{stage}_q' for stage in stages])
                p_values = []

                for stage in stages:
                    for hour in range(24):
                        BSL_df = [df.loc[hour, stage] for df in BSL_dfs]
                        post_df = [df.loc[hour, stage] for df in post_dfs]

                        # Normality test
                        normality_BSL = stats.shapiro(BSL_df).pvalue
                        normality_post = stats.shapiro(post_df).pvalue

                        # Homogeneity of variance test
                        equal_var = stats.levene(BSL_df, post_df).pvalue > 0.05

                        # Selection of t-test
                        if normality_BSL > 0.05 and normality_post > 0.05:
                            if equal_var:
                                t_test = stats.ttest_rel(BSL_df, post_df)
                            else:
                                t_test = stats.ttest_ind(BSL_df, post_df, equal_var=False)
                        else:
                            t_test = stats.wilcoxon(BSL_df, post_df)

                        p_values.append(t_test.pvalue)
                        test_results.loc[hour, f'{stage}_p'] = t_test.pvalue

                p_values_adj = multipletests(p_values, alpha=0.05, method='holm')[1]
                
                for i, (stage, hour) in enumerate(itertools.product(stages, range(24))):
                    test_results.loc[hour, f'{stage}_p-adj'] = p_values_adj[i]

                # Write to Excel sheet
                test_results.to_excel(writer, sheet_name=sheet_name)

                test_results_dict[f'{condition}_{sheet_name}'] = test_results

        print(f'{condition}_data compile  <Done>')

    return per_12h_results, per_1h_results, test_results_dict




def plot_sleep_stage(condition, stage, results, test_results):
    
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    plt.suptitle(stage)

    y_limit = (0, 60) if stage in ['W', 'NR'] else (0, 8)
    y_ticks = (0, 30, 60) if stage in ['W', 'NR'] else (0, 4, 8)
    colors = {'W': (1, 0.5882, 0.5882), 'NR': (0.5333, 0.7412, 0.9882), 'R': (0.4392, 0.6784, 0.2784)}

    # BSL - Post1
    axs[0].errorbar(per_1h_results[condition]['per_1h_mean_BSL'].index + 0.5, per_1h_results[condition]['per_1h_mean_BSL'][stage], 
                    yerr=per_1h_results[condition]['per_1h_sem_BSL'][stage], fmt='-o', label='BSL', 
                    color='gray', capsize=8, linewidth=5, capthick=4, markersize=8)    
    axs[0].errorbar(per_1h_results[condition]['per_1h_mean_post1'].index + 0.5, per_1h_results[condition]['per_1h_mean_post1'][stage], 
                    yerr=per_1h_results[condition]['per_1h_sem_post1'][stage], fmt='-o', label='Post(0-24h)', 
                    color=colors[stage], capsize=8, linewidth=5, capthick=4, markersize=8)
    axs[0].axvspan(0, 12, color='gray', alpha=0.2)
    axs[0].set_xlim(0, 24)
    axs[0].set_xticks([0, 6, 12, 18, 24])
    axs[0].set_xlabel('Time (h)')
    axs[0].set_ylim(y_limit)
    axs[0].set_yticks(y_ticks)
    axs[0].set_ylabel('Duration (min)')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].legend(loc='upper left', bbox_to_anchor=(0.55, 1.05), fontsize=20, frameon=False)
    BSL_test_result = test_results[0][f'{stage}_p-adj']
    significant_hours = test_results[0][BSL_test_result < 0.05].index
    for hour in significant_hours:
        axs[0].plot(hour+0.5, y_limit[1]*0.98, marker='s', color=colors[stage], markersize=8, alpha=0.5)


    # BSL - Post2
    axs[1].errorbar(per_1h_results[condition]['per_1h_mean_BSL'].index + 0.5, per_1h_results[condition]['per_1h_mean_BSL'][stage], 
                    yerr=per_1h_results[condition]['per_1h_sem_BSL'][stage], fmt='-o', label='BSL', 
                    color='gray', capsize=8, linewidth=5, capthick=4, markersize=8)
    axs[1].errorbar(per_1h_results[condition]['per_1h_mean_post2'].index + 0.5, per_1h_results[condition]['per_1h_mean_post2'][stage], 
                    yerr=per_1h_results[condition]['per_1h_sem_post2'][stage], fmt='-o', label='Post(24-48h)', 
                    color=colors[stage], capsize=8, linewidth=5, alpha=0.5, capthick=4, markersize=8)
    axs[1].axvspan(0, 12, color='gray', alpha=0.2)
    axs[1].set_xlim(0, 24)
    axs[1].set_xticks([0, 6, 12, 18, 24])
    axs[1].set_xlabel('Time (h)')
    axs[1].set_ylim(y_limit)
    axs[1].set_yticks(y_ticks)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].legend(loc='upper left', bbox_to_anchor=(0.55, 1.05), fontsize=20, frameon=False)

    post_test_result = test_results[1][f'{stage}_p-adj']
    significant_hours = test_results[1][post_test_result < 0.05].index
    for hour in significant_hours:
        axs[1].plot(hour + 0.5, y_limit[1]*0.98, marker='s', color=colors[stage], markersize=8, alpha=0.5)    

    return fig



def per_12h_plot(per_12h_results, condition, stage):

    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(figsize=(4, 6))

    y_limit = (0, 100) if stage in ['W', 'NR'] else (0, 10)
    y_ticks = (0, 50, 100) if stage in ['W', 'NR'] else (0, 5, 10)
    colors = {'W': (1, 0.5882, 0.5882), 'NR': (0.5333, 0.7412, 0.9882), 'R': (0.4392, 0.6784, 0.2784)}

    per_12h_result = per_12h_results[condition]
    
    time_labels = ['BSL', '0–12h', '24–36h']
    x = np.arange(len(time_labels))

    bsl_vals = per_12h_result['per_12h_BSL'][stage]
    post1_vals = per_12h_result['per_12h_post1'][stage]
    post2_vals = per_12h_result['per_12h_post2'][stage]

    means = [np.mean(bsl_vals), np.mean(post1_vals), np.mean(post2_vals)]
    sems = [np.std(bsl_vals)/np.sqrt(len(bsl_vals)),
            np.std(post1_vals)/np.sqrt(len(post1_vals)),
            np.std(post2_vals)/np.sqrt(len(post2_vals))]

    bar_colors = ['gray', colors[stage], colors[stage]]
    alphas = [1, 1, 0.5]

    for i in range(3):
        axs.bar(x[i], means[i], yerr=sems[i], color=bar_colors[i], alpha=alphas[i], capsize=5, width=0.6)

    for i, values in enumerate([bsl_vals, post1_vals, post2_vals]):
        jitter = (np.random.rand(len(values)) - 0.5) * 0.2
        axs.scatter(np.full_like(values, x[i]) + jitter, values, color='black', s=15, alpha=0.7)

    comparisons = [
        (0, 1, bsl_vals, post1_vals),
        (1, 2, post1_vals, post2_vals),
        (0, 2, bsl_vals, post2_vals)
    ]

    p_values = []
    for i, j, data1, data2 in comparisons:
        stat, p = stats.ttest_rel(data1, data2, nan_policy='omit')
        p_adj = min(p * 3, 1.0)  # Bonferroni補正（3比較）
        p_values.append((i, j, p_adj))

    # p値注釈表示
    max_y = y_limit[1] * 0.8 
    step = y_limit[1] * 0.075
    for idx, (i, j, p) in enumerate(p_values):
        if p < 0.05:
            y = max_y + idx * step
            axs.plot([x[i], x[j]], [y, y], color='black')

            if p >= 0.001:
                axs.text((x[i] + x[j]) / 2, y + step / 10, f"{p:.3f}", ha='center', va='bottom', fontsize=14)
            else:
                axs.text((x[i] + x[j]) / 2, y + step / 10, "<0.001", ha='center', va='bottom', fontsize=14)


    # 軸設定
    axs.set_xticks(x)
    axs.set_xticklabels(time_labels, fontsize=16, rotation=45)
    axs.set_ylim(y_limit)
    axs.set_yticks(y_ticks)
    axs.set_ylabel('Time spent (%)')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    plt.tight_layout()
        
    return fig


# Execution
per_12h_results, per_1h_results, test_results_dict = total_amount_df(conditions, stages, base_path, output_path)

for stage in stages:
    for condition in conditions:

        test_results = []
        sheet_names = ['Post(0-24h)', 'Post(24-48h)']

        for sheet_name in sheet_names:

            key = f'{condition}_{sheet_name}'

            if key in test_results_dict:

                test_results.append(test_results_dict[key])

        per_1h_fig = plot_sleep_stage(condition, stage, per_1h_results, test_results)

        per_12h_fig = per_12h_plot(per_12h_results, condition, stage)

        # save
        per_1h_fig.savefig(f'{output_path}/{condition}/{condition}_{stage}.tif', format='tif', dpi=350)
        per_12h_fig.savefig(f'{output_path}/{condition}/{condition}_{stage}_12h.tif', format='tif', dpi=350)

        print(f'{condition}_{stage}_figure  <Done>')
