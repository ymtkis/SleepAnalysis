import os
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Load
base_path = '/mnt/d/Q project'
if not os.path.exists(base_path):
    base_path = '/mnt/e/Q project'
conditions = {'QIH(48h)': 12}
# 'QIH(24h)': 10, 'QIH(48h)': 12, 'QIH(72h)': 14, 'Ctrl_Stim(24h)': 10, 'SD': 9 


# Calculation of transition probability
def calculate_transition_probabilities(data):
    transitions = {'W→W': 0, 'W→NR': 0, 'W→R': 0,
                   'NR→W': 0, 'NR→NR': 0, 'NR→R': 0,
                   'R→W': 0, 'R→R': 0, 'R→NR': 0}

    total_transitions = {'W': 0, 'NR': 0, 'R': 0}

    for i in range(len(data) - 1):
        current_state = data.iloc[i]  
        next_state = data.iloc[i + 1] 
        transition = f"{current_state}→{next_state}"
        transitions[transition] += 1
        total_transitions[current_state] += 1

    transition_probabilities = {}
    for transition, count in transitions.items():
        start_state = transition.split('→')[0]
        if total_transitions[start_state] == 0:
            transition_probabilities[transition] = 0
        else:
            transition_probabilities[transition] = count / total_transitions[start_state]

    return transition_probabilities


# Statistics
def statistical_tests(sheets):
    results = {}
    for transition in sheets[next(iter(sheets))].columns:
        bsl_d = sheets['BSL_D'][transition]
        post_d = sheets['Post_D'][transition]
        bsl_l = sheets['BSL_L'][transition]
        post_l = sheets['Post_L'][transition]

        # Normality test
        normality_bsl_d = stats.shapiro(bsl_d).pvalue
        normality_post_d = stats.shapiro(post_d).pvalue
        normality_bsl_l = stats.shapiro(bsl_l).pvalue
        normality_post_l = stats.shapiro(post_l).pvalue

        # Homogeneity of variance test
        equal_var_d = stats.levene(bsl_d, post_d).pvalue > 0.05
        equal_var_l = stats.levene(bsl_l, post_l).pvalue > 0.05

        # Select appropriate t test
        if normality_bsl_d > 0.05 and normality_post_d > 0.05 and equal_var_d:
            t_test_d = stats.ttest_ind(bsl_d, post_d, equal_var=True)
        else:
            t_test_d = stats.mannwhitneyu(bsl_d, post_d)

        if normality_bsl_l > 0.05 and normality_post_l > 0.05 and equal_var_l:
            t_test_l = stats.ttest_ind(bsl_l, post_l, equal_var=True)
        else:
            t_test_l = stats.mannwhitneyu(bsl_l, post_l)

        results[transition] = {
            'BSL_D_vs_Post_D': t_test_d.pvalue,
            'BSL_L_vs_Post_L': t_test_l.pvalue
        }

    return results


def plot_significance(ax, x1, x2, transition, p_value):

    if transition in ['W→W', 'NR→NR','R→R']:
        y = 1.05
        height = 0.02
    else:
        y = 0.20
        height = 0.004

    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.5, c='black')
    sign = ''
    if p_value < 0.001:
        sign = '***'
    elif p_value < 0.01:
        sign = '**'
    elif p_value < 0.05:
        sign = '*'
    elif p_value > 0.05:
        sign = 'ns'
    if sign:
        ax.text((x1 + x2) * 0.5, y + height, sign, ha='center', va='bottom', color='black')



# Visualization
def plot_transition_probabilities(sheets, test_results, condition):

    sheet_order = ['BSL_D', 'Post_D', 'BSL_L', 'Post_L']

    bar_width = 0.4  
    font_size = 20  
    title_font_size = 20
    tick_font_size = 20
    error_linewidth = 2

    colors = {'BSL_L': 'lightyellow', 'Post_L': 'lightyellow',
              'BSL_D': 'gray', 'Post_D': 'gray'}
    scatter_colors = {'BSL_L': 'yellow', 'Post_L': 'yellow',
                      'BSL_D': 'darkgray', 'Post_D': 'darkgray'}
    bar_edge_color = 'black' 
    scatter_edge_color = 'black' 
    scatter_linewidth = 1  
    jitter = 0.06

    
    for transition in sheets[next(iter(sheets))].columns:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        x_labels = []
        for i, sheet_name in enumerate(sheet_order):
            df = sheets[sheet_name]
            mean = df[transition].mean()
            sem = df[transition].sem()
            bar_color = colors[sheet_name]
            scatter_color = scatter_colors[sheet_name]

            plt.bar(i, mean, yerr=sem, capsize=5, color=bar_color, edgecolor=bar_edge_color,
                    width=bar_width, error_kw={'linewidth': error_linewidth, 'capthick': error_linewidth})
            jitters = np.random.uniform(-jitter, jitter, len(df.index))
            plt.scatter([i + jitter for jitter in jitters], df[transition], alpha=0.7, c=scatter_color,
                        edgecolors=scatter_edge_color, linewidth=scatter_linewidth)

            x_labels.append(sheet_name)

        plot_significance(ax, 0, 1, transition, test_results[transition]['BSL_D_vs_Post_D'])
        plot_significance(ax, 2, 3, transition, test_results[transition]['BSL_L_vs_Post_L'])

        plt.xticks(range(len(sheets)), x_labels, fontsize=font_size)
        if transition in ['W→W', 'NR→NR', 'R→R']:
            ticks = np.arange(0, 1.15, 0.25)
            ax.set_ylim(0, 1.15) 
            ax.axhline(y=1, color='gray', linestyle='--')
        else:
            ticks = np.arange(0, 0.25, 0.05)
            ax.set_ylim(0, 0.21)
 
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2g}" for tick in ticks], fontsize=tick_font_size)
        plt.ylabel('Probability', fontsize=font_size)
        plt.title(transition, fontsize=title_font_size)


        plt.tight_layout()
        file_name = f"{transition.replace('→', 'to')}.tif"
        plt.savefig(f'/mnt/e/Q project/SleepAnalysis/Transition/{condition}/{file_name}', format='tif')
        plt.close()

    print(f'{condition}_figure  <Done>')



for condition, div_num in conditions.items():
    staging_file = f'{base_path}/EEGEMG/Compile/{condition}/{condition}_staging.xlsx'        
    output_path = f'{base_path}/SleepAnalysis/transition/{condition}/{condition}_transition.xlsx'
    output_test_path = f'{base_path}/SleepAnalysis/transition/{condition}/{condition}_transition_test.xlsx'
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
    else:
        subsets = {
            'BSL_D': staging_data[:division_size],
            'BSL_L': staging_data[division_size:2 * division_size],
            'Post_D': staging_data[(div_num - 4) * division_size:(div_num - 3) * division_size],
            'Post_L': staging_data[(div_num - 3) * division_size:(div_num - 2) * division_size]
        }   

    probabilities_per_timeseg = {timeseg: {} for timeseg in subsets.keys()}
    for timeseg, subset in subsets.items():
        subset.columns = range(subset.shape[1])
        for column in subset.columns:
            probabilities_col = calculate_transition_probabilities(subset[column].dropna().reset_index(drop=True))
            for transition, prob in probabilities_col.items():
                probabilities_per_timeseg[timeseg].setdefault(transition, []).append(prob)

    # Transform to dataframe - Output
    sheets = {}
    for timeseg, probabilities in probabilities_per_timeseg.items():
        df_prob = pd.DataFrame(probabilities, index=[f'Individual {i + 1}' for i in range(len(subset.columns))])
        sheets[timeseg] = df_prob

    test_results = statistical_tests(sheets)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for time_seg, df in sheets.items():
            df.to_excel(writer, sheet_name=time_seg, index=True)
    print(f'{condition}_data compile  <Done>')

    plot_transition_probabilities(sheets, test_results, condition)