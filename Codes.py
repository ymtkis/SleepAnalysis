import os
import subprocess

# Python file list
base_path = '/mnt/d/Q project/SleepAnalysis/Codes/'
if not os.path.exists(base_path):
    base_path = '/mnt/e/Q project/SleepAnalysis/Codes/'
python_files = [
    f'{base_path}Episode_duration.py',
    f'{base_path}Structure.py',
    f'{base_path}Total_amount.py',
    f'{base_path}Transition.py'
]
graphs = ['Episode duration', 'Structure', 'Total amount', 'Transition']

# Execution
for file, graph in zip(python_files, graphs):
    print(f'{graph}  <Start>')
    subprocess.run(['python', file])
    print(f'{graph}  <Done>')
