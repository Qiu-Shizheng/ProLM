import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pymannkendall as mk
from scipy.interpolate import make_interp_spline
import os

# Define protein list
proteins = ['pla2g7', 'tmprss15', 'gdf15', 'pon3', 'apom']

# Create output folder
os.makedirs('/path/protein_results', exist_ok=True)


# Main analysis function
def analyze_protein(protein_name):
    # 1. Load baseline information data (including follow-up time, gender, age, etc.)
    t2d = pd.read_csv('/path/information_with_followup.csv')

    # Convert date format
    t2d['p200'] = pd.to_datetime(t2d['p200'])
    t2d['p130708'] = pd.to_datetime(t2d['p130708'])

    # Exclude participants diagnosed before the baseline date
    t2d = t2d[~((t2d['p130708'].notna()) & (t2d['p130708'] <= t2d['p200']))]

    # Calculate time from diagnosis to baseline (in years)
    t2d.loc[t2d['p130708'].notna(), 'Time_to_diagnosis'] = (t2d['p200'] - t2d['p130708']).dt.days / 365.25

    # Select T2D patients (samples with a diagnosis time)
    t2d_diabetic = t2d[t2d['Time_to_diagnosis'].notna()].copy()
    t2d_diabetic['Time_to_diagnosis_year'] = t2d_diabetic['Time_to_diagnosis'].astype(int)

    # 2. Control samples
    sampled_controls = pd.read_csv('/path/healthy_control.csv')

    controls_info = sampled_controls.merge(t2d[['eid', 'age', 'sex', 'p200', 'p130708']],
                                           on='eid', how='left')
    # Also filter out samples diagnosed before baseline date
    controls_info = controls_info[
        ~((controls_info['p130708'].notna()) & (controls_info['p130708'] <= controls_info['p200']))].copy()

    # Set the number of control samples to twice the T2D sample number
    num_controls_needed = 2 * len(t2d_diabetic)
    if len(controls_info) < num_controls_needed:
        raise Exception("Insufficient control samples!")

    # Randomly select the required number of controls
    selected_controls = controls_info.sample(n=num_controls_needed, random_state=42).reset_index(drop=True)

    # 3. Construct matching table
    diabetic_repeated = t2d_diabetic[['eid', 'Time_to_diagnosis_year', 'age', 'sex']].copy().loc[
        t2d_diabetic.index.repeat(2)
    ].reset_index(drop=True)
    diabetic_repeated = diabetic_repeated.rename(columns={'eid': 'participant_id'})

    matched_pairs_df = pd.DataFrame({
        'participant_id': diabetic_repeated['participant_id'],
        'control_id': selected_controls['eid'],
        'Time_to_diagnosis_year': diabetic_repeated['Time_to_diagnosis_year'],
        'age': diabetic_repeated['age'],
        'sex': diabetic_repeated['sex']
    })

    # 4. Load protein expression data
    protein_data = pd.read_csv('/path/proteomic_data.csv')
    protein_data_sub = protein_data[['eid', protein_name]]

    # Merge protein expression
    diabetic_data = matched_pairs_df[['participant_id', 'Time_to_diagnosis_year']].merge(
        protein_data_sub, left_on='participant_id', right_on='eid', how='left'
    )
    diabetic_data['group'] = 'T2D'

    control_data = matched_pairs_df[['control_id', 'Time_to_diagnosis_year']].merge(
        protein_data_sub, left_on='control_id', right_on='eid', how='left'
    )
    control_data['group'] = 'Control'

    # Merge datasets
    combined_data = pd.concat([
        diabetic_data.rename(columns={'participant_id': 'eid'}),
        control_data.rename(columns={'control_id': 'eid'})
    ], ignore_index=True)

    combined_data = combined_data.dropna(subset=[protein_name, 'Time_to_diagnosis_year'])

    # Merge groups -14 and -15 into group -14.5
    combined_data['Time_to_diagnosis_year'] = np.where(
        combined_data['Time_to_diagnosis_year'].isin([-14, -15]),
        -14.5,
        combined_data['Time_to_diagnosis_year']
    )

    # Group by follow-up time and group, compute mean, standard deviation, sample count, and standard error
    grouped_data = combined_data.groupby(['Time_to_diagnosis_year', 'group']).agg(
        mean_protein=(protein_name, 'mean'),
        std_protein=(protein_name, 'std'),
        n=(protein_name, 'count')
    ).reset_index()
    grouped_data['sem_protein'] = grouped_data['std_protein'] / np.sqrt(grouped_data['n'])

    # Plotting
    plt.figure(figsize=(10, 6))

    colors = {'T2D': 'lightcoral', 'Control': 'skyblue'}

    for group_name, group_data in grouped_data.groupby('group'):
        group_data = group_data.sort_values('Time_to_diagnosis_year')
        x = group_data['Time_to_diagnosis_year'].values
        y = group_data['mean_protein'].values
        yerr = group_data['sem_protein'].values

        plt.errorbar(x, y, yerr=yerr, fmt='o', color=colors[group_name],
                     ecolor=colors[group_name], capsize=3, label=group_name)
        if len(x) > 3:
            x_new = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_new)
            plt.plot(x_new, y_smooth, color=colors[group_name])
        else:
            plt.plot(x, y, color=colors[group_name])

    plt.xticks(np.arange(-15, 1, 1))
    plt.xlabel('Time to diagnosis (years)')
    plt.ylabel(f'{protein_name} protein expression')
    plt.legend()

    # Mann-Kendall trend test
    diabetic_group = grouped_data[grouped_data['group'] == 'T2D'].sort_values('Time_to_diagnosis_year')
    control_group = grouped_data[grouped_data['group'] == 'Control'].sort_values('Time_to_diagnosis_year')

    diabetic_test = mk.original_test(diabetic_group['mean_protein'])
    control_test = mk.original_test(control_group['mean_protein'])

    # Regression analysis, convert group to numerical variable
    combined_data_for_regression = combined_data.copy()
    combined_data_for_regression['group_numeric'] = combined_data_for_regression['group'].map(
        {'Control': 0, 'T2D': 1})

    model = smf.ols(f'{protein_name} ~ Time_to_diagnosis_year * group_numeric', data=combined_data_for_regression).fit()
    interaction_pvalue = model.pvalues['Time_to_diagnosis_year:group_numeric']

    plt.text(0.05, 0.95, f'P = {interaction_pvalue:.4f}', transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/path/protein_results/{protein_name}_trajectory.pdf')
    plt.close()

    # Save results
    with open(f'/path/protein_results/{protein_name}_results.txt', 'w') as f:
        f.write(f"Protein: {protein_name}\n\n")
        f.write("Mann-Kendall test for T2D group:\n")
        f.write(str(diabetic_test) + "\n\n")
        f.write("Mann-Kendall test for Control group:\n")
        f.write(str(control_test) + "\n\n")
        f.write("Linear Regression Model Summary:\n")
        f.write(model.summary().as_text())

    grouped_data.to_csv(f'/path/protein_results/{protein_name}_grouped_data.csv', index=False)


for protein in proteins:
    try:
        print(f"Processing {protein}...")
        analyze_protein(protein)
        print(f"{protein} analysis completed successfully.")
    except Exception as e:
        print(f"Error processing {protein}: {e}")