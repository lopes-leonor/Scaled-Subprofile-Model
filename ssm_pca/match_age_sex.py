
import pandas as pd

def match_age_sex_range(df_patients, df_controls, age_range=5, age_col='age', sex_col='sex'):

    matched_controls = []
    # Convert age columns to numeric type
    df_patients[age_col] = df_patients[age_col].astype(int)
    df_controls[age_col] = df_controls[age_col].astype(int)

    # Create a copy of df_controls to keep track of available controls
    available_controls = df_controls.copy()

    available_controls = available_controls.reset_index()
    df_patients = df_patients.reset_index()

    # Iterate through each PD patient
    for index, patient in df_patients.iterrows():

        # Define the age range for matching
        min_age = patient[age_col] - age_range
        max_age = patient[age_col] + age_range

        try:
            # Find controls with matching sex and age within the defined range
            matched_control = available_controls[(available_controls[sex_col] == patient[sex_col]) &
                                          (available_controls[age_col] >= min_age) &
                                          (available_controls[age_col] <= max_age)].sample(n=1, random_state=40) #42 not bad

            # If a matching control is found, add it to the list
            if not matched_control.empty:
                matched_controls.append(matched_control)
                # Remove the matched control from the available_controls DataFrame
                available_controls = available_controls.drop(matched_control.index)

        except ValueError as e:
            print(e)

    # Concatenate the matched controls into a single DataFrame
    matched_controls_df = pd.concat(matched_controls)
    return matched_controls_df