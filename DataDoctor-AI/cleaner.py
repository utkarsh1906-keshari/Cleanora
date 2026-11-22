
import pandas as pd


def clean_dataset(df, fill_missing, remove_duplicates):

    # Create a copy so original data remains unchanged
    cleaned_df = df.copy()

    # FEATURE 1: HANDLE MISSING VALUES
    if fill_missing:

        for column in cleaned_df.columns:

            if cleaned_df[column].isnull().sum() > 0:

                # Numeric columns: fill with median
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):

                    median_value = cleaned_df[column].median()

                    cleaned_df[column] = cleaned_df[column].fillna(
                        median_value
                    )

                # Categorical columns: fill with mode
                else:

                    mode_value = cleaned_df[column].mode()

                    if not mode_value.empty:

                        cleaned_df[column] = cleaned_df[column].fillna(
                            mode_value[0]
                        )

    # FEATURE 2: REMOVE DUPLICATES
    if remove_duplicates:

        cleaned_df = cleaned_df.drop_duplicates()

    return cleaned_df
