import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Define the root directory for responses
RESPONSES_DIR = Path("../../responses") # Assuming this script is in visualizations/streamlit

# @st.cache_data # Consider uncommenting for performance with large datasets
def find_summary_files(base_dir):
    """Finds all summary.json files within the subdirectories of base_dir."""
    summary_files = []
    for entry in base_dir.iterdir():
        if entry.is_dir() and entry.name not in ['.git', '.hidden', '.incomplete']:
            summary_file_path = entry / "results" / "summary.json"
            if summary_file_path.exists():
                summary_files.append(summary_file_path)
    return summary_files

# @st.cache_data # Consider uncommenting for performance
def load_summary_data(file_path):
    """Loads data from a single summary.json file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        data['experiment_name'] = file_path.parent.parent.name
        # Ensure essential keys 'model' and 'test' are present, provide defaults if not
        if 'model' not in data:
            st.warning(f"'model' key missing in {file_path}. Using experiment name as fallback.")
            data['model'] = data['experiment_name']
        if 'test' not in data:
            st.warning(f"'test' key missing in {file_path}. Setting to 'Unknown'.")
            data['test'] = "Unknown"
        return data
    except Exception as e:
        st.error(f"Error loading summary {file_path}: {e}")
        return None

# @st.cache_data # Consider uncommenting for performance
def find_detailed_csv_files(base_dir):
    """Finds all detailed.csv files."""
    detailed_files = []
    for entry in base_dir.iterdir():
        if entry.is_dir() and entry.name not in ['.git', '.hidden', '.incomplete']:
            detailed_file_path = entry / "results" / "detailed.csv"
            if detailed_file_path.exists():
                detailed_files.append(detailed_file_path)
    return detailed_files

# @st.cache_data # Consider uncommenting for performance
def load_detailed_data(file_path):
    """Loads data from a single detailed.csv file and adds folder-derived info."""
    try:
        df = pd.read_csv(file_path)
        df['experiment_name'] = file_path.parent.parent.name
        if 'location_id' not in df.columns and 'round' in df.columns:
            df.rename(columns={'round': 'location_id'}, inplace=True)
        elif 'location_id' not in df.columns:
            df['location_id'] = range(1, len(df) + 1)
        # Ensure boolean interpretation for country_correct
        if 'country_correct' in df.columns:
            df['country_correct'] = df['country_correct'].astype(bool)
        return df
    except Exception as e:
        st.error(f"Error loading detailed CSV {file_path}: {e}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("Model Results Dashboard")

    st.sidebar.header("Options")
    if st.sidebar.button("Refresh Data", key="refresh_button"):
        st.cache_data.clear() # Clear cache on refresh
        st.experimental_rerun()

    # --- 1. Load All Data --- 
    summary_files = find_summary_files(RESPONSES_DIR)
    all_summary_data_list = []
    if summary_files:
        for s_file in summary_files:
            data = load_summary_data(s_file)
            if data:
                all_summary_data_list.append(data)
    
    detailed_csv_files = find_detailed_csv_files(RESPONSES_DIR)
    all_detailed_data_list = [] # List of DFs, each from a detailed.csv
    if detailed_csv_files:
        for d_file in detailed_csv_files:
            df_detail_partial = load_detailed_data(d_file)
            if df_detail_partial is not None:
                all_detailed_data_list.append(df_detail_partial)

    # --- 2. Consolidate and Enrich DataFrames --- 
    summary_df = pd.DataFrame()
    if all_summary_data_list:
        summary_df = pd.DataFrame(all_summary_data_list)
        if 'model' in summary_df.columns: # model from summary.json is the clean name
            summary_df.rename(columns={'model': 'cleaned_model_name'}, inplace=True)
        else: # Should be handled by load_summary_data providing a fallback
            summary_df['cleaned_model_name'] = summary_df['experiment_name'] 
        if 'test' not in summary_df.columns: # Should be handled by load_summary_data
             summary_df['test'] = "Unknown"

    detailed_df_processed_list = []
    if all_detailed_data_list:
        for df_detail_partial in all_detailed_data_list:
            exp_name = df_detail_partial['experiment_name'].iloc[0]
            
            # Get corresponding cleaned_model_name and test from summary_df
            model_name_from_summary = exp_name # Fallback
            test_name_from_summary = "Unknown"  # Fallback

            if not summary_df.empty and 'experiment_name' in summary_df.columns:
                summary_row = summary_df[summary_df['experiment_name'] == exp_name]
                if not summary_row.empty:
                    model_name_from_summary = summary_row['cleaned_model_name'].iloc[0]
                    test_name_from_summary = summary_row['test'].iloc[0]
                else:
                    st.warning(f"No summary.json data found for experiment '{exp_name}' to get clean model/test names for its detailed.csv. Using fallbacks.")
            else:
                 st.warning(f"Summary data is empty or missing 'experiment_name' column. Cannot enrich detailed data for '{exp_name}'.")

            df_detail_partial['cleaned_model_name'] = model_name_from_summary
            df_detail_partial['test'] = test_name_from_summary
            detailed_df_processed_list.append(df_detail_partial)

    detailed_df = pd.DataFrame()
    if detailed_df_processed_list:
        detailed_df = pd.concat(detailed_df_processed_list, ignore_index=True)

    if summary_df.empty and detailed_df.empty:
        st.warning(f"No data loaded from {RESPONSES_DIR}. Check folder structure and file contents.")
        return

    # --- 3. Global Test Filter (Sidebar) --- 
    unique_tests = []
    if 'test' in summary_df.columns: 
        unique_tests.extend(summary_df['test'].unique())
    if 'test' in detailed_df.columns:
        unique_tests.extend(detailed_df['test'].unique())
    
    all_available_tests = sorted(list(set(t for t in unique_tests if pd.notna(t))))
    
    selected_test = "All Tests"
    if not all_available_tests:
        st.sidebar.warning("No 'test' information found in any data.")
    else:
        selected_test = st.sidebar.selectbox(
            "Filter by Test",
            options=["All Tests"] + all_available_tests,
            index=0,
            key="global_test_filter"
        )

    # Apply global test filter
    filtered_summary_df = summary_df.copy()
    filtered_detailed_df = detailed_df.copy()

    if selected_test != "All Tests":
        if 'test' in filtered_summary_df.columns:
            filtered_summary_df = filtered_summary_df[filtered_summary_df['test'] == selected_test]
        else: # if 'test' column somehow missing after load, create empty to avoid errors
            filtered_summary_df = pd.DataFrame(columns=summary_df.columns)
            
        if 'test' in filtered_detailed_df.columns:
            filtered_detailed_df = filtered_detailed_df[filtered_detailed_df['test'] == selected_test]
        else:
            filtered_detailed_df = pd.DataFrame(columns=detailed_df.columns)

    # --- 4. Display Aggregated Experiment Summaries (Main Area) ---
    # if not filtered_summary_df.empty:
    #     st.header(f"Aggregated Experiment Summaries (Test: {selected_test})")
    #     st.dataframe(filtered_summary_df)

    #     st.sidebar.subheader("Summary Plot Configuration")
    #     summary_cols = filtered_summary_df.columns.tolist()
    #     numeric_summary_cols = filtered_summary_df.select_dtypes(include=['number']).columns.tolist()

    #     default_group_by = 'cleaned_model_name' if 'cleaned_model_name' in summary_cols else (summary_cols[0] if summary_cols else None)
    #     default_metric = 'average_score' if 'average_score' in numeric_summary_cols else (numeric_summary_cols[0] if numeric_summary_cols else None)

    #     if default_group_by and default_metric and default_group_by in summary_cols and default_metric in numeric_summary_cols:
    #         group_by_col_summary = st.sidebar.selectbox(
    #             "Group Summaries by", 
    #             options=summary_cols, 
    #             index=summary_cols.index(default_group_by), 
    #             key="summary_group_by"
    #         )
    #         metric_to_plot_summary = st.sidebar.selectbox(
    #             "Metric to Plot (Summary)", 
    #             options=numeric_summary_cols, 
    #             index=numeric_summary_cols.index(default_metric),
    #             key="summary_metric"
    #         )

            # if group_by_col_summary and metric_to_plot_summary:
            #     try:
            #         plot_summary_df = filtered_summary_df.groupby(group_by_col_summary)[metric_to_plot_summary].mean().reset_index()
            #         st.subheader(f"Mean {metric_to_plot_summary} by {group_by_col_summary}")
            #         st.bar_chart(plot_summary_df.set_index(group_by_col_summary)[metric_to_plot_summary])
            #     except Exception as e:
            #         st.error(f"Could not generate summary plot: {e}")
    #     else:
    #         st.sidebar.info("Not enough data or columns in (filtered) summary to create plots.")
    # else:
    #     st.info(f"No summary data to display for Test: {selected_test}.")

    # --- 5. Compare Model Score Distributions (Main Area) ---
    st.header(f"Compare Model Score Distributions")
    if not filtered_detailed_df.empty and 'cleaned_model_name' in filtered_detailed_df.columns and 'score' in filtered_detailed_df.columns:
        available_models_for_dist = sorted(filtered_detailed_df['cleaned_model_name'].unique())
        if available_models_for_dist:
            selected_models_for_dist = st.multiselect(
                "Select Models to Compare Score Distributions", 
                options=available_models_for_dist,
                key="dist_model_select"
            )
            if selected_models_for_dist:
                plot_dist_df = filtered_detailed_df[filtered_detailed_df['cleaned_model_name'].isin(selected_models_for_dist)]
                if not plot_dist_df.empty:
                    fig_dist, ax_dist = plt.subplots()
                    sns.histplot(data=plot_dist_df, x='score', hue='cleaned_model_name', ax=ax_dist, kde=True, element="step", fill=True, alpha=0.7)
                    ax_dist.set_title(f"Score Distributions for Selected Models")
                    ax_dist.set_xlabel("Score")
                    ax_dist.set_ylabel("Density")
                    st.pyplot(fig_dist)
                else:
                    st.info("No data for selected models to plot score distribution.")
        else:
            st.info(f"No models available in detailed data for Test: '{selected_test}' to compare distributions.")
    elif not filtered_detailed_df.empty:
         st.info(f"Detailed data for Test: '{selected_test}' is missing 'cleaned_model_name' or 'score' column for distribution plot.")
    else:
        st.info(f"No detailed data to display for Test: '{selected_test}'.")

    # --- 6. Country Performance Comparison (Main Area) ---
    st.header(f"Country Performance Comparison")
    if not filtered_detailed_df.empty and all(col in filtered_detailed_df.columns for col in ['cleaned_model_name', 'country_true', 'country_correct', 'score']):
        available_models_for_country = sorted(filtered_detailed_df['cleaned_model_name'].unique())
        available_countries = sorted(filtered_detailed_df['country_true'].dropna().unique())

        if not available_models_for_country:
            st.info(f"No models found in detailed data for Test: '{selected_test}' for country comparison.")
        elif not available_countries:
            st.info(f"No country data found in detailed data for Test: '{selected_test}' for country comparison.")
        else:
            selected_models_for_country = st.multiselect(
                "Select Models for Country Comparison",
                options=available_models_for_country,
                default=available_models_for_country[:min(3, len(available_models_for_country))], # Default to first 3 or fewer
                key="country_model_select"
            )
            selected_countries_for_comp = st.multiselect(
                "Select Countries for Comparison (or type to search)",
                options=available_countries,
                default=available_countries[:min(5, len(available_countries))], # Default to first 5 or fewer
                key="country_select"
            )
            metric_for_country = st.radio(
                "Select Metric for Country Comparison",
                options=["Country Identification Rate", "Average Score by Country"],
                key="country_metric_select"
            )

            if selected_models_for_country and selected_countries_for_comp:
                country_comp_df = filtered_detailed_df[
                    (filtered_detailed_df['cleaned_model_name'].isin(selected_models_for_country)) &
                    (filtered_detailed_df['country_true'].isin(selected_countries_for_comp))
                ].copy() # Use .copy() to avoid SettingWithCopyWarning

                if not country_comp_df.empty:
                    if metric_for_country == "Country Identification Rate":
                        # Ensure 'country_correct' is boolean/numeric for mean calculation
                        country_comp_df['country_correct'] = country_comp_df['country_correct'].astype(float)
                        agg_country_df = country_comp_df.groupby(['cleaned_model_name', 'country_true'])['country_correct'].mean().reset_index()
                        agg_country_df.rename(columns={'country_correct': 'Identification Rate'}, inplace=True)
                        y_label = "Identification Rate (0.0 - 1.0)"
                    else: # Average Score by Country
                        agg_country_df = country_comp_df.groupby(['cleaned_model_name', 'country_true'])['score'].mean().reset_index()
                        agg_country_df.rename(columns={'score': 'Average Score'}, inplace=True)
                        y_label = "Average Score"
                    
                    if not agg_country_df.empty:
                        fig_country, ax_country = plt.subplots(figsize=(max(10, len(selected_countries_for_comp) * 1.5), 6))
                        sns.barplot(data=agg_country_df, x='country_true', y=agg_country_df.columns[-1], hue='cleaned_model_name', ax=ax_country)
                        ax_country.set_title(f"{metric_for_country} for Selected Models & Countries")
                        ax_country.set_xlabel("Country")
                        ax_country.set_ylabel(y_label)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig_country)

                        st.subheader("Data for Country Comparison Plot")
                        st.dataframe(agg_country_df)
                    else:
                        st.info("No aggregated data to plot for the selected country comparison criteria.")
                else:
                    st.info("No data found for the selected models and countries for comparison.")
            else:
                st.info("Please select at least one model and one country for comparison.")
    elif not filtered_detailed_df.empty:
        st.info(f"Detailed data for Test: '{selected_test}' is missing one or more required columns for country comparison: 'cleaned_model_name', 'country_true', 'country_correct', 'score'.")
    else:
        st.info(f"No detailed data available for Test: '{selected_test}' to perform country comparison.")

    # --- 7. Detailed Round-by-Round Analysis for a Single Experiment (Main Area) ---
    st.header(f"Drill Down: Single Experiment Details")
    if not filtered_detailed_df.empty:
        unique_models_in_detail = sorted(filtered_detailed_df['cleaned_model_name'].unique())
        if not unique_models_in_detail:
            st.info(f"No models found in detailed data for Test: '{selected_test}' for drilldown.")
        else:
            selected_model_for_drilldown = st.selectbox("Select Model for Drilldown", options=unique_models_in_detail, key="drill_model")
            if selected_model_for_drilldown:
                df_model_drilldown = filtered_detailed_df[filtered_detailed_df['cleaned_model_name'] == selected_model_for_drilldown]
                unique_experiments_for_model = sorted(df_model_drilldown['experiment_name'].unique())
                
                if not unique_experiments_for_model:
                    st.info(f"No specific experiment runs for model '{selected_model_for_drilldown}' in Test: '{selected_test}'.")
                else:
                    selected_experiment_for_drilldown = st.selectbox("Select Specific Experiment Run", options=unique_experiments_for_model, key="drill_exp")
                    if selected_experiment_for_drilldown:
                        single_exp_detailed_df = df_model_drilldown[df_model_drilldown['experiment_name'] == selected_experiment_for_drilldown]
                        if not single_exp_detailed_df.empty:
                            st.subheader(f"Score Distribution for Run: {selected_experiment_for_drilldown}")
                            if 'score' in single_exp_detailed_df.columns:
                                fig_hist_single, ax_hist_single = plt.subplots()
                                single_exp_detailed_df['score'].plot(kind='hist', ax=ax_hist_single, bins=20, title='Score Distribution')
                                ax_hist_single.set_xlabel("Score")
                                ax_hist_single.set_ylabel("Frequency")
                                st.pyplot(fig_hist_single)
                            else:
                                st.warning("No 'score' column for histogram.")

                            st.subheader(f"Scores per Round for Run: {selected_experiment_for_drilldown}")
                            if 'score' in single_exp_detailed_df.columns and 'location_id' in single_exp_detailed_df.columns:
                                line_plot_df = single_exp_detailed_df.sort_values(by='location_id')
                                st.line_chart(line_plot_df.set_index('location_id')['score'])
                            else:
                                st.warning("Missing 'score' or 'location_id' for scores per round plot.")

                            st.subheader(f"Detailed Data Table for Run: {selected_experiment_for_drilldown}")
                            st.dataframe(single_exp_detailed_df)
                        else:
                            st.info(f"No detailed data for experiment run: {selected_experiment_for_drilldown}")
    else:
        st.info(f"No detailed data available for Test: '{selected_test}' for drilldown analysis.")

if __name__ == "__main__":
    main() 