import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Callable, Optional
from canon import standardize_country_name


def process_geoguessr_results(
    base_dir: str = "responses",
    stats_function: Optional[Callable] = None,
    group_by: str = "country_true"
) -> Dict[str, Dict[str, Any]]:
    """
    Process GeoGuessr benchmark results to calculate statistics by country for each model.
    
    Args:
        base_dir: Base directory containing response folders
        stats_function: Function to calculate statistics, defaults to mean score
        group_by: Column to group by, defaults to 'country_true'
        
    Returns:
        Dictionary containing model statistics and counts
    """
    # Initialize data structures to collect all results
    all_scores = defaultdict(lambda: defaultdict(list))
    all_counts = defaultdict(lambda: defaultdict(int))
    
    # Walk through directories
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        detailed_csv = os.path.join(folder_path, "results", "detailed.csv")
        summary_json = os.path.join(folder_path, "results", "summary.json")
        
        # Check if both files exist
        if not (os.path.isfile(detailed_csv) and os.path.isfile(summary_json)):
            continue
        
        # Read the model name from summary.json
        try:
            with open(summary_json, 'r') as f:
                summary_data = json.load(f)
                model_name = summary_data.get("model", folder)
        except (json.JSONDecodeError, FileNotFoundError):
            model_name = folder
        
        # Read and process the detailed results
        try:
            df = pd.read_csv(detailed_csv)
            
            # Standardize country names to group territories with their parent countries
            df[group_by] = df[group_by].apply(standardize_country_name)
            
            # Group by country and collect scores
            for country, country_df in df.groupby(group_by):
                # Add scores to the collection for this model and country
                all_scores[model_name][country].extend(country_df['score'].tolist())
                # Update count for this model and country
                all_counts[model_name][country] += len(country_df)
                
        except Exception as e:
            print(f"Error processing {detailed_csv}: {e}")
    
    # Calculate statistics from accumulated data
    results = {
        'stats': {},
        'counts': {}
    }
    
    # Calculate mean scores from all collected data
    for model in all_scores:
        results['stats'][model] = {}
        results['counts'][model] = {}
        
        for country in all_scores[model]:
            if stats_function:
                # Use custom stats function if provided
                country_df = pd.DataFrame({'score': all_scores[model][country]})
                results['stats'][model][country] = stats_function(country_df)
            else:
                # Default: calculate mean
                results['stats'][model][country] = np.mean(all_scores[model][country])
            
            results['counts'][model][country] = all_counts[model][country]
    
    return results


def format_results_as_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the results dictionary to a DataFrame for easier viewing.
    
    Args:
        results: Dictionary mapping model names to country statistics
        
    Returns:
        DataFrame with countries as rows and models as columns
    """
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    
    # Sort the DataFrame for better readability
    df = df.sort_index()
    
    return df


def main():
    # Process results to get average scores by country for each model
    results = process_geoguessr_results()
    
    # Convert average scores to DataFrame and display
    avg_scores_df = format_results_as_dataframe(results['stats'])
    print("Average Country Scores by Model:")
    print(avg_scores_df)
    avg_scores_df.to_csv("country_avg_scores_by_model.csv")
    
    # Convert counts to DataFrame and display
    counts_df = format_results_as_dataframe(results['counts'])
    print("\nSample Counts by Country and Model:")
    print(counts_df)
    counts_df.to_csv("country_sample_counts_by_model.csv")
    
    # Create a combined dataframe with format: avg (n=count)
    combined_df = pd.DataFrame(index=avg_scores_df.index, columns=avg_scores_df.columns)
    
    for country in combined_df.index:
        for model in combined_df.columns:
            # Properly handle missing values with pandas indexing
            try:
                avg = avg_scores_df.loc[country, model]
                count = counts_df.loc[country, model]
                if pd.notna(avg):
                    combined_df.loc[country, model] = f"{avg:.1f} (n={count})"
            except (KeyError, ValueError):
                combined_df.loc[country, model] = None
    
    combined_df.to_csv("country_scores_with_counts_by_model.csv")
    print("\nCombined Scores with Sample Counts:")
    print(combined_df)


if __name__ == "__main__":
    main()