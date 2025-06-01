def export_results(results, output_format='json', output_path='output/results'):
    import json
    import pandas as pd

    if output_format == 'json':
        with open(f"{output_path}.json", 'w') as json_file:
            json.dump(results, json_file, indent=4)
    elif output_format == 'csv':
        df = pd.DataFrame(results)
        df.to_csv(f"{output_path}.csv", index=False)
    else:
        raise ValueError("Unsupported output format. Use 'json' or 'csv'.")

def filter_results(results, criteria):
    filtered_results = [result for result in results if meets_criteria(result, criteria)]
    return filtered_results

def meets_criteria(result, criteria):
    # Implement your filtering logic based on criteria
    return True  # Placeholder for actual filtering logic