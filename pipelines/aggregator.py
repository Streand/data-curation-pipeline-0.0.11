def aggregate_results(*results):
    aggregated_data = {}
    
    for result in results:
        for key, value in result.items():
            if key not in aggregated_data:
                aggregated_data[key] = []
            aggregated_data[key].append(value)
    
    return aggregated_data

def summarize_metadata(aggregated_data):
    summary = {}
    
    for key, values in aggregated_data.items():
        summary[key] = {
            'count': len(values),
            'examples': values[:5]  # Show first 5 examples
        }
    
    return summary

def export_aggregated_data(aggregated_data, file_path):
    import json
    
    with open(file_path, 'w') as f:
        json.dump(aggregated_data, f, indent=4)