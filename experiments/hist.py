import os
import numpy as np
import matplotlib.pyplot as plt

Frobenius = True

def read_and_merge_dictionaries(file_paths):
    all_data = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Process the lines in chunks of 9 (one dictionary per 9 lines)
            for i in range(0, len(lines), 9):
                dict_lines = lines[i:i+9]
                dictionary = {}

                for line in dict_lines:
                    # Strip whitespace and split by comma
                    key_value = line.strip().split(',')
                    
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        
                        # Check if the value is a number and convert it
                        if value.isdigit():
                            value = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            value = float(value)
                        
                        dictionary[key] = value

                # Add the dictionary to the all_data list
                all_data.append(dictionary)

        except FileNotFoundError:
            print(f"File at {file_path} not found.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    return all_data

# Example usage:
file_names = ['Spreadsheet.txt', 'Spreadsheet3.txt', 'Spreadsheet4.txt']


parsed_data = read_and_merge_dictionaries(file_names)
print(len(parsed_data))

def plot_multiple_log_histograms(data_sets):
    num_histograms = len(data_sets)
    
    # Find the maximum value across all datasets to adjust x-axis range
    all_values = np.concatenate(data_sets)
    max_value = np.max(all_values)
    upper_limit = 10 ** np.ceil(np.log10(max_value))  # Find the next power of 10 greater than max_value
    
    # Create a figure with multiple subplots
    fig_height = max(6, num_histograms * 4)  # Ensure a minimum height for clarity
    fig, axes = plt.subplots(num_histograms, 1, figsize=(8, fig_height))

    
    # If there is only one histogram, axes will not be a list, so make it iterable
    if num_histograms == 1:
        axes = [axes]
    
    # Loop through each dataset and create a histogram
    for i, data in enumerate(data_sets):
        # Create histogram for the current data
        axes[i].hist(data, bins=np.logspace(np.log10(np.min(data)), np.log10(upper_limit), 50), color='skyblue', edgecolor='black', alpha=0.7, density=True)
        
        # Set x-axis to logarithmic scale
        axes[i].set_xscale('log')
        
        # Set labels and title for each subplot
        axes[i].set_xlabel('The condition number (Log Scale)')
        axes[i].set_ylabel('Density')
        if Frobenius:
            if i == 0:
                axes[i].set_title('Distribution of the Initial Frobenius Condition Numbers (Log Scale)')
            elif i == 1:
                axes[i].set_title('Distribution of the Final Frobenius Condition Numbers (Log Scale)')
        else:
            if i == 0:
                axes[i].set_title('Distribution of the Initial Euclidean Condition Numbers (Log Scale)')
            elif i == 1:
                axes[i].set_title('Distribution of the Final Euclidean Condition Numbers (Log Scale)')
        
        # Add grid
        axes[i].grid(True, which="both", ls="--", linewidth=0.5)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Example usage:
if Frobenius:
    datasets = [
        [ float(entry["Initial Frobenius Condition Number"]) for entry in parsed_data ],
        [ float(entry["Final Frobenius Condition Number"]) for entry in parsed_data ]
    ]
else:
    datasets = [
        [ float(entry["Initial Euclidean Condition Number"]) for entry in parsed_data ],
        [ float(entry["Final Euclidean Condition Number"]) for entry in parsed_data ]
    ]

for entry in parsed_data:
    if float(entry["Initial Frobenius Condition Number"]) > 10**12:
        print(entry)


plot_multiple_log_histograms(datasets)

def plot_ratio_distribution_log_scale(original_values, optimized_values):
    # Ensure the lists have the same length
    if len(original_values) != len(optimized_values):
        raise ValueError("Original and optimized values must have the same number of entries.")
    
    # Calculate the ratio of original values to optimized values
    ratios = np.array(original_values) / np.array(optimized_values)
    
    # Quick check of the distribution of ratios
    print(f"Ratios: {ratios}")
    print(f"Min ratio: {np.min(ratios)}")
    print(f"Max ratio: {np.max(ratios)}")
    print(f"Mean ratio: {np.mean(ratios)}")
    
    # Create the histogram of the ratios with more bins
    plt.figure(figsize=(8, 6))
    plt.hist(ratios, bins=np.logspace(np.log10(np.min(ratios)), np.log10(np.max(ratios)), 50), color='skyblue', edgecolor='black', alpha=0.7, density=True)

    # Set the x-axis to logarithmic scale
    plt.xscale('log')

    # Labeling
    if Frobenius:
        plt.xlabel('Initial / Final Frobenius Condition Number Ratio (Log Scale)')
        plt.ylabel('Density')
        plt.title('Distribution of Initial to Final Frobenius Condition Number Ratios (Log Scale)')
    else:
        plt.xlabel('Initial / Final Euclidean Condition Number Ratio (Log Scale)')
        plt.ylabel('Density')
        plt.title('Distribution of Initial to Final Euclidean Condition Number Ratios (Log Scale)')


    # Display grid
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Show the plot
    plt.show()


# Example usage:
original_values = datasets[0]
optimized_values = datasets[1]


plot_ratio_distribution_log_scale(original_values, optimized_values)




