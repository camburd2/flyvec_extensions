import pandas as pd

input_file = '/mnt/c/Users/kobeh/OneDrive/eecs445/Project_2/flyvec/flyvec/table_7_recreation/SimLex-999/SimLex-999.txt'  # Replace with your file path
output_file = '/mnt/c/Users/kobeh/OneDrive/eecs445/Project_2/flyvec/flyvec/table_7_recreation/SimLex-999/SimLex-999_processed.txt'  # Specify the output file

# Read TSV file, specifying only the columns we want
df = pd.read_csv(input_file, delimiter='\t', usecols=['word1', 'word2', 'SimLex999'])

# Save to a new CSV file without the index
df.to_csv(output_file, index=False)