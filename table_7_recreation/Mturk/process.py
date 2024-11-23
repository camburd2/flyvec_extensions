import csv

input_file = '/mnt/c/Users/kobeh/OneDrive/eecs445/Project_2/flyvec/flyvec/table_7_recreation/Mturk/mturk-771.txt'  # Replace with your file path
output_file = '/mnt/c/Users/kobeh/OneDrive/eecs445/Project_2/flyvec/flyvec/table_7_recreation/Mturk/mturk-771_processed.txt'  # Specify the output file

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Skip the header
    next(reader, None)

    # Process rows
    for row in reader:
        # Write only word1, word2, and similarity columns
        writer.writerow(row[1:])