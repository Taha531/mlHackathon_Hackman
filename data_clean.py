# In this code we are Cleaning the Data.
# 1) Lowercasing all words.
# 2) Removing Duplicates.
# 3) Grouping all the words according to their length.
# We will first be reading the file, process the data, and then save it to a new file. 

# This function takes the file paths of input and output file.
def clean_corpus(input_file, output_file): 
    with open(input_file, 'r') as f:           # Opening the file.
        words = f.readlines()                   

    cleaned_words = []                          # Creating the list for words.
    for w in words:
        w = w.strip().lower()                   # Lowercase all words.
        if w.isalpha():                         # Keep only alphabetic words.
            cleaned_words.append(w)     

    cleaned_words = list(set(cleaned_words))    # Using set to make sure all words are unique.
    cleaned_words.sort()

    with open(output_file, 'w') as f:
        for word in cleaned_words:
            f.write(word + '\n')                # Writing the cleaned data into the new file.

    print(f"[+] Cleaned corpus written to {output_file}")
    print(f"[+] Total words after cleaning: {len(cleaned_words)}")

    grouped = {}                                # Grouping words according to their length.
    for word in cleaned_words:
        length = len(word)
        if length not in grouped:
            grouped[length] = []
        grouped[length].append(word)

    for length, words in grouped.items():       # Saving each group into a separate file.  
        filename = f'len{length}.txt'
        with open(filename, 'w') as f:
            for w in words:
                f.write(w + '\n')
        print(f"[+] Saved {len(words)} words to {filename}")
                                                
    print("\nWord Length Summary:")             # Printing a summary table.
    for length in sorted(grouped.keys()):
        print(f"Length {length}: {len(grouped[length])} words")


input_file = "Data/corpus.txt"                 # Input File path.
output_file = "cleaned_corpus.txt"             # Output File path.
clean_corpus(input_file, output_file)
