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
        w = w.strip().lower()                   # Lowercase all words
        if w.isalpha():
            cleaned_words.append(w)     

    cleaned_words = list(set(cleaned_words))    # Using set to make sure all words are unique.
    cleaned_words.sort()

    with open(output_file, 'w') as f:
        for word in cleaned_words:
            f.write(word + '\n')                # Writing the cleaned data into the new file

    print(f"[+] Cleaned corpus written to {output_file}")
    print(f"[+] Total words after cleaning: {len(cleaned_words)}")


input_file = "Data/corpus.txt"                       # Input File path.
output_file = "cleaned_corpus.txt"              # Output File path
clean_corpus(input_file, output_file)   
