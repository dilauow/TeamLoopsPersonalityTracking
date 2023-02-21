from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Define the maximum length of the padded sequence
from TestCodeChunks.preprocessing import TextPreprocessor


# Define the input strings
texts = ["I often forget to complete tasks and I'm not very good at taking on tasks that require a lot of effort.",
         "I don't do well when it comes to deadlines and often miss them.",
         "I'm not very organized and I often lose important documents.",
         "I'm not always reliable and don't always follow through with my commitments."]

Tr = TextPreprocessor()
textArray = Tr.feedPreprocessorAnArray(texts)

print(textArray)

# Initialize a tokenizer with a maximum vocabulary size of 1000
tokenizer = Tokenizer(num_words=1000)

# Fit the tokenizer on the input strings
tokenizer.fit_on_texts(texts)

# Convert the input strings to sequences of integer indices
sequences = tokenizer.texts_to_sequences(textArray)

max_len = max([len(seq) for seq in sequences])

# Pad the sequences to a fixed length
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Print the padded sequences
print(padded_sequences)
