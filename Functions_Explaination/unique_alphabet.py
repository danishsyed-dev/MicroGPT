# 1. Your input data (docs)
docs = ["cat", "dog"]

# 2. THE LINE: Get every unique character, sorted
# - ''.join(docs) merges them into "catdog"
# - set() removes duplicates (none here, but if it was "cattt", it would just keep one 't')
# - sorted() puts them in order: ['a', 'c', 'd', 'g', 'o', 't']
uchars = sorted(set(''.join(docs)))

print(f"Unique alphabet: {uchars}")

# 3. Create a Mapping (Dictionary)
# This is how the GPT "reads" letters as numbers
char_to_int = { char:i for i, char in enumerate(uchars) }

print(f"Character to integer mapping: {char_to_int}")


# 4. Testing the 'GPT Brain':
word = "cat"
encoded = [char_to_int[c] for c in word]
print(f"The word '{word}' as numbers: {encoded}")