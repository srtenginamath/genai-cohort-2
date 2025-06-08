import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hello, I am Sharanayya"
tokens = enc.encode(text)

print("Tokens:", tokens)

tokens = [13225, 11, 357, 939, 1955, 10904, 139688]
decoded = enc.decode(tokens)

print("Decoded Text:", decoded)
