vocab_size = 32000
hidden_size = 4096
intermediate_size = 11008
num_hidden_layers = 32


res = vocab_size * hidden_size + num_hidden_layers * (hidden_size**2 * 4 + intermediate_size*hidden_size * 3 + 2*hidden_size + hidden_size)
print(res / 1e9)