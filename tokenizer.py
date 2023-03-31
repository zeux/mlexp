vocab_size = 128

def encode(s):
    return [min(ord(c), 127) for c in s]

def decode(l):
    return ''.join([chr(i) for i in l])