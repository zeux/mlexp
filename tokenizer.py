vocab_size = 2048

vocab = []

vocab.append('') # UNK

for ch in range(32, 128):
    vocab.append(chr(ch))

for ch in ['\t', '\n']:
    vocab.append(ch)

# hopefully this helps with indentation; ideally we'd renormalize that on load but ah well
for ind in ['  ', '    ']:
    vocab.append(ind)

# c++ specific
for op in ['->', '::', '*/', '//', '/*', '==', '&&', '<<', '!=', '||', '+=', '++', '>=', '|=', '>>', '<=', '--', '-=', '*=', '/=', '&=', '...', '%=']:
    vocab.append(op)

with open('tokens.txt', 'r') as f:
    for w in f:
        if len(vocab) < vocab_size:
            vocab.append(w.strip())

# build a trie for faster lookup
vocab_trie = {}

for i, w in enumerate(vocab):
    node = vocab_trie
    for ch in w:
        node.setdefault(ch, {})
        node = node[ch]
    node['index'] = i

def encode(s):
    result = []

    i = 0
    while i < len(s):
        node = vocab_trie.get(s[i])
        if node is None:
            result.append(0) # unk
            i += 1 # skip
        else:
            token = node['index']
            j = i + 1
            while j < len(s) and node:
                if 'index' in node:
                    token = node['index']
                node = node.get(s[j])
                j += 1

            assert(token)
            result.append(token)
            i += len(vocab[token])

    return result

def decode(l):
    return ''.join([vocab[i] for i in l])

if __name__ == "__main__":
    import random
    import re
    import sys

    path = sys.argv[1]

    freq = {}
    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines[:100000]:
            line = line.strip()

            for word in re.findall(r'[^a-zA-Z0-9 ]+', line):
                if len(word) == 1:
                    continue

                if word in freq:
                    freq[word] += 1
                else:
                    freq[word] = 1

    words = sorted(freq.items(), key = lambda p: p[1], reverse = True)

    # we'll use a prefix of this, so only some words will stay
    for w, c in words[:vocab_size]:
        print(w)