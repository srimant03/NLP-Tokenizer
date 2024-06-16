import re
from collections import defaultdict

class Tokenizer:
    def __init__(self,corpus,num_merges):
        self.vocab = defaultdict(int)
        self.corpus = corpus
        self.num_merges = num_merges
        self.split_pattern = re.compile(r'(\W+)')
        self.learnt_vocab, self.merge_rules = self.learn_vocabulary()

    def learn_vocabulary(self):
        vocab = defaultdict(int)
        for line in self.corpus:
            for word in line.split():
                vocab[' '.join(list(word)) + ' $'] += 1
        self.vocab = vocab

        # Perform byte pair encoding based on the specified number of merges
        self.vocab, merge_rules = self.byte_pair_encoding(self.num_merges)
        learnt_vocab = self.vocab
        return learnt_vocab, merge_rules

    def tokenize(self, sample):
        '''
        # Tokenize the input sample based on the vocabulary learnt
        tokens = []
        #print(sample)
        for word in sample.split():
            word = ''.join(list(word)) + '$'
            #print(word)
            word_tokens = []
            #print(word_tokens)
            #print(self.learnt_vocab)
            while word:
                found = False
                for i in range(len(word), 0, -1):
                    if word[:i] in self.learnt_vocab:
                        word_tokens.append(word[:i])
                        word = word[i:]
                        found = True
                        break
                if not found:
                    word_tokens.append(word[0])
                    word = word[1:]
            tokens.extend(word_tokens)
        return tokens
        '''
        result = []
        for line in sample.split('\n'):
            tokens = []
            #print(sample)
            for word in sample.split():
                word = ''.join(list(word)) + '$'
                #print(word)
                word_tokens = []
                #print(word_tokens)
                #print(self.learnt_vocab)
                while word:
                    found = False
                    for i in range(len(word), 0, -1):
                        if word[:i] in self.learnt_vocab:
                            word_tokens.append(word[:i])
                            word = word[i:]
                            found = True
                            break
                    if not found:
                        word_tokens.append(word[0])
                        word = word[1:]
                tokens.extend(word_tokens)
            result.append(tokens)
            with open('/content/drive/MyDrive/NLP/tokenized_sample.txt', 'w') as f:
                f.write(','.join(tokens) + '\n')
        return result

    def get_freq_pair(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            new_words = word.split()
            for i in range(len(new_words)-1):
                pairs[new_words[i],new_words[i+1]] += freq
        return pairs

    def merge_vocab(self, pair):
        new_vocab = {}
        for word, freq in self.vocab.items():
            new_word = word.replace(' '.join(pair), ''.join(pair))
            new_vocab[new_word] = freq
        return new_vocab

    def byte_pair_encoding(self, num_merges):
        merge_rules=[]
        for i in range(num_merges):
            pairs = self.get_freq_pair()
            if not pairs:
                break
            best_merged_pair = max(pairs, key=pairs.get)
            merge_rules.append(best_merged_pair)
            self.vocab = self.merge_vocab(best_merged_pair)
        return self.vocab, merge_rules

    def write_merge_rules(self, filename):
        with open(filename, 'w') as f:
            for pair in self.merge_rules:
                f.write(f"{pair[0]}, {pair[1]}\n")

    def write_all_tokens(self, filename):
        all_tokens = set()
        for word in self.learnt_vocab.keys():
            subwords = word.split()
            for i in range(len(subwords)):
                for j in range(i+1, len(subwords)+1):
                    all_tokens.add(''.join(subwords[i:j]))
        with open(filename, 'w') as f:
            for token in sorted(all_tokens):
                f.write(f"{token}\n")

def read_corpus(corpus):
    with open(corpus, 'r') as f:
        lines = [line.rstrip() for line in f]
    return lines

corpus = '/content/drive/MyDrive/NLP/data-20240130T044130Z-001/data/corpus.txt'
print(read_corpus(corpus))
tokenizer = Tokenizer(read_corpus(corpus),850)
tokenizer.learn_vocabulary()
tokenizer.write_merge_rules("/content/drive/MyDrive/NLP/merge_rules.txt")
tokenizer.write_all_tokens("/content/drive/MyDrive/NLP/tokens.txt")
print("Learned vocabulary (BPE pairs):")
print(tokenizer.learnt_vocab)

sample_text = "this is an nlp course"
tokens = tokenizer.tokenize(sample_text)
print("\nTokenized text:")
print(tokens)
