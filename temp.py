import lavse
import numpy as np

a = np.load('/opt/jonatas/datasets/lavse/jap_precomp/test_ids.npy')
np.savetxt('/opt/jonatas/datasets/lavse/jap_precomp/test_ids.txt', a.astype(np.int32),)

print(a)
exit()

def print_stats(x):
    print(f'Min : {np.min(x)}')
    print(f'Max : {np.max(x)}')
    print(f'Mean: {np.mean(x)}')
    print(f'Std : {np.std(x)}')


# ds = lavse.data.datasets.CrossLanguageLoader(
#     data_path='/opt/jonatas/datasets/lavse/',
#     data_name='jap_precomp',
#     lang='en-jt',
#     data_split='train',
#     tokenizers=[lavse.data.tokenizer.Tokenizer('.vocab_cache/char.json')],
# )

# # def collect_stats(sentences):

# wlen = []
# char_per_word = []
# nchar = []
# for caption in ds.lang_a:
#     words = ds.tokenizer.split_sentence(caption)
#     wlen.append(len(words))
#     char_per_word.extend([len(w) for w in words])
#     nchar.append(len(caption))

# print('\nLang A')
# print('Word Lengths')
# print_stats(wlen)

# print('\nChar per word')
# print_stats(char_per_word)

# print('\nN Char')
# print_stats(nchar)


# wlen = []
# char_per_word = []
# nchar = []
# for caption in ds.lang_b:
#     words = ds.tokenizer.split_sentence(caption)
#     wlen.append(len(words))
#     char_per_word.extend([len(w) for w in words])
#     nchar.append(len(caption))

# print('\nLang B')
# print('Word Lengths')
# print_stats(wlen)

# print('\nChar per word')
# print_stats(char_per_word)

# print('\nN Char')
# print_stats(nchar)
