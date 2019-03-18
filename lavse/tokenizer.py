from .tokenizers.word import WordTokenizer


class Tokenizer(object):
    """
    This class converts texts into character or word-level tokens
    """

    def __init__(
        self, word_level=True, 
        char_level=True, vocab_path=None
    ):  
        self.word_level = word_level 
        self.char_level = char_level 
        
        if word_level:
            self.word_tokenizer = WordTokenizer() 
            if vocab_path is not None:
                self.word_tokenizer = self.load(vocab_path)
        
        if char_level:
            self.char_tokenizer = lambda x: x

    def save(self, outpath):
        if self.word_level:
            self.word_tokenizer.save(outpath)
    
    def load(self, path):
        self.word_tokenizer = self.word_tokenizer.load(path)
        return self.word_tokenizer
    
    def get_nb_words(self, ):
        return len(self.word_tokenizer)

    def tokenize(self, sentence):
        word = None
        char = None

        if self.word_level:
            word = self.word_tokenizer.tokenize(sentence)
        if self.char_level:
            char = self.char_tokenizer.tokenize(sentence)
            
        return word, char

    def __call__(self, sentence):
        return self.tokenize(sentence)
