from interface import implements, Interface

class LanguageModelTranslator(Interface):

    def convert(self, words_list):
        pass
    
    def load_model(self):
        pass