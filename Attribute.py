class Attribute:
    def commDutchWords(self, sentence):
        dutch_word_list = [' ook ', ' onze ', ' meest ', ' hun ', ' zo ', ' ben ', ' met ', ' dan ', ' weten ', ' ons ' ,' jouw ',' voor '
                           ' ik ', ' deze ', ' ze ', ' niet ', ' hem ', ' naar ', ' er ', ' het ']

        sentence_list = sentence.split()

        for word in sentence_list:
            if word in dutch_word_list:
                return True
            else:
                return False

    def commEnglishWords(self,sentence):

        english_word_list = [' there ',' here ',' they ',' their ',' so ',' our ',' him ',' her ',' us ',' as ',' with ',' it ',' them ',
                             ' about ',' she ',' will ',' his ','hers','mine ',' me ',' not ',' to ',' for ',' we ',' he ','an', 'in']

        sentence_list = sentence.split()

        for word in sentence_list:
            if word not in english_word_list:
                return True
            else:
                return False

    def qInWord(self,sentence):
        if 'q' in sentence:
            return False
        else:
            return True

    def ooInWord(self,sentence):
        if 'oo' in sentence:
            return True
        else:
            return False

    def eenInWord(self,sentence):
        if ' een ' in sentence:
            return True
        else:
            return False

    def aaInWord(self,sentence):
        if 'aa' in sentence:
            return True
        else:
            return False

    def ijInWord(self,sentence):
        if 'ij' in sentence:
            return True
        else:
            return False

    def eeInWord(self,sentence):
        if 'ee' in sentence:
            return True
        else:
            return False

    def deInWord(self,sentence):
        if ' de ' in sentence:
            return True
        else:
            return False

    def enInWord(self,sentence):
        if 'en' in sentence:
            return True
        else:
            return False

    def vanInWord(self,sentence):
        if ' van ' in sentence:
            return True
        else:
            return False

def sentenceFeature(sentence):
    boolSentenceVal = []
    obj = Attribute()

    boolSentenceVal.append(obj.commDutchWords(sentence))
    boolSentenceVal.append(obj.commEnglishWords(sentence))
    boolSentenceVal.append(obj.aaInWord(sentence))
    boolSentenceVal.append(obj.ijInWord(sentence))
    boolSentenceVal.append(obj.ooInWord(sentence))
    boolSentenceVal.append(obj.eeInWord(sentence))
    boolSentenceVal.append(obj.qInWord(sentence))
    boolSentenceVal.append(obj.deInWord(sentence))
    boolSentenceVal.append(obj.enInWord(sentence))
    boolSentenceVal.append(obj.eenInWord(sentence))
    boolSentenceVal.append(obj.vanInWord(sentence))

    return boolSentenceVal
