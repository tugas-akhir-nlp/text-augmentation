import re

def number_masking(sentence):
    sentence = re.sub(r'[0-9]+', r'<NUM>', sentence)
    
    return sentence