"""
Regular expressions
"""

import re

sentence = 'Hello everyone, this is python regular expressions practise.'

"""
search specific characters in a sequence, return character match and position (start, end)
return only first instance of character
"""
print(re.search('on', sentence))


"""
finding how many times a sequence of character appear in the dataset.
re.findall() function return list of all occurances of the character.
"""
print(re.findall('on', sentence))


"""
To split, dataset based on a specific character. Mostly useful in spliting data based on a period, comma etc.
returns a list of all the data elements after split.
"""
print(re.split('on', sentence))


"""
to replace any characters with new sequence of characters, re.sub() function is used.
first paramter is what you want to replace
second paramter is what you want the characters to be replaced with
"""
print(re.sub('everyone', 'people', sentence))


