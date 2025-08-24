"""
Not typically a part of preprocessing, but can be used in filtering out data as well as replacing.

Regular expressions

.               any character
^               start of the string, but when used in [] braces, it means not
$               end of the string
*               zero or more occurances of the character it is placed after
+               one or more occurances of the character it is placed after
?               exactly one or none occurance of character is place after
\w              alphabets, numbers, and underscore
\d              digits from 0 to 9. One occurance. For multiple, use it repeatedly
\s              blank space
\b              boundary values
{number}        specify the number of character occurance it is placed after
[characters]    specify list of characters to find, any occurance
(groups)        specify the groups of characters
|               OR

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


print(re.findall(r'\bis\b', sentence))

print('--------')

with open('Datasets/regex.txt', 'r') as f:
    dataset = f.read()

#   findall digits separated
print(re.findall(r'(\d)',dataset))

#   findall digits in groups
print(re.findall(r'(\d+)',dataset))

