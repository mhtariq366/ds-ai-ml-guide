import json

fruits = {}

fruits['orange'] = {
    'name': 'oranges',
    'color': 'orange',
    'season': 'winter',
}

fruits['mango'] = {
    'name': 'mango',
    'color': 'yellow',
    'season': 'summer',
}

fruits['apple'] = {
    'name': 'apple',
    'color': 'red green',
    'season': 'all',
}

j = json.dumps(fruits)

with open('Datasets/json_format.txt', 'w') as f:
    f.write(j) 