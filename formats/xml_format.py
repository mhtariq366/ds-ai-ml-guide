import xml.etree.ElementTree as ET


student = '''
        <student>
            <name> Jon </name>
            <subject> AI </subject>
        </student>
'''

print(student)


#   An other way using xml module

student = ET.Element('student')
name = ET.SubElement(student, 'name')
name.text = 'Jon'

tree = ET.ElementTree(student)

print(tree)

