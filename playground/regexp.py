import re

path = 'tmp/20160916-r63/'

p = re.compile('[0-9]{8}-r[0-9]*')
for m in p.finditer(path):
    print(m.start(), m.group())

path = path[m.start():m.end()]

print(path)

# get parameters from image path
path = '/Users/schiend/processing/N1-19/20160706/N1-19-20160617-wdB2-plusAb-pouch-iso.tif'

for match in re.finditer('-[0-9]+', '-1-r0'):
    pass

print(match)
