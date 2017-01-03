import logging
from datetime import date
from string import Template
t = Template('${village}folk send $$10 to $cause.')
s = t.substitute(village='Nottingham', cause='the ditch fund')
print(s)  # 'Nottinghamfolk send $10 to the ditch fund.'

# The substitute() method raises a KeyError when a placeholder is not supplied
# in a dictionary or a keyword argument
# For mail-merge style applications, the safe_substitute() method may be more appropriate
# it will leave placeholders unchanged if data is missing:
t = Template('Return the $item to $owner.')
d = dict(item='unladen swallow')
s = t.safe_substitute(d)
print(s)  # 'Return the unladen swallow to $owner.'


# Debug and info are by default suppressed
logging.debug('Debugging information')
logging.info('Informational message')
# logging.warning('Warning:config file %s not found', 'server.conf')
# logging.error('Error occurred')
# logging.critical('Critical error -- shutting down')

now = date.today()
print("Formated date now: ", now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B."))
# '12-02-03. 02 Dec 2003 is a Tuesday on the 02 day of December.'
print("Unformated date now: ", now)

# *********************************************************************
# lambda
a = [(lambda x: x*x)(x) for x in range(10)]
print("Printing a version 1: ", a)
a = [x*x for x in range(10)]
print("Printing a version 2: ", a)
a = map(lambda x: x*x, range(10))
print("Printing map: ", a)
a = list(a)
print("Printing list from map: ", a)
print("Printing list item from map: ", a[2])


# List comprehension
x = [2, 3, 4, 5, 6]
y = [v * 5 for v in x]
print("Printing y list: ", y)
# Same result. Map takes a function and an iterable and then return another iterable.
y = map(lambda v: v*5, x)
print("Printing y map: ", y)
print("Printing y list from map: ", list(y))


# Multiply odd values and add to list
y = [v * 5 for v in x if v % 2]
# filter() takes in a function (lambda) and iterable(x) and return an iterable containing "filtered" values
y = map(lambda v: v*5, filter(lambda u: u % 2, x))
y = list(y)
print("Printing y list of odds from map: ", y)


x = [2, 3, 4]
y = [4, 5]
z = []
for v in x:
    for w in y:
        z += [v + w]
assert z == [2+4, 2+5, 3+4, 3+5, 4+4, 4+5]
assert z == [6, 7, 7, 8, 8, 9]
# With list comprehension
z = [v + w for v in x for w in y]
assert z == [6, 7, 7, 8, 8, 9]

# With "pseudo map and lambda"
# map(lambda for v: in map(lambda for w: v + w, in y), in x)
# With map and lambda
t = map(lambda v: map(lambda g: v+g, y), x)
# *********** NOT AS EXPECTED ****************
# *********** Must be reviewed later ****************
print("Printing list from map t", list(t))
print("Printing t==(v1)", t == [[6, 7], [7, 8], [8, 9]])
print("Printing t==(v2)", list(t) == [[6, 7], [7, 8], [8, 9]])
print("Printing z", z)
# z = sum(t, [])
print("Printing t==(v1)", z == [2+4, 2+5, 3+4, 3+5, 4+4, 4+5])
print("Printing t==(v2)", z == [6, 7, 7, 8, 8, 9])


