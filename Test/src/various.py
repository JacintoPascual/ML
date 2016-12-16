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
logging.warning('Warning:config file %s not found', 'server.conf')
logging.error('Error occurred')
logging.critical('Critical error -- shutting down')


now = date.today()
now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B.")
# '12-02-03. 02 Dec 2003 is a Tuesday on the 02 day of December.'

