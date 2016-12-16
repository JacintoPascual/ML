
# Working with List
# The array module provides an array() object that is like a list that stores
# only homogeneous data and stores it more compactly.
# The following example shows an array of numbers stored as
# two byte unsigned binary numbers (typecode "H") rather than the usual 16 bytes
from array import array
a = array('H', [4000, 10, 700, 22222])
sum(a) # 26932
a[1:3]

# The collections module provides a deque() object that is like a list with faster
# appends and pops from the left side but slower lookups in the middle.
# These objects are well suited for implementing queues and breadth first tree searches:
from collections import deque
d = deque(["task1", "task2", "task3"])
d.append("task4")
print("Handling", d.popleft()) # Handling task1

# In addition to alternative list implementations, the library also offers other tools
#  such as the bisect module with functions for manipulating sorted lists:
import bisect
scores = [(100, 'perl'), (200, 'tcl'), (400, 'lua'), (500, 'python')]
bisect.insort(scores, (300, 'ruby'))
# [(100, 'perl'), (200, 'tcl'), (300, 'ruby'), (400, 'lua'), (500, 'python')]

# The heapq module provides functions for implementing heaps based on regular lists.
# The lowest valued entry is always kept at position zero.
# This is useful for applications which repeatedly access the smallest element
# but do not want to run a full list sort:
from heapq import heapify, heappop, heappush
data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
heapify(data)   # rearrange the list into heap order
heappush(data, -5)  # add a new entry
[heappop(data) for i in range(3)]  # fetch the three smallest entries
# [-5, 0, 1]

# List comprehension
squares = [x**2 for x in range(10)]
# When the expression is a tuple it must be parenthesized
squares = [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
# Call a method on each element
freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
strip_freshfruit = [strpfruit.strip() for strpfruit in freshfruit]

matrix = [[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12]]
matrix_transpose = [[row[i] for row in matrix] for i in range(4)]

# Named tuples - May be useful in some situations to access fields by names
from collections import namedtuple
import csv
import sqlite3
EmployeeRecord = namedtuple('EmployeeRecord', 'name, age, title, department, paygrade')
for emp in map(EmployeeRecord._make, csv.reader(open("employees.csv", "rb"))):
    print(emp.name, emp.title)

conn = sqlite3.connect('/companydata')
cursor = conn.cursor()
cursor.execute('SELECT name, age, title, department, paygrade FROM employees')
for emp in map(EmployeeRecord._make, cursor.fetchall()):
    print(emp.name, emp.title)

# Set - Created either with curly braces or with set()
# Unordered with no duplicates
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)

