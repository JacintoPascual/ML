
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
