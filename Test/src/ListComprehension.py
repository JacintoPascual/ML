'''
Created on Sep 27, 2016

@author: Jacinto
'''

# Some standard libraries
# See > https://docs.python.org/3.5/tutorial/stdlib2.html
from src.apihelper import *
import string
import math as m
from collections import deque
import os as opSys
import sys
import re # Regular expression
import random
import statistics
from datetime import date

# For data compression
# zlib, gzip, bz2, lzma, zipfile, tarfile

# For Performance measurements
# timeit, profile and pstats modules

# For testing
# doctest and unittest

# For Remote Procedure Call
# import xmlrpc.client and xmlrpc.server modules

import json
import sqlite3

print(string.punctuation)

aMath = m.ceil(2.8)

sys.stderr.write('Warning, log file not found starting a new one\n')

re.findall(r'\bf[a-z]*', 'which foot or hand fell fastest')
# Starting with 'f' follow by characters [a-z]
# ['foot', 'fell', 'fastest']
re.sub(r'(\b[a-z]+) \1', r'\1', 'cat in the the hat')
'cat in the hat'

vec = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
[num for elem in vec for num in elem]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# create a list of 2-tuples like (number, square)
[(x, x**2) for x in range(6)]
# Result [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]

x = 5
while x > 0:
    if x == 5:
        print('Value of x is:', x)
        print('Value of x is:', x, end=", ")
    elif x == 1:
        print(x)
    else:
        print(x)
        # print( x, end=', ')
    x -= x

# equivalent list comprehension
# syntax: [expression for variable in iterable if condition]
nums = [1, 2, 3, 4, 5]
cubes = [num**3 for num in nums]
cubes_of_even = [num**3 for num in nums if num % 2 == 0]
cubes_and_squares = [num**3 if num % 2 == 0 else num**2 for num in nums]

matrix = [[1, 2], [3, 4]]
# equivalent list comprehension
items = [item for row in matrix for item in row]

fruits = ['apple', 'banana', 'cherry']
unique_lengths = {len(fruit) for fruit in fruits}
print(unique_lengths)

fruit_lengths = {fruit: len(fruit) for fruit in fruits}
fruit_lengths


# map applies a function to every element of a sequence and returns a iterator:
simpsons = ['homer', 'marge', 'bart']
map(len, simpsons)

# filter returns a iterator containing the elements from a sequence for which a condition is True:
nums = range(5)
filter(lambda x: x % 2 == 0, nums)


words = ['cat', 'window', 'defenestrate']
for w in words:
    print(w, len(w))

# A tuple is an immutable list. A tuple can not be changed in any way once it is created.
t = ("a", "b", "mpilgrim", "z", "example")
print(t)


li = []
print(li)
print("\nHei " + str(li) + "\n")

info(li)


queue = deque(["Eric", "John", "Michael"])
queue.append("Terry")           # Terry arrives
queue.append("Graham")          # Graham arrives
queue.popleft()                 # The first to arrive now leaves
queue.popleft()                 # The second to arrive now leaves

print("\nThis is the queue", queue)                   # Remaining queue in order of arrival


squares = []
for x in range(10):
    squares.append(x**2)

print(squares)
# x still exists after the for loop...!
print(x)

# Two other ways of declaring the List
squares = list(map(lambda x: x**2, range(10)))
print(squares)
squares = [x**2 for x in range(10)]
print(squares)

list = [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
print(list)  # [(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]

vec = [-4, -2, 0, 2, 4]
l = [x*2 for x in vec]
print(l)
l = [x for x in vec if x >= 0]
print(l)
l = [abs(x) for x in vec]
print(l)
l = [(x, x**2) for x in range(6)]
print(l)
vec = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
l = [num for elem in vec for num in elem]  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(l)

# This is a Set
s = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
s = set()  # This is an empty set

# This is a dictionary
d = {"one": "1", "two": "2"}
d = {}  # This is an empty dictionary. NOT an empty Set

data = 'golf'
r = range(len(data)-1, -1, -1)
print(r)

ops = dir(opSys)
print(ops)
ops = opSys.getcwd()
print(ops)

c = m.ceil(2.56)
print(c)

