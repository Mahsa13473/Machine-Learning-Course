#!/usr/bin/env python

import assignment1 as a1
import numpy as np

(countries, features, values) = a1.load_unicef_data()

print("5.1")
#5.1.1
maximum = 0
index = 0
m = len(countries)

for i in range(1,m):
    if values[i,0] > maximum:
        maximum = values[i, 0]
        index = i

print("The highest child mortality rate in 1990 =", maximum, "for", countries[index])

#5.1.2
maximum = 0
index = 0
m = len(countries)

for i in range(1,m):
    if values[i,1] > maximum:
        maximum = values[i, 1]
        index = i

print("The highest child mortality rate in 2011 =", maximum, "for", countries[index])
