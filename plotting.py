#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:59:43 2020

@author: automatus
"""
# based on: https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py

import matplotlib.pyplot as plt


fig, ax = plt.subplots()

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
# https://www.science-emergence.com/Articles/How-to-change-the-color-background-of-a-matplotlib-figure-/
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='yellow')
ax.tick_params(axis='y', colors='yellow')
# https://stackoverflow.com/questions/1982770/matplotlib-changing-the-color-of-an-axis#12059429

ax.plot([1, 2, 3, 4, 5], [1, 30, 12, 25, 50], color="yellow")
