---
layout: single
title: "Ticket"
categories : Algorithm
tag: [Algorithm, dfs, programmers]
toc: true
author_profile : false
use_math: true
sidebar:
     nav: "docs"
---

# DFS

> ## What Was Difficult in the Problem
> * I created a data structure using a dictionary, but struggled a lot with how to handle **duplicate keys**.
>> 1. Tried declaring a class to avoid key duplication:

```python
class ticket(object):
    def __init__(self, depart):
        self.depart = depart
    # Since it's a class, define its print format
    def __str__(self):
        return self.depart
    def __repr__(self):
        return "'" + self.depart + "'"

def pre_process(tickets):
    data = {}

    for t in tickets:
        data[ticket(t[0])] = t[1]
    return data
```

>>> When implemented like this, you run into problems retrieving `val` from the dictionary using a key.  
>>> Even if `v` is the same, every call to `ticket(v)` creates a new object with a different memory address, so the value can't be found.

```python
for v in val:
    if data.get(ticket(v)) == None:
        print(f"v = {v}, get(v) = {data.get(v)}")
        first = v
        break
```

>> 2. If a key has multiple values, I tried storing the values as a list (this took a lot of thinking to figure out...):

```python
def pre_process(tickets):
    data = {}

    for t in tickets:
        ex = []

        if data.get(t[0]) == None:
            data[t[0]] = t[1]
        else:
            if type(data[t[0]]) == str:
                ex.append(data[t[0]])
                ex.append(t[1])
                data[t[0]] = ex
            else:
                for d in data[t[0]]:
                    ex.append(d)
                    ex.append(t[1])
                    data[t[0]] = ex

    return data
```

>>> The tricky part was that Python doesn’t clearly distinguish when `val` is a single string versus when it’s already a list of multiple values, which made handling both cases quite challenging...

> * I thought the problem would be easier if I could define a clear start and end city,  
>   but if a city is visited multiple times, you can’t determine the start and end clearly.  
>   I ended up spending a lot of time here.
>   Perhaps the real intent of the problem is not to find explicit start/end points, but to solve it by connecting paths using a **linked-list-like** approach.

---

> ## The Standard Solution
> * This is a classic DFS + stack problem.
>> 1. Use a dictionary to build key and value pairs (super easy):

```python
for t in tickets:
    routes[t[0]] = routes.get(t[0], []) + [t[1]]
```

>> * Very simple and clean.
>> 2. Declare a stack starting with `"ICN"` as the initial condition, and also declare `path` to store the final route when the stack is empty.
>> 3. Repeat the loop until the stack is empty!

---

> ## Question

```python
def solution(tickets):
    answer = []

    trip = {}
    for t in tickets:
        trip[t[0]] = trip.get(t[0], []) + [t[1]]

    for t in trip:
        trip[t].sort(reverse=True)

    print(trip)

    stack = ['ICN']

    while len(stack) > 0:
        prev = stack[-1]
        if prev not in trip or trip.get(prev) == []:
            break
        else:
            next = trip[prev][-1]
            stack.append(next)
            trip[prev].pop()

    return stack
```

>> * Why **doesn't this code work**?

```python
def solution(tickets):
    answer = []

    trip = {}
    for t in tickets:
        trip[t[0]] = trip.get(t[0], []) + [t[1]]

    for t in trip:
        trip[t].sort(reverse=True)

    print(trip)

    stack = ['ICN']
    path = []
    while len(stack) > 0:
        prev = stack[-1]
        if prev not in trip or trip.get(prev) == []:
            path.append(stack.pop())
        else:
            next = trip[prev][-1]
            stack.append(next)
            trip[prev].pop()

    return path[::-1]
```

>> * But this one **does** work... what’s the issue here?
