---
layout: single
title: "Ticket"
categories : Algorithm/DFS
tag: [Algorithm, DFS, programmers]
toc: true
author_profile : false
use_math: true
sidebar:
     nav: "docs"
---

# DFS 

> ## 문제에서 어려웠던 점
> * 딕셔너리로 자료구조를 제작했는데, 키가 중복되어 이를 어떻게 처리할까 고민을 많이 했다.
>> 1. 클래스를 선언해서 키중복 막기 

```python
  class ticket(object):
    def __init__(self, depart):
        self.depart = depart
    # 클래스니까 출력 포맷을 정한다.
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
>>> 다음과 같이 풀게되면, 키를 이용해서 딕셔너리에서 val를 찾을 때 문제가 생긴다. 아마 ticket(v)할때마다, v가 같더라고 하더라도 객체를 생성하면서 
>>> 객체마다 주소가 달라지니까 val를 찾지못한다.
```python
 for v in val:
        if data.get(ticket(v)) == None:
            print(f"v = {v}, get(v) = {data.get(v)}")
            first = v
            break
```
>> 2. val가 여러개라면 val를 리스트화하기(정말 머리많이 써서 만든 코드이다..)
```python
def pre_process(tickets):
    data = {}

    for t in tickets:
        ex = []

        if data.get(t[0]) == None:
            data[t[0]] = t[1]
        else :
            if type(data[t[0]]) == str:
                ex.append(data[t[0]])
                ex.append(t[1])
                data[t[0]] = ex
            else :
                for d in data[t[0]]:
                    ex.append(d)
                    ex.append(d[1])
                    data[t[0]] = ex 

    return data
```
>>> 그런데, 좀 어려웠던게 파이썬이 val가 하나일때, 그리고 지금 리스트로 이미 여러개가 val일때를 구분하지 못하니까, 이걸 처리하는데 애를 좀 먹었다....
> * 처음 출발점과 도착점을 정의하면 문제풀기 쉽다고 생각했는데, 한 도시를 여러번 방문할 경우, 출발점과 도착점을 찾을 수 없어서 여기서 시간을 많이보냈다.
> 아마, 출발점과 도착점을 굳이 찾지말고 연결리스트를 이용해서 문제를 푸는게 이 문제의 의도같다. 


> ## 정석풀이
> * dfs와 스택을 이용하는 전형적인 문제 
>> 1. 딕셔너리를 이용해 key와 val값 설정을 한다.(아주 쉽게~)
```python
for t in tickets:
        routes[t[0]] = routes.get(t[0], []) + [t[1]]
```

>> * 정말 심플하게 짯다... 
>> 2. 시작 조건이 ICN으로 고정되게 stack 자료구조 선언, 그리고 stack이 모두 찼을때, 최종경로를 기록할 path도 선언
>> 3. stack에 아무 원소가 없을때까지, 반복!   

> ## 궁금점

```python

def solution(tickets):
    answer = []
    
    trip = {}
    for t in tickets:
        trip[t[0]] = trip.get(t[0], []) + [t[1]]
    
    for t in trip:
        trip[t].sort(reverse = True)
    
    print(trip)
    
    stack = ['ICN']
    
    while len(stack) > 0:
        prev = stack[-1]
        if prev not in trip or trip.get(prev) == []:
            break
        else :
            next = trip[prev][-1]
            stack.append(next)
            trip[prev].pop()
        
    return stack
```
>> * 이 코드는 왜 작동안해?

```python
def solution(tickets):
    answer = []
    
    trip = {}
    for t in tickets:
        trip[t[0]] = trip.get(t[0], []) + [t[1]]
    
    for t in trip:
        trip[t].sort(reverse = True)
    
    print(trip)
    
    stack = ['ICN']
    path = []
    while len(stack) > 0:
        prev = stack[-1]
        if prev not in trip or trip.get(prev) == []:
            path.append(stack.pop())
        else :
            next = trip[prev][-1]
            stack.append(next)
            trip[prev].pop()
        
    return path[::-1]
```

>> * 이건 또 작동한단 말이야... 무슨 문제야? 