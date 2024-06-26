---
layout: single
title: "Deeplearning"
categories : Artificial_Intelligence
tag: [Artificial_Intelligence, Deeplearning]
toc: true
author_profile : false
use_math: true
sidebar:
     nav: "docs/Deeplearning"
---

# 딥러닝의 학습 방법

> ## 신경망이란?
> * 신경망을 공부한다는것은 비선형모델을 공부한다는것이며, 선형모델과 비선형 함수의 결합인 것이다.
> * **softmax** 와 **활성함수**의 정의를 알아야 신경망을 이해할 수 있다. 
> * 선형모델의 동작방식을 알아야 어떻게 비선형모델을 배울 수 있을지 알 수 있다. 
> * 그에 앞서 행렬을 먼저 정의하고 가자. 딥러닝에서의 행렬은 크게 2가지로 나뉘는데, **데이터를 모아둔 행렬** 과, **이 데이터를 다른 차원의 공간으로 보내주는 행렬** 두가지가 있다.
>  <img width="1222" alt="스크린샷 2022-07-23 오후 11 42 42" src="https://user-images.githubusercontent.com/63406434/180610035-fd62034a-27f9-4fd2-bf0f-6cd42c549b90.png">
> * 위의 이미지에서 X는 데이터 행렬이고, W는 가중치 행렬, b는 절편에 해당하는 행렬이며, 각 행들이 모두 값이 같다. 이 때, 연산을 거치고 나면 출력벡터의 차원이 d -> p로 변경이 된다. 
> * 위 말은 d개의 변수로 p개의 선형모델을 만들어서 p개의 잠재변수로 복잡한 식을 설정한다?라는 말과 같다. 
> ### softmax연산 
> * 소프트맥스(softmax)함수는 모델의 출력을 확률로 해석할수있게 변환해주는 연산이다.
> * 분류문제에서 특정클래스 k에 속하는지 아닌지를 판단할때 사용된다. 
> * 즉, 분류문제 -> 선형모델 + softmax 함수로 푼다.
> * 그렇다면, **등장한 배경**은?
> * 단순히 추론만 한다면 원핫벡터를 이용해 단순하게 접근하지만, 일반적인 경우 선형모델로 출력한 값이 확률벡터가 아닌경우가 굉장히 많다. 이때, softmax를 이용해 출력값을 변환해 분류문제로 푼다.
> ### 활성함수(activation function)
> * 등장배경? 비선형함수를 모델링할때 사용되는 트릭
> * 활성함수는 비선형함수로서 선형모델로 나오게 되는 출력의 각각의 원소에 적용하게된다. 
> * 각각의 주소에 해당하는 출력물에만 적용한다.
> * 주로 input은 실수이며 벡터가 아니다. 실수값의 입력을 받아서 다시 출력을 실수값으로 한다.
> * 이 활성함수를 이용해서, 선형모델로 나온 출력물을 비선형모델로 **변경가능**하다.
> * 오늘날 가장 많이 쓰는 것 → ReLU 
> <img width="590" alt="스크린샷 2022-07-24 오전 12 08 25" src="https://user-images.githubusercontent.com/63406434/180610867-1699cc3f-c679-4ad7-a815-ceda75272459.png">

> ## 딥러닝은 어떻게 학습할까?
> * 쉽게 말해, 선형모델과 활성함수를 반복적으로 사용하는것!
> * 다음 2층 신경망을 보며 이해해보자!  
> <img width="192" alt="스크린샷 2022-07-24 오전 12 08 29" src="https://user-images.githubusercontent.com/63406434/180610868-ba5085db-f8eb-445b-9913-e4c51eaaa10e.png">
> * 각각의 행렬의 구성원소 z에다가 활성함수를 씌운다.   
> $\displaystyle\sum_{n=1} ^{\infty} z$
> * 왜 2층 가지고는 안될까? 층을 많이 쌓을 수록 좋은이유는 목적함수를 근사하는데 필요한 뉴런의 숫자가 훨씬 빨리 줄어든다. 
> * forward propagation → x의 입력을 받아서 선형모델과 합성함수 반복적으로 적용하는것
> * backward propation -> 경사하강법 적용, 각각의 가중치의 그래디언트 알아야한다. 각 층에 존재하는 파라미터의 미분을 가지고, 행렬들의 모든 원소에 대해서 경사하강법 적용해서 x를 구하는것 같다. 
> * 딥러닝이란 층별로 계산을 하되, 단 층을 2개이상 연속으로 못뛰어넘는것 같다. 위층의 gradient 계산하고 그 밑의 층 gradient 계산하고, 차례차례로 진행하는것 같다. 


