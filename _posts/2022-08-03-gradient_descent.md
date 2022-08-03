---
layout: single
title: "Gradient Descent"
categories : Artificial_Intelligence
tag: [Artificial_Intelligence, gradient, gradient_descent]
toc: true
author_profile : false
use_math: true
sidebar:
     nav: "docs"
---



## 경사하강법

* 직관적의미 : 길을 잃었을때 어떻게 하면 가장 낮은곳을 내려갈수 있는가? 기울기가 가장 낮은 방향으로 한걸음 한걸음 하면 성공!

* 목적: 함수의 최솟값 찾을때, 굳이 쓰는 이유는 함수가 닫힌 형태가 아니거나 미분계수 구하기 어려운 경우

* 수식유도: x의 값을 어디로 옮길 때 최솟값을 가지게 되는지??

* 변수가 벡터인경우에 편미분을 사용

* 최솟값에 가까울 수록 조심조심히 움직여야함

* 각 변수 별로 편미분을 계산한 그래디언트 벡터를 경사하강법에 사용가능

* 그래디언트 벡터에 음수를 취하면, 각 점마다 최솟점을 향하는 방향으로의 벡터로 볼 수있다. (이때는 Norm을 사용한다.)

* 경사하강법을 이용해 선형모델을 찾아보자 → 무어테러즈 역행렬은 선형모델에서만, 경사하강법은 선형적이지 않은 모델에서도 활용가능

* 그래디언트 벡터를 쓰는 이유

  >다음의 공간에서 그래디언트 벡터는 각 점에서 최솟값으로 향하는 방향이다. 이 방향을 이용해 공간에서의 최솟값을 찾는다.

![스크린샷 2022-08-02 오후 1.57.09](/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-02 오후 1.57.09.png)

* 선형회귀의 목적식

  >y: 주어진 데이터에서 정답이라고 여겨지는것
  >
  >선형모델 :  x * beta 행렬 
  >
  >목적 : 2차 norm의 최솟값을 찾기 
  >
  >**다음과 같은 목적식을 최소화해야함**

$$
\lVert  y - X\beta  \rVert_2
$$
* 선형회귀의 목적식을 미분을 이용해 풀기

![스크린샷 2022-08-02 오후 2.11.05](/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-02 오후 2.11.05.png)

> 다음과 같이 정리됨

![스크린샷 2022-08-02 오후 2.11.49](/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-02 오후 2.11.49.png) 

> 그래디언트 벡터로 다음과 같이 나타낸다. 

![스크린샷 2022-08-02 오후 2.15.23](/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-02 오후 2.15.23.png)

>(t+1). 번째 값을 구하려면 t번째 값에서 그래디언트 벡터를 빼준다.
>
>이론적으로, 경사하강법은 미분가능하고 convex한 함수에 대해서는 수렴이 보장된다.
>
>하지만, **비선형회귀 문제의 경우에는 convex가 보장되지 않는다** -> **문제**
>
>대책 :  **확률적 경사하강법의 등장!**
>
>데이터의 일부만 사용하지만, 데이터의 일부만 사용해도 모든 데이터를 사용한 방법과 유사하다. 딥러닝의 경우에는 sgd가 더 낫다. 연산량 감소가 혁명적으로 일어난다.

![스크린샷 2022-08-02 오후 2.53.53](/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-02 오후 2.53.53.png)

* 경사하강법 예시1(하나의 정보로부터 숫자하나를 예측하는 모델)

  >* 다음과 같이 hours와 points가 주어졌을때, 어떻게 모델의 좋고 나쁨을 판단할것인가?
  >
  ><img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 2.18.39.png" alt="스크린샷 2022-08-03 오후 2.18.39" style="zoom:50%;" />
  >
  >* cost function(예측값과 실제값의 차이) 으로 좋고 나쁨을 판단한다.
  >* ![스크린샷 2022-08-03 오후 2.20.09](/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 2.20.09.png)
  >* cost function은 w에 관한 2차함수이며, cost function을 최소화하기 위해서는 미분을 이용한다. 이때, 기울기가 음수일땐 w가 더 커져야하고, 기울기가 양수일때는 w가 더 작아져야함. gradient를 이용해 cost를 줄인다. ![스크린샷 2022-08-03 오후 2.25.18](/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 2.25.18.png)

* Minibatch gradient descent (전체데이터를 쓰지말고, 일부데이터만 학습하자)

  >
  >
  >* 업데이트를 빠르게 할수 있다. 
  >* 전체 데이터를 쓰지않아 잘못된 방향으로 업데이트 할 수 도 있다. 