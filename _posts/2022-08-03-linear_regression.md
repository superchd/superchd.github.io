---
layout: single
title: "Linear Regression"
categories : Artificial_Intelligence
tag: [Artificial_Intelligence, tensor, linear_regression]
toc: true
author_profile : false
use_math: true
sidebar:
     nav: "docs"

---

### 목차

* 선행지식

  >
  >
  >* vector, matrix, tensor
  >* 1차원 -> vector, 2차원 -> matrix,  3차원 -> tensor를 의미하는것 같다. 
  >* <img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 12.33.28.png" alt="스크린샷 2022-08-03 오후 12.33.28" style="zoom:50%;" />

* Data definition

  ><img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 12.21.07.png" alt="스크린샷 2022-08-03 오후 12.21.07" style="zoom:50%;" />

  * 1시간 공부 -> 2점 , 2시간 공부 -> 4점, 3시간 공부 -> 6점 받았을때 이 데이터 set을 **training dataset** 이라고 한다. 4시간 공부했을때 몇점 나올지 예측하는것을 **test dataset**이라고 한다.

  <img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 12.23.48.png" alt="스크린샷 2022-08-03 오후 12.23.48" style="zoom: 33%;" />

  * 다음과 같이 pytorch에서는 입력, 출력 데이터로 표현할 수 있다. 

* Hypothesis -> 인공신경망의 구조 

  <img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 12.26.48.png" alt="스크린샷 2022-08-03 오후 12.26.48" style="zoom: 33%;" />

  

* Compute loss

  <img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 12.27.58.png" alt="스크린샷 2022-08-03 오후 12.27.58" style="zoom: 50%;" />

  * 다음과 같이 한줄로 간단하게 :)

* Gradient descent<img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 12.29.55.png" alt="스크린샷 2022-08-03 오후 12.29.55" style="zoom:50%;" />

* Full training code

​	<img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 12.31.09.png" alt="스크린샷 2022-08-03 오후 12.31.09" style="zoom:50%;" />