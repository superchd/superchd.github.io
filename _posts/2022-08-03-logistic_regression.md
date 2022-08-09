---
layout: single
title: "Logistic Regression"
categories : Artificial_Intelligence
tag: [Artificial_Intelligence, logistic_regression, cross_entropy]
toc: true
author_profile : false
use_math: true
sidebar:
     nav: "docs"
---

## Logistic Regression

* binary classification 문제 (m x d의 데이터 x가 있을때, 이 벡터가 0 또는 1중 어디에 더 가까운지 판별하는것)
* 데이터 x와 가중치 행렬 w를 곱해서 판단 (sigmoid 함수를 이용한다는데?? )
* <img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 3.30.42.png" alt="스크린샷 2022-08-03 오후 3.30.42" style="zoom:25%;" />

### Hypothesis

$$
H(X) = \frac{1}{1+e^{-W^T X}}
$$

### Cost


$$
cost(W) = -\frac{1}{m} \sum y \log\left(H(x)\right) + (1-y) \left( \log(1-H(x) \right)
$$

- If $y \simeq H(x)$, cost is near 0.

- If $y \neq H(x)$, cost is high.



### Weight Update via Gradient Descent


$$
W := W - \alpha \frac{\partial}{\partial W} cost(W)
$$


$\alpha$: Learning rate



## Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

```python
\# For reproducibility

torch.manual_seed(1)
# 똑같은 결과를 지원해주기 위해서 
```



## Training Data

```python
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```



## Computing the Hypothesis

$$
H(X) = \frac{1}{1+e^{-W^T X}}
$$



PyTorch has a `torch.exp()` function that resembles the exponential function.

```python
print('e^1 equals: ', torch.exp(torch.FloatTensor([1])))
```

We can use it to compute the hypothesis function conveniently.

```python
W = torch.zeros((2, 1), requires_grad=True)
# gradient 배울꺼야
b = torch.zeros(1, requires_grad=True)
# gradient 배울꺼야
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
# x_train.matmul(W) -> X * W
print(hypothesis)
print(hypothesis.shape)
```

 tensor([[0.5000],

​      [0.5000],

​      [0.5000],

​      [0.5000],

​      [0.5000],

​      [0.5000]], grad_fn=<MulBackward>)

  torch.Size([6, 1])

Or, we could use `torch.sigmoid()` function! This resembles the sigmoid function:

```python
print('1/(1+e^{-1}) equals: ', torch.sigmoid(torch.FloatTensor([1])))
1/(1+e^{-1}) equals:  tensor([0.7311])
# hypothesis는 sigmoid 모듈을 사용하면 훨씬 쉽다.
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
```



## Computing the Cost Function (Low-level)

$$
cost(W) = -\frac{1}{m} \sum y \log\left(H(x)\right) + (1-y) \left( \log(1-H(x) \right)
$$

We want to measure the difference between `hypothesis` and `y_train`.

For **one element**, the loss can be computed as follows:

```python
-(y_train[0] * torch.log(hypothesis[0]) + (1 - y_train[0]) * torch.log(1 - hypothesis[0]))
```

  tensor([0.6931], grad_fn=<NegBackward>)



To compute the losses for the **entire batch**, we can simply input the entire vector.

```python
losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))

print(losses)
```

  tensor([[0.6931],

​      [0.6931],

​      [0.6931],

​      [0.6931],

​      [0.6931],

​      [0.6931]], grad_fn=<NegBackward>)



Then, we just `.mean()` to take the mean of these individual losses.

```python
cost = losses.mean()
print(cost)
```

  tensor(0.6931, grad_fn=<MeanBackward1>)



## Computing the Cost Function with `F.binary_cross_entropy`

In reality, binary classification is used so often that PyTorch has a simple function called `F.binary_cross_entropy` implemented to lighten the burden.

괜히 뻘짓하지말고 library 쓰자는 말 ㅎ

```python
F.binary_cross_entropy(hypothesis, y_train)
```

 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)



## Training with Low-level Binary Cross Entropy Loss

```python
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
  # Cost 계산
  hypothesis = torch.sigmoid(x_train.matmul(W) + b) # or .mm or @
  cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()
  # cost로 H(x) 개선
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()
  # 100번마다 로그 출력
  if epoch % 100 == 0:
		print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

```



  Epoch  0/1000 Cost: 0.693147

  Epoch 100/1000 Cost: 0.134722

  Epoch 200/1000 Cost: 0.080643

  Epoch 300/1000 Cost: 0.057900

  Epoch 400/1000 Cost: 0.045300

  Epoch 500/1000 Cost: 0.037261

  Epoch 600/1000 Cost: 0.031673

  Epoch 700/1000 Cost: 0.027556

  Epoch 800/1000 Cost: 0.024394

  Epoch 900/1000 Cost: 0.021888

  Epoch 1000/1000 Cost: 0.019852



### 평가하는 방법은 아직 미흡하다...



