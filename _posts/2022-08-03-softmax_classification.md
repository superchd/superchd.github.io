---
layout: single
title: "Softmax Classification"
categories : Artificial_Intelligence
tag: [Artificial_Intelligence, softmax, cross_entropy]
toc: true
author_profile : false
use_math: true
sidebar:
     nav: "docs"

---

## Softmax

> 의미 
>
> * 소프트맥스(softmax)함수는 모델의 출력을 확률로 해석할수있게 변환해주는 연산이다.
> * 분류문제에서 특정클래스 k에 속하는지 아닌지를 판단할때 사용된다. 
> * 즉, 분류문제 -> 선형모델 + softmax 함수로 푼다.
> * 그렇다면, **등장한 배경**은?
> * 단순히 추론만 한다면 원핫벡터를 이용해 단순하게 접근하지만, 일반적인 경우 선형모델로 출력한 값이 확률벡터가 아닌경우가 굉장히 많다. 이때, softmax를 이용해 출력값을 변환해 분류문제로 푼다.

## Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# For reproducibility
torch.manual_seed(1)

```

## 

Convert numbers to probabilities with softmax.
$$
P(class=i) = \frac{e^i}{\sum e^i}
$$

```python
z = torch.FloatTensor([1, 2, 3])
# PyTorch has a `softmax` function.
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
# tensor([0.0900, 0.2447, 0.6652])
```

* 원래 [0, 0, 1]로 값이 나와야할 max 함수는 softmax 함수를 거치면, 합이 1이되는 tensor([0.0900, 0.2447, 0.6652]) 로 나온다.(Since they are probabilities, they should add up to 1. Let's do a sanity check.)
* tensor의 성분을 각각 순서대로 가위, 바위, 보라고 하면 P (보 | 가위) = 0.6652 ? 이해가 안된다.



## Cross Entropy

* 정의 : 두개의 확률분포가 있을때, 두 확률분포가 얼마나 비슷한지를 나타낼 수 있는 수치
* <img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 5.16.30.png" alt="스크린샷 2022-08-03 오후 5.16.30" style="zoom:50%;" />
* 확률분포 p에서 샘플링해서 구한 값들을 확률분포 q에 넣고, log를 씌운값의 평균 H(P, Q)
* 만약, 철수가 가위를 낸다음 무엇을 낼까? 에 대한 확률분포를 구할때, cross entropy를 구해 이를 최소화 하도록 하면, 우리는 Q2 -> Q1 -> P로 갈 수 있다. 점점 p에 근사하게 된다. 

## Cross Entropy Loss (Low-level)

For multi-class classification, we use the cross entropy loss.


$$
L = \frac{1}{N} \sum - y \log(\hat{y})
$$
where $\hat{y}$ is the predicted probability and $y$ is the correct probability (0 or 1).



```python
z = torch.rand(3, 5, requires_grad=True)

hypothesis = F.softmax(z, dim=1)
print(hypothesis)
'''
  tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
					[0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
      		[0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward>)
'''
# 임의의 정답 샘플을 추출
y = torch.randint(5, (3,)).long()
print(y)
# tensor([0, 2, 1]), 이제 Index 값을 one - hot vector로 표현

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

'''
    tensor([[1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0.]])
'''
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
# tensor(1.4689, grad_fn=<MeanBackward1>)

# Low level
torch.log(F.softmax(z, dim=1))
'''
    tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
            [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
            [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]], grad_fn=<LogBackward>)
'''
# High level
F.log_softmax(z, dim=1)
# pytorch 에서는 다음과 같이 간단하게 나타낼 수 있음 
    tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
            [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
            [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]],
           grad_fn=<LogSoftmaxBackward>)

  

```



## Training with Low-level Cross Entropy Loss

```python
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# 4차원의 벡터를 받아서 어떤 class인지 예측하기를 원함

# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산 (1)
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1) # or .mm or @
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()

    # Cost 계산 (2) // cross entropy로 쉽게 계산하는 경우 
    z = x_train.matmul(W) + b # or .mm or @
    cost = F.cross_entropy(z, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```





### summary

* Binary classfication -> sigmoid 함수 사용
* CE -> softmax를 사용 





# Tips



## Maximum Likelihood Estimation(MLE)

* Observation을 가장 잘 설명하는 $\theta$ 를 찾아내는 과정 
* 주어진 데이터에서 과도하게 fitting 되어버린 상황 -> overfitting , 주어진 데이터에서 그 데이터를 가장 잘 설명하는 확률분포 함수를 찾다 보니까 생기는 일 
* 그렇다면 이 Overfitting을 어떻게 없앨 수 있을까? 
* overfitting이 크게 일어나기전에 stop하고 pick theta
* <img src="/Users/hyundae/Library/Application Support/typora-user-images/스크린샷 2022-08-03 오후 10.17.06.png" alt="스크린샷 2022-08-03 오후 10.17.06" style="zoom: 33%;" />



## Regularization

>* early stopping : validation loss 가 더이상 낮아지지 않을때
>* reducing network size
>* weight decay
>* Dropout
>* batch nomarlization 



## Basic approach to DNN

* make a neural network architecture
* train and check that model is over-fitted
* Repeat from step 2



## Example


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```


```python
# For reproducibility
torch.manual_seed(1)
```


    <torch._C.Generator at 0x7f0708f8ffb0>

## Training and Test Datasets


```python
x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]
                            ])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])
```


```python
x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])
```

* 같은 데이터 -> 같은 분포로 부터 얻어진 데이터 

## Model


```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
    def forward(self, x):
        return self.linear(x)
```


```python
model = SoftmaxClassifierModel()
```


```python
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)
```


```python
def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.cross_entropy(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```


```python
def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print('Accuracy: {}% Cost: {:.6f}'.format(
         correct_count / len(y_test) * 100, cost.item()
    ))
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 2.203667
    Epoch    1/20 Cost: 1.199645
    Epoch    2/20 Cost: 1.142985
    Epoch    3/20 Cost: 1.117769
    Epoch    4/20 Cost: 1.100901
    Epoch    5/20 Cost: 1.089523
    Epoch    6/20 Cost: 1.079872
    Epoch    7/20 Cost: 1.071320
    Epoch    8/20 Cost: 1.063325
    Epoch    9/20 Cost: 1.055720
    Epoch   10/20 Cost: 1.048378
    Epoch   11/20 Cost: 1.041245
    Epoch   12/20 Cost: 1.034285
    Epoch   13/20 Cost: 1.027478
    Epoch   14/20 Cost: 1.020813
    Epoch   15/20 Cost: 1.014279
    Epoch   16/20 Cost: 1.007872
    Epoch   17/20 Cost: 1.001586
    Epoch   18/20 Cost: 0.995419
    Epoch   19/20 Cost: 0.989365



```python
test(model, optimizer, x_test, y_test)
```

    Accuracy: 0.0% Cost: 1.425844


## Learning Rate

Gradient Descent 에서의 $\alpha$ 값

`optimizer = optim.SGD(model.parameters(), lr=0.1)` 에서 `lr=0.1` 이다

learning rate이 너무 크면 diverge 하면서 cost 가 점점 늘어난다 (overshooting).


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e5)
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 1.280268
    Epoch    1/20 Cost: 976950.812500
    Epoch    2/20 Cost: 1279135.125000
    Epoch    3/20 Cost: 1198379.000000
    Epoch    4/20 Cost: 1098825.875000
    Epoch    5/20 Cost: 1968197.625000
    Epoch    6/20 Cost: 284763.250000
    Epoch    7/20 Cost: 1532260.125000
    Epoch    8/20 Cost: 1651504.000000
    Epoch    9/20 Cost: 521878.500000
    Epoch   10/20 Cost: 1397263.250000
    Epoch   11/20 Cost: 750986.250000
    Epoch   12/20 Cost: 918691.500000
    Epoch   13/20 Cost: 1487888.250000
    Epoch   14/20 Cost: 1582260.125000
    Epoch   15/20 Cost: 685818.062500
    Epoch   16/20 Cost: 1140048.750000
    Epoch   17/20 Cost: 940566.500000
    Epoch   18/20 Cost: 931638.250000
    Epoch   19/20 Cost: 1971322.625000


learning rate이 너무 작으면 cost가 거의 줄어들지 않는다.


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-10)
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 3.187324
    Epoch    1/20 Cost: 3.187324
    Epoch    2/20 Cost: 3.187324
    Epoch    3/20 Cost: 3.187324
    Epoch    4/20 Cost: 3.187324
    Epoch    5/20 Cost: 3.187324
    Epoch    6/20 Cost: 3.187324
    Epoch    7/20 Cost: 3.187324
    Epoch    8/20 Cost: 3.187324
    Epoch    9/20 Cost: 3.187324
    Epoch   10/20 Cost: 3.187324
    Epoch   11/20 Cost: 3.187324
    Epoch   12/20 Cost: 3.187324
    Epoch   13/20 Cost: 3.187324
    Epoch   14/20 Cost: 3.187324
    Epoch   15/20 Cost: 3.187324
    Epoch   16/20 Cost: 3.187324
    Epoch   17/20 Cost: 3.187324
    Epoch   18/20 Cost: 3.187324
    Epoch   19/20 Cost: 3.187324


적절한 숫자로 시작해 발산하면 작게, cost가 줄어들지 않으면 크게 조정하자.


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 1.341573
    Epoch    1/20 Cost: 1.198802
    Epoch    2/20 Cost: 1.150877
    Epoch    3/20 Cost: 1.131977
    Epoch    4/20 Cost: 1.116242
    Epoch    5/20 Cost: 1.102514
    Epoch    6/20 Cost: 1.089676
    Epoch    7/20 Cost: 1.077479
    Epoch    8/20 Cost: 1.065775
    Epoch    9/20 Cost: 1.054511
    Epoch   10/20 Cost: 1.043655
    Epoch   11/20 Cost: 1.033187
    Epoch   12/20 Cost: 1.023091
    Epoch   13/20 Cost: 1.013356
    Epoch   14/20 Cost: 1.003968
    Epoch   15/20 Cost: 0.994917
    Epoch   16/20 Cost: 0.986189
    Epoch   17/20 Cost: 0.977775
    Epoch   18/20 Cost: 0.969660
    Epoch   19/20 Cost: 0.961836


## Data Preprocessing (데이터 전처리)

데이터를 zero-center하고 normalize하자.


```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

$$ x'_j = \frac{x_j - \mu_j}{\sigma_j} $$

여기서 $\sigma$ 는 standard deviation, $\mu$ 는 평균값 이다.


```python
mu = x_train.mean(dim=0)
```


```python
sigma = x_train.std(dim=0)
```


```python
norm_x_train = (x_train - mu) / sigma
```


```python
print(norm_x_train)
```

    tensor([[-1.0674, -0.3758, -0.8398],
            [ 0.7418,  0.2778,  0.5863],
            [ 0.3799,  0.5229,  0.3486],
            [ 1.0132,  1.0948,  1.1409],
            [-1.0674, -1.5197, -1.2360]])


Normalize와 zero center한 X로 학습해서 성능을 보자


```python
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
```


```python
model = MultivariateLinearRegressionModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```


```python
train(model, optimizer, norm_x_train, y_train)
```

    Epoch    0/20 Cost: 29785.091797
    Epoch    1/20 Cost: 18906.164062
    Epoch    2/20 Cost: 12054.674805
    Epoch    3/20 Cost: 7702.029297
    Epoch    4/20 Cost: 4925.733398
    Epoch    5/20 Cost: 3151.632568
    Epoch    6/20 Cost: 2016.996094
    Epoch    7/20 Cost: 1291.051270
    Epoch    8/20 Cost: 826.505310
    Epoch    9/20 Cost: 529.207336
    Epoch   10/20 Cost: 338.934204
    Epoch   11/20 Cost: 217.153549
    Epoch   12/20 Cost: 139.206741
    Epoch   13/20 Cost: 89.313782
    Epoch   14/20 Cost: 57.375462
    Epoch   15/20 Cost: 36.928429
    Epoch   16/20 Cost: 23.835772
    Epoch   17/20 Cost: 15.450428
    Epoch   18/20 Cost: 10.077808
    Epoch   19/20 Cost: 6.633700


## Overfitting

너무 학습 데이터에 한해 잘 학습해 테스트 데이터에 좋은 성능을 내지 못할 수도 있다.

이것을 방지하는 방법은 크게 세 가지인데:

1. 더 많은 학습 데이터
2. 더 적은 양의 feature
3. **Regularization**

Regularization: Let's not have too big numbers in the weights


```python
def train_with_regularization(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)
        
        # l2 norm 계산
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)
            
        cost += l2_reg

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch+1, nb_epochs, cost.item()
        ))
```


```python
model = MultivariateLinearRegressionModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
train_with_regularization(model, optimizer, norm_x_train, y_train)
```

    Epoch    1/20 Cost: 29477.810547
    Epoch    2/20 Cost: 18798.513672
    Epoch    3/20 Cost: 12059.364258
    Epoch    4/20 Cost: 7773.400879
    Epoch    5/20 Cost: 5038.263672
    Epoch    6/20 Cost: 3290.066406
    Epoch    7/20 Cost: 2171.882568
    Epoch    8/20 Cost: 1456.434570
    Epoch    9/20 Cost: 998.598267
    Epoch   10/20 Cost: 705.595398
    Epoch   11/20 Cost: 518.073608
    Epoch   12/20 Cost: 398.057220
    Epoch   13/20 Cost: 321.242920
    Epoch   14/20 Cost: 272.078247
    Epoch   15/20 Cost: 240.609131
    Epoch   16/20 Cost: 220.465637
    Epoch   17/20 Cost: 207.570602
    Epoch   18/20 Cost: 199.314804
    Epoch   19/20 Cost: 194.028214
    Epoch   20/20 Cost: 190.642014







## MNIST

* 손으로 쓰여진 숫자들 
* 