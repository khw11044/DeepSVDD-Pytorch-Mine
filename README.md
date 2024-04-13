## CIFA10으로 Anomaly Detection - DeepSVDD 기반 


DeepSVDD가 2019년에 나온 논문으로 가장 대표적인 Anomaly detection 기법인데 

MNIST에 대해서는 뭐 98% 나오지만 CIFAR10에 대해서는 58~60% 정확도만 나온다. 

아래 paperwithcode를 참고하면 32위로 65.7%라고 한다. 

더 좋은 방법을 찾아야 한다. 

[paperwithcode: DeepSVDD-CIFAR10](https://paperswithcode.com/sota/anomaly-detection-on-one-class-cifar-10)


이 코드는 DeepSVDD-Pytorch 가장 대표적인 깃허브 코드에서 
visual studio code에서 간단하게 Run 버튼만 눌러도 실행하게 세팅하였다.

[Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch)


```python
CiFAR10

AUC
airplane : 57.82%
automobile : 60.60%
bird : 45.53% 
cat : 60.26%
deer : 54.63%
dog : 64.41%
frog : 57.58%
horse : 56.78%
ship : 76.06%
truck : 67.50%

```