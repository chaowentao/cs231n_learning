### tensorflow sigmoid_cross_entropy_with_logits 函数解释及公式推导
[tensorflow官方文档解释参考](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)

[pytorch--BCELoss](https://pytorch.org/docs/stable/nn.html?highlight=bceloss#torch.nn.BCELoss)
[pytorch--BCELoss解释参考](https://github.com/pytorch/pytorch/issues/751)

定义在[tensorflow/python/ops/nn_impl.py](https://www.tensorflow.org/code/stable/tensorflow/python/ops/nn_impl.py).

功能：计算在给定``logits``和``label``之间的sigmoid cross
entropy。测量离散分类任务中的概率误差，其中每个类是独立的，而不是相互排斥的。
例如，可以执行多标签分类，其中图片可以同时包含大象和狗。

通俗的解释是在进行分类任务时，计算我们得到的``logits``值（也有说scores分数值）与期望值（类别标签）``label``之间的差别。

```
tf.nn.sigmoid_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```
计算公式：
![交叉熵](https://img-blog.csdn.net/20170606182819132)
这就是标准的Cross Entropy算法实现，对得到的值logits进行sigmoid激活，保证取值在0到1之间，然后放在交叉熵的函数中计算Loss。

公式推导：
为了简便, 让``x = logits``, ``z = labels``. 上述公式可以写为：
```
  z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
= (1 - z) * x + log(1 + exp(-x))
= x - x * z + log(1 + exp(-x))
```
对于x < 0，为了避免exp(-x)中的溢出，我们重新制定了上面的公式:
```
  x - x * z + log(1 + exp(-x))
= log(exp(x)) - x * z + log(1 + exp(-x))
= log(exp(x)) - x * z + log((1 + exp(x)) / exp(x))
= log(exp(x)) - x * z + log(1 + exp(x) - log(exp(x))
= - x * z + log(1 + exp(x))
```
因此，为了保证稳定性和避免溢出，实现使用了这个等价的公式
```
max(x, 0) - x * z + log(1 + exp(-abs(x)))
```
``logits``和``labels``必须具有相同的类型和形状。
参数:
```
sentinel: 用于防止位置参数。内部，请勿使用。
labels: 与logits相同类型和形状的张量。
logits: 浮点型张量，32或64。
name: 操作的名称(可选)。
```
返回值：
```
一种形状与logits张量相同的张量，具有分量逻辑损失。
```

