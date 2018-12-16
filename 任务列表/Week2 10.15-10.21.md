**学习时长**
10.15-10.21日

1. 学习线性分类器[中 下], 损失函数和优化器
2. **slides:** lecture03（链接： http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture03.pdf ）

    观看视频 p7 和 p8，了解更多关于线性分类器，损失函数以及优化器的相关知识。
    
    学习线性分类笔记中下（链接： https://zhuanlan.zhihu.com/p/20945670?refer=intelligentunit ）和最优化笔记（链接： https://zhuanlan.zhihu.com/p/21360434?refer=intelligentunit ），了解 SVM 和梯度下降法

**作业**

1. 简述 KNN 和线性分类器的优劣

    **KNN**优势：
    1. 原理简单，算法容易实现
    2. 重新训练的代价较低，只需把新数据加入数据集中即可
    3. 计算时间和空间线性于训练集的规模，适用于小数据集
    4. 对于类别交叉或重叠较多的待分样本集来说，KNN方法较其他方法更为适合。
    
    **KNN**劣势：
    1. 计算量大， 特别是数据集较大时
    2. 计算速度慢，KNN算法只是记住训练数据，并没有进行学习，其他分类方法速度较快
    3. 当训练集中样本类别分布不均匀时（有的类别数据很多，有的很少），则可能会影响判别结果
    
    **线性分类器**优势：
    1. 原理简单，计算速度与类别个数相关，与数据集大小无关，准确率较高；
    2. 可以解决小样本情况下的机器学习问题，可以提高泛化性能
    3. 可以避免神经网络结构选择和局部极小点问题
    
    **线性分类器**劣势：
    1. 对缺失数据敏感。
    2. 不能解决非线性分类问题


2. (可选)学习[矩阵求导]( https://zhuanlan.zhihu.com/p/25063314)的方法
3. 完成assignment1 中 ``svm.ipynb``

**课程资料：**

课程主页： http://cs231n.stanford.edu /

course note： http://cs231n.github.io /

知乎翻译： https://zhuanlan.zhihu.com/p/21930884

推荐b站的视频观看  https://www.bilibili.com/video/av17204303/?p=3 

注册一个github账号: github.com
后续发布的一些project和exercise会在这个github下：https://github.com/orgs/sharedeeply/dashboard

配置环境：  https://github.com/sharedeeply/DeepLearning-StartKit