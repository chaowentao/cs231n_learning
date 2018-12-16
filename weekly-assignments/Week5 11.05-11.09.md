**学习时长**
11.05-11.09日

1. 理解 CNN 中的卷积
- slides: lecture05（ http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf ）
- 观看视频 p11, p12 
2. 理解 CNN 中的 pooling
- slides: lecture05（ http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf ）
- 观看视频 p13
- 学习卷积神经网络笔记 （ https://zhuanlan.zhihu.com/p/22038289?refer=intelligentunit ）
3. 完成 CNN 的第一个应用练习，人脸关键点检测
- 阅读 facial keypoint 小项目（https://github.com/udacity/P1_Facial_Keypoints）
- 参考代码（https://github.com/L1aoXingyu/P1_Facial_Keypoints）


**作业**
1. 思考一下卷积神经网络对比传统神经网络的优势在哪里？为什么更适合处理图像问题，知识星球打卡上传

- 共性：每个hidden neuron对应的weights|filter都在提炼学习特征，越深层filter提炼越抽象复杂的特征
- CNN重视空间信息的保留（DNN，否）
- CNN眼里只有局部小特征（DNN，只有全局大特征）
- CNN：理解图片，是一步一步，一块一块，比对小特征的过程（DNN：全局性大特征比对）

四句话

从input，filter，feature map上对比DNN与CNN在的实质区别

**input**

> DNN input 打破空间信息，压缩成一个长vector，作为整体被学习背后数据特征
> CNN input 保留空间信息，截取成多个局部小图片，分别被学习背后数据特征

**filter**

> DNN filter|weights 无需空间信息，对应DNN input的长vector，提炼全局图片中的特征（正向传递）和更新学到的特征（反向传递）
> CNN filter|weights 重视空间信息，对应每一个局部小图片，提炼小图片中的特征（正向传递）和更新学到的特征（反向传递）

**output or feature map** 

> DNN的weights完成工作后，生成单一scalar，是对全局图片信息，做全局特征提炼和学习后的感想和评估
> CNN的filter完成工作后，生成一个matrix，是将所有局部小图片信息，分别做特征提炼和学习后的感想和评估后，在做空间结构信息的还原整合的结果


2. 完成 assignment2 中 ``FullyConnectedNets.ipynb``

**参看文献**

http://cs231n.github.io/neural-networks-3/#sgd%20f

[1] Tijmen Tieleman and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." COURSERA: Neural Networks for Machine Learning 4 (2012).

[2] Diederik Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.

[3] Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015.

[4] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer Normalization." stat 1050 (2016): 21.


**课程资料：**

课程主页： http://cs231n.stanford.edu /

course note： http://cs231n.github.io /

知乎翻译： https://zhuanlan.zhihu.com/p/21930884

推荐b站的视频观看  https://www.bilibili.com/video/av17204303/?p=3 

注册一个github账号: github.com
后续发布的一些project和exercise会在这个github下：https://github.com/orgs/sharedeeply/dashboard

配置环境：  https://github.com/sharedeeply/DeepLearning-StartKit