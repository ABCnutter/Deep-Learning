# 一个悄然萌生的计算机视觉DL开源项目

## 2023/5/27
1. #### 发布vision transformer backbone，支持隐藏层输出

改进pytorch官方的`vision transformer`， 使其能够输出不同深度的transformer encoder layer的隐藏层特征图，常适用于分割与检测任务，并且模型支持加载pytorch官方给出的vit预训练权重。

可自由指定encoder layer的数目（对于图像分割、检测任务，常使用**4层encoder  layer**）以及**多头注意力的head数目**。

固定图像输入尺寸为**224*224**。

支持的vit模型有：

-   vit_b_16
-   vit_b_32
-   vit_l_16
-   vit_l_32
-   vit_h_14

## 2023/5/28

1. #### 发布常用的卷积神经网络注意力机制

支持的注意力机制有：

-   SENet
-   SKNet
-   SCSE
-   ECANet
-   CBAM

2. #### 发布通用conv2dBNRelu集成Block, 支持轻量化卷积

类似于torch的**Conv2dNormActivation**模块，属于其子集，不过本次发布**支持将普通2d卷积替换为深度可分离2d卷积**（**depth_wise_separable_conv**，引用自轻量型网络mobilenet，主要分为两个过程，分别为逐通道卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution））

3. #### 发布深监督模块

深监督常搭配backbone输出的多尺度、多深度的中间隐藏层特征图使用，在深度神经网络的某些中间隐藏层加了一个辅助的分类器作为一种网络分支来对主干网络进行监督的技巧，用来解决深度神经网络训练梯度消失和收敛速度过慢等问题。

4. #### 发布激活函数集成模块

参考自pytorch-image-models仓库。