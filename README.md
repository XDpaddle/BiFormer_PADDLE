# BiFormer_PADDLE

## 模型简介
BiFormer的核心是提出了一种名为Bi-Level Routing Attention（BRA）的新型动态稀疏注意力机制。这种机制通过两级路由来实现对计算资源的灵活分配，同时保持对内容的敏感性。在BRA中，首先在粗略的区域级别过滤掉与查询不相关的键值对，然后在剩余的候选区域（即路由区域）中应用细粒度的令牌到令牌的注意力。这种设计使得BiFormer能够以一种查询自适应的方式关注一小部分相关令牌，而不受其他不相关令牌的干扰，从而在保持良好性能的同时，显著提高了计算效率。BiFormer的架构采用了四阶段金字塔结构，每个阶段都由一系列BiFormer块组成，这些块通过3x3深度卷积隐式编码相对位置信息，然后应用BRA模块和MLP模块来建模跨位置关系和每个位置的嵌入。


## 使用方法
1.下载paddleclas，将BIFORMERMODEL复制到paddleclass根目录下。

2.ppcls\configs\ImageNet中包含yaml配置文件的BiFormer文件夹放入paddleclas对应位置（ppcls\configs\ImageNet）。

## 下载权重
权重文件链接：https://pan.baidu.com/s/19Lt3JcvX8tCu_cCJNDPS5Q?pwd=1111, 提取码：1111 


## 参考
本项目代码参考自：https://paperswithcode.com/paper/biformer-vision-transformer-with-bi-level.
