# 一个自用的半监督语义分割学习框架

特别感谢❤️ 以下仓库提供的代码：

[https://github.com/HiLab-git/SSL4MIS.git](https://)

[https://github.com/CoinCheung/pytorch-loss.git](https://)

[](https://)

## 目前支持训练方式：

1. 监督学习
2. 半监督学习

## 目前支持的网络

* U-Net

* [ ]  U-Net(ResNet as encoder)
* [ ]  Segformer

## 目前支持的半监督学习方案

* Semi-supervised Medical Image Segmentation through Dual-task Consistency（3D -> 2D）

* [ ]  Dual Consistency Enabled Weakly and Semi-Supervised Optic Disc and Cup Segmentation with Dual Adaptive Graph Convolutional Networks(输入的数据格式有点多，带实现)

# TODO List

* [ ]  调试DTC在多类别和2D场景中训练的细节
  * [ ]  多类别之间的标签问题，多个类别转化为多个任务，所以最后的logits概率归一化采用的是sigmiod，那么这里的每个类别的标签都该是1还是依次递增呢？
  * [ ]  RIM的问题
