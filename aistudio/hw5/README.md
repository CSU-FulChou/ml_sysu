# 本次作业所做工作：



本次作业我采用了老师上课所讲的几个经典的卷积模型进行实验。总共尝试了 resnet18，resnet50，VGG等三个模型，以及不同的优化器，epoch，batch_size。代码见 notebook，部分实验结果图片见img文件夹

## 选择模型：

在这三个模型中，我首先都采用 50个epoch， lr=0.0001， 优化器为 Adam，并且在训练集上使用 paddle 自带的 transforms来进行数据增强 进行对照实验， 发现resnet18的正确率，会更优。

最优正确率分别是：

Resnet18： 验证集 0.9166667， 测试集0.9

Resnet50：验证集 0.8666667，测试集 0.8

VGG: 验证集  0.9083333611488342，测试集 0.9

**结果分析**：在训练过程中观察也发现，由于resnet50，层数特别深，以及训练的数据集又很小，很容易就过拟合训练集导致在 验证集上面的效果还不如resnet18

### 注意：我发现在使用 transfrom 进行图片的翻转和旋转的操作之后，不一定能够提高我们的模型正确率，至少在药材这个数据类型上，这个操作反而不利于我们的模型找到合适的pattern，所以后续的操作，均不采用 数据增强

## 选择训练的epoch

通过在每一个epoch训练结束之后进行 validation，可以发现模型基本上在50-60的epoch能够达到训练得到稳定结果，得到收敛。

这里就不去对比epoch的选择图像了，因为只需要训练足够多的次数，就能够看到何时 validation的正确率已经得到收敛，在我训练了150个epoch图中，发现其实模型在大于50次，就能够收敛，所以后续的使用，epoch都选择了60次。

## 选择优化器：

在resnet18模型中，我尝试了第二次作业中所列举的一些常用的优化器： 如 Adam，SGD，Adagrad，Momentum

Adam： 验证集正确率 0.9166667， 测试集正确率 0.9

SGD： 验证集正确率 0.875， 测试集正确率 0.8

Adagrad:  验证集正确率 0.9166667， 测试集正确率0.8

Momentum:  验证集正确率 0.89166665 ， 测试集正确率  0.9

综合下来 Adam 确实是一个比较好的优化器，这也难怪，大部分的算法默认采用这个优化器来进行学习~

## 选择 batch_size：

在上面实验的基础上，我采用不同的batch size 来看看哪个比较合适

尝试了 batch_size=16, 32, 48 分别得到如下训练结果：

batch_size=16: 验证集正确率 0.9166667， 测试集正确率 0.9

batch_size=32: 验证集正确率 0.875， 测试集正确率 0.8

batch_size=48: 验证集正确率 0.875， 测试集正确率 0.9

## 总结：

在上述的对比实验下，我最终选择模型 resnet18，以及相应的训练参数为 optimizer=Adam,  epoch=50,  lr=0.0001, batch_size=32

最终在验证集最好的正确率： 0.9166667  在测试集的正确率：0.9

多次实验表示，正确率由于随机种子的问题，或者是在paddle 后台的机器gpu分配改变了。在某一次训练的最终训练结果可能会不同，但是5次平均来说，正确率水平在0.9左右。这也和测试集的正确率基本吻合~