# 本次作业所做工作：



1. 复习老师上课提到的RNN，LSTM，GRU，BOW等针对nlp领域的模型，完善notebook中的模型，一共采用4种模型进行了对比，并且自己实现了一个回调函数，能够记录训练过程中在测试集上表现最好的模型参数。

   1. RNN 测试集正确率: 0.5018，训练过程可视化：<img src="/Users/vincent/Development/ml_sysu/aistudio/hw6/img/RNN.png" alt="RNN" style="zoom:50%;" />

   2. LSTM 测试集正确率：0.8476，训练过程可视化：

      <img src="/Users/vincent/Development/ml_sysu/aistudio/hw6/img/LSTM.png" alt="LSTM" style="zoom:50%;" />

   3. GRU 测试集正确率: 0.8668, 训练过程可视化：

      <img src="/Users/vincent/Development/ml_sysu/aistudio/hw6/img/GRU.png" alt="GRU" style="zoom:50%;" />

   4. BOW 测试集正确率：0.8640，训练过程可视化：

   <img src="/Users/vincent/Development/ml_sysu/aistudio/hw6/img/bow.png" alt="bow" style="zoom:50%;" />

2. 自定义 paddle model 中的回调函数，保存训练过程中，测试集上面表现最好的参数，保存为best_model。

代码:

```python
# 自定义 存储最优模型参数 回调函数：
class Best_model_checkpoint(paddle.callbacks.Callback):
    def __init__(self, baseline=0, save_dir=None):
        self.baseline = baseline
        self.save_dir = save_dir

    def on_eval_end(self, logs=None):
        acc = logs['acc']
        if acc > self.baseline:
            self.baseline = acc
            path = '{}/best_model'.format(self.save_dir)
            print('save checkpoint at {}'.format(path))
            self.model.save(path)
        print(logs)
```

## 结论：

在所以模型中，GRU的效果是最好的，正确率有0.8668。另一种最简单的模型，BOW效果也还不错。普通的RNN模型的正确率不太行，只有0.5018，就是在瞎猜。说明这次数据集里面的顺序不是特别的重要，而是里面关键情感词，起到了重要的作用。