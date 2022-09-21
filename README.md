# Sequential-Recommendation
Final Project of Deep Learning

代码目录结构如下：
``` python

# 数据集 
|--- dataset 
   |--- ... 
   
# 模型搭建和训练代码 
|--- 0 GRU4REC.ipynb 
|--- 1 SRGNN.ipynb 
|--- 2 NARM.ipynb 
|--- 3 GRUwATT.ipynb 
|--- 4 GGNNwATT.ipynb 

# 输出最终结果 
|--- get_final_result.ipynb 

# 数据处理 
|--- process_data.py
```


* `0 GRU4REC.ipynb` ：实现了GRU4REC模型的搭建和训练，保存最佳模型为 gru4rec.pt 
* `1 SRGNN.ipynb` ：实现了SRGNN模型的搭建和训练
* `2 NARM.ipynb` ：实现了NARM模型的搭建和训练
* `3 GRUwATT.ipynb` ：实现了GRUwATT模型的搭建和训练，保存最佳模型为 GRUwATT.pt 
* `4 GGNNwATT.ipynb` ：实现了GGNNwATT模型的搭建和训练
每个文件的编写逻辑基本相同，分为模型搭建、模型训练和保存模型三个部分，其中模型训练部分包括评价指标的
定义、模型训练打包（ ModelTrain 类的定义）、超参数设置和训练过程。

`get_final_result.ipynb` 用于读取保存的模型，并输出最后的结果文件，保存在 results 文件夹中

`process_data.py` ：
* 对数据进行预处理，除去总出现次数小于5的item、只有一个item的session并将item编码
* 定义了 TrainDataset 和 MyDataset 两个类，继承了PyTorch的 Dataset 类 TrainDataset 和 TestDataset 类的区别在于其 __getitem__() 方法返回的label不同，具体区别为：
  * TrainDataset ：返回的label除了数据集中给定的label外，还有该session除了第一个item外剩下的item序列，相当于做了一种适合RNN的data augmentation（灵感来源于最后一次作业的文本生成任务）
  例：当session = [1, 2, 3, 4]、label=[5]时， __getitem__() 返回的样本是X = [1, 2, 3, 4]和y = [2, 3, 4, 5]
  * MyDataset ：返回的label就是数据集中的label，即单个item
> 需要注意的是，由于NARM和SRGNN模型中的attention机制需要取最后一个时间步的特征，所以得到的输出形状是[batch_size, embedding_size]，因此并不适合用 TrainDataset 类进行训练。
* 定义了 get_dataloader() 函数，返回一个PyTorch DataLoader 类的对象，用于在训练和测试时获取batch数据。在 get_dataloader() 中进行了以下两个操作：
  * 设置了 DataLoader 的 batch_sampler 属性，使得loader在抽样时，会先将样本按照session的长度进行排序，使batch中每个样本session的真实长度都相近
  * 设置了 DataLoader 的 collate_fn 属性，令loader为batch中的样本进行零填充，使得返回的batch中，每个样本的session长度都相等，且等于该batch中最长session的长度
