"""
进行了数据预处理：
* 除去总出现次数小于5的item
* 除去只有一个item的session
* 将item编码

定义了TrainDataset、MyDataset类和get_dataloader()
"""
import pandas as pd
import operator
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.model_selection import train_test_split

'''
# 本来想实现一下论文中的augmentation，然后发现老师给的数据集已经做过这一步了

def data_augment(df):
    session_list = []
    label_list = []
    for i in range(len(df)):
        sample = df.iloc[i]
        s = sample.session
        l = sample.label
        for j in range(1, len(s)):
            session_list.append(s[: j])
            label_list.append(s[j])
        session_list.append(s)
        label_list.append(l)

    aug_sample = pd.DataFrame({'session': session_list,
                               'label': label_list})
    return aug_sample

df = data_augment(data)
'''


def pre_process(file):
    # file = input('which file (Beauty/Cell) ? : ')
    data = pd.read_csv(f'dataset/Amazon_{file}/train_sessions.csv')
    test_data = pd.read_csv(f'dataset/Amazon_{file}/test_sessions.csv')

    # 把session字符串转换为列表
    data["session"] = data.session.apply(eval)
    test_data["session"] = test_data.session.apply(eval)

    # 计算每个item出现的总次数, 除去总出现次数小于5的item
    iid_counts = {}
    for s in range(len(data)):
        seq = data['session'][s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1
    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1), reverse=True)
    less_appear_items = [k[0] for k in sorted_counts if k[1] < 5]
    data = data[~data.session.isin(less_appear_items)]

    # 除去只有一个item的session
    data = data[data["session"].apply(len) >= 2]

    # 记录所有session中出现过的全部item
    item_set = set()

    def f(input):
        if isinstance(input, list):
            for item in input:
                item_set.add(item)
        else:
            item_set.add(input)

    data["session"].apply(f)
    data["label"].apply(f)
    test_data['session'].apply(f)
    print("")

    # 将item编码为整数，并对应地修改session数据
    item2index = {item_num: idx for (idx, item_num) in enumerate(item_set, start=1)}
    index2item = {idx: item_num for (idx, item_num) in enumerate(item_set, start=1)}

    def f1(input):
        if isinstance(input, list):
            return [item2index[input[i]] for i in range(len(input))]
        else:
            return item2index[input]

    data["session"] = data["session"].apply(f1)
    data["label"] = data["label"].apply(f1)
    return data, item_set, item2index, index2item


# 现在的session和label值都不是item id，而是编码（index）了
# data, item_set, item2index, index2item = pre_process('Beauty')
data, item_set, item2index, index2item = pre_process('Cell')
data = data.reset_index(drop=True)
train_data, test_data = train_test_split(data, test_size=0.2)

# =====================
# 数据加载
# ====================


class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        session = self.df.iloc[idx, 0]  # session i
        late_session = self.df.iloc[idx, 0][1:]  # session i除去第一个元素
        label = [self.df.iloc[idx, 1]]  # label i，以list形式返回
        return session, late_session + label

    def __len__(self):
        return len(self.df)


class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        # 获取session和label
        session = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        return session, label

    def __len__(self):
        return len(self.df)


def get_dataloader(batch_size, dataset, train=True):
    """
    给定batch_size和dataset，返回迭代器
    Param：
        dataset：TrainDataset类（train=True）或MyDataset类（train=False）
    Return：
        返回一个迭代器，每次返回一个batch的数据，包括X和y
        len(dataloader) =  n_sample / batch_size
    """

    class batch_sampler(Sampler):
        def __init__(self, dataset, batch_size):
            self.dataset = dataset

            # indices，记录每条样本的索引和session长度
            indices = [(i, len(s[0])) for i, s in enumerate(dataset)]
            random.shuffle(indices)
            # 以140*batch_size为一组, 排序后加入pooled_indices
            pooled_indices = []
            for i in range(0, len(indices), batch_size*140):
                curr_data_indices = sorted(indices[i: i + batch_size*140],
                                           key=lambda x: x[1],  # 按照session长度进行排序
                                           reverse=True)
                pooled_indices.extend(curr_data_indices)
            self.pooled_indices = [x[0] for x in pooled_indices]  # 取索引 i

            self.batch_size = batch_size

        def __iter__(self):
            for i in range(0, len(self.pooled_indices), self.batch_size):
                # 遍历上面分的每一块，每一块作为一个batch，返回对应的索引
                yield self.pooled_indices[i: i + self.batch_size]

        def __len__(self):
            return int(len(self.dataset) / self.batch_size)

    def collate_fn(batch):
        """
        用零填充，把每个batch中的样本调成等长
        """
        rec_list, target_list = [], []
        if train:
            for record in batch:
                rec_list.append(torch.tensor(record[0]))  # [0] -> X
                target_list.append(torch.tensor(record[1]))  # [1] -> y
            # 用0填充
            X = pad_sequence(rec_list, padding_value=0, batch_first=True)
            y = pad_sequence(target_list, padding_value=0, batch_first=True)
            return X, y
        else:
            for record in batch:
                rec_list.append(torch.tensor(record[0]))
                target_list.append(record[1])
                # test的target只有一个值，所以y不用填充，直接转成tensor就行
            X = pad_sequence(rec_list, padding_value=0, batch_first=True)
            y = torch.LongTensor(target_list)
            return X, y

    '''
    batch_sampler: the strategy to draw samples from the dataset, returns a batch of indices at a time
    collate_fn: merges a list of samples to form a mini-batch of Tensor
    '''
    loader = DataLoader(dataset=dataset,
                        batch_sampler=batch_sampler(dataset, batch_size),
                        collate_fn=collate_fn)

    return loader


