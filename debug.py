from datasets import load_dataset

# 加载数据集
dataset = load_dataset("cnn_dailymail", "3.0.0")

# 方法1：查看数据集结构
print("Dataset splits:", dataset.keys())
print("\nDataset info:")
print(dataset)

# 方法2：查看训练集的信息
train_dataset = dataset['train']
print("\nTrain dataset info:")
print(train_dataset.info.description)

# 方法3：查看数据集特征
print("\nDataset features:")
print(train_dataset.features)

# 方法4：查看样本数据
print("\nFirst example:")
example = train_dataset[0]
print("Article preview:", example['article'][:500])
print("\nHighlights:", example['highlights'])
print("\nID:", example['id'])
