import random
import os

# 创建一个新的文件夹bootstrapping1000
folder_name = 'bootstrapping2000'
os.makedirs(folder_name, exist_ok=True)

# Step 1: 读取整个数据集
with open('whole2059.txt', 'r') as f:
    data = f.readlines()

# 进行1000次bootstrap重采样
for i in range(2000):
    bootstrap_data = random.choices(data, k=len(data))

    # 创建新的文件夹，并将抽样得到的数据存放在文件夹中
    new_folder_name = os.path.join(folder_name, f'bootstrap{i+1}')
    os.makedirs(new_folder_name, exist_ok=True)

    # 数据集划分
    test_data = bootstrap_data[:613]
    train_val_data = bootstrap_data[613:]

    # 保存数据集
    with open(os.path.join(new_folder_name, 'test.txt'), 'w') as f:
        f.writelines(test_data)

    with open(os.path.join(new_folder_name, 'train_val.txt'), 'w') as f:
        f.writelines(train_val_data)

    # Step 2: 读取train_val.txt
    with open(os.path.join(new_folder_name, 'train_val.txt'), 'r') as f:
        train_val_data = f.readlines()

    # Step 3: 随机选择200个数据作为val.txt，剩下的数据保存在train.txt
    random.shuffle(train_val_data)
    val_data = train_val_data[:140]
    train_data = train_val_data[140:]

    # 保存val_data和train_data
    with open(os.path.join(new_folder_name, 'val.txt'), 'w') as f:
        f.writelines(val_data)

    with open(os.path.join(new_folder_name, 'train.txt'), 'w') as f:
        f.writelines(train_data)

print("Bootstrap sampling completed successfully.")