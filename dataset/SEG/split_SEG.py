'''
将下面的这个分数据集改写为bootstrapping的思想：
然后新建一个文件夹为bootstrapping1000,存放1-1000的各个文件

'''
whole_path = 'whole2575_h5.txt'

import random

# Step 1: Read the whole_path file
with open(whole_path, 'r') as f:
    data = f.readlines()

# Step 2: Randomly select 857 data and save them in test.txt, while the remaining data are saved in train_val.txt
random.shuffle(data)
test_data = data[:613]
train_val_data = data[613:]

# Step 3: Save test_data in test.txt and train_val_data in train_val.txt
with open('test_h5.txt', 'w') as f:
    f.writelines(test_data)

with open('train_val_h5.txt', 'w') as f:
    f.writelines(train_val_data)

# Step 4: Read the train_val.txt
with open('train_val_h5.txt', 'r') as f:
    train_val_data = f.readlines()

# Step 5: Randomly select 200 data from train_val.txt and save them in val.txt, while the remaining data are saved in train.txt
random.shuffle(train_val_data)
val_data = train_val_data[:140]
train_data = train_val_data[140:]

# Save val_data in val.txt and train_data in train.txt
with open('val_h5.txt', 'w') as f:
    f.writelines(val_data)

with open('train_h5.txt', 'w') as f:
    f.writelines(train_data)

print("Data splitting completed successfully.")