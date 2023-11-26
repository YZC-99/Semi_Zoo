'''
1、读取whole_path
2、whole_path中每一行就是一条数据
3、随机的取857条数据存入到test.txt里面，剩下的存入到train_val.txt里面
4、读取train_val.txt
5、将train_val.txt随机读取200条到val.txt，剩下的放到train.txt

'''
whole_path = 'whole.txt'

import random

# Step 1: Read the whole_path file
with open(whole_path, 'r') as f:
    data = f.readlines()

# Step 2: Randomly select 857 data and save them in test.txt, while the remaining data are saved in train_val.txt
random.shuffle(data)
test_data = data[:857]
train_val_data = data[857:]

# Step 3: Save test_data in test.txt and train_val_data in train_val.txt
with open('test.txt', 'w') as f:
    f.writelines(test_data)

with open('train_val.txt', 'w') as f:
    f.writelines(train_val_data)

# Step 4: Read the train_val.txt
with open('train_val.txt', 'r') as f:
    train_val_data = f.readlines()

# Step 5: Randomly select 200 data from train_val.txt and save them in val.txt, while the remaining data are saved in train.txt
random.shuffle(train_val_data)
val_data = train_val_data[:200]
train_data = train_val_data[200:]

# Save val_data in val.txt and train_data in train.txt
with open('val.txt', 'w') as f:
    f.writelines(val_data)

with open('train.txt', 'w') as f:
    f.writelines(train_data)

print("Data splitting completed successfully.")