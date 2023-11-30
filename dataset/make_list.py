import pickle

txt_path = './SEG/whole2059.txt'
with open(txt_path,'r') as f:
    ids = f.read().splitlines()
    name = [i.split(' ')[0].split('.')[0] for i in ids]
