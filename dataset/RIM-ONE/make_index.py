
with open('val.txt','r+') as f:
    idxs = f.read().splitlines()
    for idx in idxs:
        img_ = 'images_cropped/'+idx
        mask_ = 'my_gts_cropped/'+idx.replace('jpg','png')
        f.write(img_ + ' ' + mask_ + '\n')