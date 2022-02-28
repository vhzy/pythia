import numpy as np
test=np.load('/home/ubuntu/hzy/pythia/data/imdb/m4c_textvqa/imdb_val_ocr_en.npy',encoding = "latin1",allow_pickle=True)  #加载文件
doc = open('imdb_val_ocr_en.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #将打印内容写入文件中