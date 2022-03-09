import numpy as np
test=np.load('/home/ubuntu/hzy/pythia/data/textvqa_gcy/test/0a5be8c3ad1036d4_info.npy',encoding = "latin1",allow_pickle=True)  #加载文件
doc = open('contrast2.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #将打印内容写入文件中