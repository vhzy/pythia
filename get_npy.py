import numpy as np
test=np.load('/home/ubuntu/hzy/pythia/data/m4c_textvqa_ocr_en_frcn_features/train_images/f441f29812b385ad_info.npy',encoding = "latin1",allow_pickle=True)  #加载文件
doc = open('contrast9.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #将打印内容写入文件中