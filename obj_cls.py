object_clsname = [x.strip() for x in list(open('/home/ubuntu/hzy/pythia/data/objects_vocab.txt','r'))]
object_clsname = ['background'] + object_clsname
print(object_clsname[0])
