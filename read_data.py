import re
import json
import numpy

path = "Save_Data/CIF10_Relu_121.txt"
Acc = []
Precision = []
Recall = []
for i,data in enumerate(open(path, "r")):
    s_new = re.sub("'", '"', data)
    data = json.loads(s_new)
    if i%2!=0 :
        Acc.append(float(data["Acc"]))
        Precision.append(float(data["Precision"]))
        Recall.append(float(data["Recall"]))
Acc_mean = numpy.mean(Acc)

Precision_mean = numpy.mean(Precision)

Recall_std = numpy.mean(Recall)

print(Precision_mean)
print(Recall_mean)
