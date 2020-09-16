import numpy as np


with open('log/ml-1m/dcan-add-trial-1.log', 'r') as f:
    lines = f.readlines()
    # print(lines[102].split(':'))
    auc = []
    logloss = []
    for line in lines:
        if line.split(':')[0] == "test auc":
            auc.append(float(line.split(':')[1]))
        if line.split(':')[0] == "test log_loss":
            logloss.append(float(line.split(':')[1]))
auc = np.array(auc).reshape((5, 1))
logloss = np.array(logloss).reshape((5, 1))

np.savetxt('log/ml-1m/dcan-add-auc.txt', auc.T, delimiter=',', fmt='%12f')
np.savetxt('log/ml-1m/dcan-add-logloss.txt', logloss.T, delimiter=',', fmt='%12f')