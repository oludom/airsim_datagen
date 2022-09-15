

import numpy as np

def dist(a,b):
    return np.linalg.norm(a-b)

def read_file_tum(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list = [l[1:] for l in list if len(l) > 1]
    return np.array(list).astype(np.float)



data = read_file_tum("trajectory.tum")
print(data.shape)

dis = []
for i in range(1, data.shape[0]):
    dis.append(dist(data[i-1, 0:3], data[i, 0:3]))

print(np.sum(dis))
print("distance travelled: " + str(np.sum(dis)))

tid = 347153 / 1000
vel = np.sum(dis) / tid
print(f"avg velocity: {vel}")