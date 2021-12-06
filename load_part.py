import numpy as np

train_x = np.loadtxt("train_x", dtype=int)
train_y = np.loadtxt("train_y", dtype=int)

#print(train_x[0])
#print(train_y[0])

# shuffle the train set
p = np.random.permutation(len(train_x))
train_x_sh, train_y_sh = train_x[p], train_y[p]

part_train_x = train_x_sh[:5000]
part_train_y = train_y_sh[:5000]

"""for i in range(5000):
    part_train_x = np.concatenate(part_train_x, [train_x_sh[i]])
    part_train_x = np.concatenate(part_train_x, [train_y_sh[i]])"""

print(train_x[0])
print("\n")
print(part_train_x[0])


np.savetxt("part_train_x", part_train_x, delimiter=' ', fmt='%i')
np.savetxt("part_train_y", part_train_y, fmt='%i')
