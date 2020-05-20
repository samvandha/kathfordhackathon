from numpy import array
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import random

def split_seq(seq, n_steps):
    x,y = list(), list()
    for i in range(len(seq)):
        end_ix = i + n_steps
        if end_ix > (len(seq)-1):
            break
        seqx, seqy = seq[i:end_ix], seq[end_ix]
        x.append(seqx)
        y.append(seqy)
    return x,y

raw_seq = [580, 845, 1317, 2015, 2800, 4581, 6058, 7813,
     9823, 11950, 14553, 17391,20630, 24545, 28266, 31439, 34876,
     37552, 40553,43099, 45134, 59287, 64438, 67100, 69197, 71329,
     73332, 75184, 75700,76677, 77673, 78651, 79205, 80087, 80828, 
     81820, 83112,84615, 86604, 88585, 90443, 93016, 95314, 98425,
     102050, 106099, 109991,114381, 118948, 126948, 134576, 145483,
     156653, 169593,182490, 198238, 218822, 244933, 275597, 305036,
    337000, 378000, 422000, 471000, 531000, 596000, 663000, 723000,
    784000, 858000, 935000, 1010000]

n_steps = 12
x_axis = []
for i in range(len(raw_seq)):
    raw_seq[i] = raw_seq[i]/ 2000000
for i in range(len(raw_seq)):
    x_axis.append(i)

new_x_axis = len(raw_seq)-1
plt.plot(x_axis, raw_seq)   

x,y = split_seq(raw_seq, n_steps)
x = np.array(x)
y = np.array(y)
#print(x)
#print(y)
x_train = []
y_train = []
x_val = []
y_val = []
for i in range(len(x)):
    rand = random.random()
    if rand < 0.3:
        x_val.append(x[i])
        y_val.append(y[i])
    else:
        x_train.append(x[i])
        y_train.append(y[i])

x_val = np.array(x_val)
y_val = np.array(y_val)
x_train = np.array(x_train)
y_train = np.array(y_train)


##    print(x[i], y[i])
n_features = 1   
model = Sequential()
model.add(LSTM(50, activation = 'relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss = 'mse')
print("Model built")
n_features = 1
x = x.reshape((x.shape[0],x.shape[1], n_features))

x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], n_features))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))

model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 1000)
model.save("cases.h5")
#model = load_model('cases.h5')
x_axis_values = [new_x_axis]
y_axis_values = [raw_seq[new_x_axis]]
for i in range(10): 
    x_input = raw_seq[(len(raw_seq)-n_steps):]
    x_input = np.array(x_input)
    x_input = x_input.reshape((1,n_steps, n_features))
    new_y = np.array(model.predict(x_input, verbose = 0))
    print(new_y[0][0])
    raw_seq.append(new_y[0][0])
    #print("Len of raw seq:", len(raw_seq))
    new_x_axis += 1
    x_axis_values.append(new_x_axis)
    y_axis_values.append(new_y[0][0])
    
    #plt.plot(new_x_axis, new_y)
    print(new_y)

plt.plot(x_axis_values, y_axis_values)
plt.show()

