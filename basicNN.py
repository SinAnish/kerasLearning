
from keras.models import Sequential #linear stack of layers
from keras.layers import Dense		#regular densely connected neural network layer

import numpy						#for the math

numpy.random.seed(7)				#random initialisation but with a reproducible seed


#normal dataset loading and splitting into x and y 
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]


model = Sequential()								 	
model.add(Dense(12, input_dim=8, activation='relu')) 	# deep layers
model.add(Dense(8, activation='relu'))					# deep layers
model.add(Dense(1, activation='sigmoid'))				# final answer so we need binary output

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  #compile


model.fit(X, Y, epochs=150, batch_size=10)				#this just trains it

#evaluation
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
