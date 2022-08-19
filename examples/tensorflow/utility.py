import tensorflow as tf 
from sklearn.model_selection import train_test_split
import os 
from tensorflow import keras
import random
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def get_model():
	model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
	model.compile(
	optimizer=keras.optimizers.RMSprop(),  # Optimizer
	# Loss function to minimize
	loss=keras.losses.SparseCategoricalCrossentropy(),
	# List of metrics to monitor
	metrics=[keras.metrics.SparseCategoricalAccuracy()],
		)
	# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
	return model

def load_data():
	trainset, testset = tf.keras.datasets.cifar10.load_data()
	num_examples = {"trainset": len(trainset[0]), "testset": len(testset[0])}
	return trainset, testset, num_examples


def load_partition(idx: int):
	"""Load 1/10th of the training and test data to simulate a partition."""
	trainset, testset, num_examples = load_data()
	n_train = int(num_examples["trainset"] / 10)
	n_test = int(num_examples["testset"] / 10)
	X,y = trainset
	X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=90, random_state=random.randint(1,100))
	train_parition = (X_train,y_train)
	test_parition = (X_test,y_test)
	return (train_parition, test_parition)

def train(net, trainloader, valloader, epochs, device):
	"""Train the network on the training set."""
	print("Starting training...")
	X_train,y_train =  trainloader
	history =  net.fit(X_train,y_train,epochs =1,batch_size = 16,validation_data = valloader)
	results = {
	"train_loss": history.history['loss'][0],
	"train_accuracy": history.history['sparse_categorical_accuracy'][0],
	"val_loss": history.history['val_loss'][0],
	"val_accuracy": history.history['val_sparse_categorical_accuracy'][0],
	}
	return results

def test(net, testloader, steps: int = None, device: str = "cpu"):
	"""Validate the network on the entire test set."""
	x_test,y_test =  testloader
	history =  net.fit(x_test,y_test,epochs =1)
	loss = history.history['loss']
	accuracy = history.history['sparse_categorical_accuracy']
	return loss, accuracy

def get_model_params(model):
	"""Returns a model's parameters."""
	return model.get_weights()
	# return [val.cpu().numpy() for _, val in model.state_dict().items()]