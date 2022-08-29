import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as F


def get_model():
	net = nn.Sequential()
	net.add(nn.Dense(256, activation="relu"))
	net.add(nn.Dense(64, activation="relu"))
	net.add(nn.Dense(10))
	net.collect_params().initialize()
	return net

def load_data():
	print("Download Dataset")
	mnist = mx.test_utils.get_mnist()
	batch_size = 100
	train_data = mx.io.NDArrayIter(
	mnist["train_data"], mnist["train_label"], batch_size, shuffle=True
	)
	val_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)

	num_examples = {"trainset": len(mnist["train_label"]), "testset": len(mnist["test_label"])}
	return train_data, val_data, num_examples


def load_partition(idx: int):
	"""Load 1/10th of the training and test data to simulate a partition."""
	assert idx in range(10)
	print("Download Dataset")
	mnist = mx.test_utils.get_mnist()
	batch_size = 100
	train_data_size = round(len(mnist['train_data'])/idx)
	##Validation on Test Data Size of 5% data
	test_data_size = round(len(mnist['test_data'])/95)
	train_indices = list(mx.gluon.data.RandomSampler(train_data_size))
	test_indices = list(mx.gluon.data.RandomSampler(test_data_size))
	train_X = mnist['train_data'][train_indices]
	train_y = mnist['train_label'][train_indices]
	test_X = mnist['test_data'][test_indices]
	test_y = mnist['test_label'][test_indices]
	test_parition = mx.io.NDArrayIter(
				test_X, test_y, batch_size, shuffle=True
				)
	train_parition = mx.io.NDArrayIter(
				train_X, train_y, batch_size, shuffle=True
				)
	return (train_parition, test_parition)


def train(net, trainloader, valloader, epoch, device: str = "cpu"):
	trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})
	trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.01})
	accuracy_metric = mx.metric.Accuracy()
	loss_metric = mx.metric.CrossEntropy()
	metrics = mx.metric.CompositeEvalMetric()
	for child_metric in [accuracy_metric, loss_metric]:
		metrics.add(child_metric)
	softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
	for i in range(epoch):
		trainloader.reset()
		num_examples = 0
		for batch in trainloader:
			data = gluon.utils.split_and_load(
				batch.data[0], ctx_list=device, batch_axis=0
			)
			label = gluon.utils.split_and_load(
				batch.label[0], ctx_list=device, batch_axis=0
			)
			outputs = []
			with ag.record():
				for x, y in zip(data, label):
					z = net(x)
					loss = softmax_cross_entropy_loss(z, y)
					loss.backward()
					outputs.append(z.softmax())
					num_examples += len(x)
			metrics.update(label, outputs)
			trainer.step(batch.data[0].shape[0])
		trainings_metric = metrics.get_name_value()
		print("Accuracy & loss at epoch %d: %s" % (i, trainings_metric))
	train_accuracy  = trainings_metric[0][1]
	train_loss  = trainings_metric[1][1]
	test_accuracy,test_loss = test(net,valloader)

	results = {
			"train_loss": train_loss,
			"train_accuracy": train_accuracy,
			"val_loss": test_loss,
			"val_accuracy": test_accuracy,
			}
	return results

def test(net, val_data, steps: int = None, device: str = "cpu"):
    accuracy_metric = mx.metric.Accuracy()
    loss_metric = mx.metric.CrossEntropy()
    metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [accuracy_metric, loss_metric]:
        metrics.add(child_metric)
    val_data.reset()
    num_examples = 0
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=device, batch_axis=0)
        label = gluon.utils.split_and_load(
            batch.label[0], ctx_list=device, batch_axis=0
        )
        outputs = []
        for x in data:
            outputs.append(net(x).softmax())
            num_examples += len(x)
        metrics.update(label, outputs)
    return metrics.get_name_value()[0][1], metrics.get_name_value()[1][1]


def get_model_params(model):
	"""Returns a model's parameters."""
	param = []
	for val in model.collect_params(".*weight").values():
		p = val.data()
		param.append(p.asnumpy())
	return param
	
	