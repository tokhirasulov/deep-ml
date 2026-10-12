import numpy as np
import math
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	mse_values = []
	N = len(features)
	bias = initial_bias
	weights = list(initial_weights)

	for _ in range(epochs):
		grad_w = [0.0] * len(initial_weights)
		grad_b = 0.0
		predictions = []
		for feature in features:
			z = sum(feat * weight for feat, weight in zip(feature,weights)) + bias
			pred = 1/(1 + math.exp(-z))
			predictions.append(pred)
		
		mse = sum((predictions - labels)**2/N)
		mse_values.append(mse)

		for feature, pred, label in zip(features, predictions, labels):
			delta = 2 * (pred - label) * pred * (1 - pred)
			for j in range(len(feature)):
				grad_w[j] += delta * feature[j]
			grad_b += delta * 1
		
		grad_b = grad_b / N
		grad_w = [g / N for g in grad_w]
		weights = [weight - learning_rate * grad for weight, grad in zip(weights, grad_w)]
		bias = bias - learning_rate * grad_b
 			

	return weights, bias, mse_values