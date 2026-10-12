import numpy as np
import math
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	weights = initial_weights
	bias = initial_bias
	mse_values = []
	
	N = len(features)

	for _ in range(epochs):
		weight_g = [0.0] * len(initial_weights)
		bias_g = 0.0
		predictions = []
		for feature in features:
			z = sum((feat * weight for feat, weight in zip(feature, weights))) + bias
			pred = 1/(1 + math.exp(-z))
			predictions.append(pred)
		mse = sum((predictions - labels) ** 2/N)
		mse_values.append(mse)
		
		for feat, label, pred in zip(features, labels, predictions):
			delta = 2 * (pred - label) * pred * (1 - pred)
			for j in range(len(feat)):
				weight_g[j] += delta * feat[j]
			bias_g += delta * 1
		bias_g = bias_g / N
		weight_g = [g / N for g in weight_g]

		weights = [weight - learning_rate * grad for weight, grad in zip(weights, weight_g)]
		bias = bias - learning_rate * bias_g
		
	return weights, bias, mse_values