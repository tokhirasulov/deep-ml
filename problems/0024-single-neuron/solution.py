import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	probabilities = []
	N = len(features)
	for feature in features:
		total_sum = 0.0
		for feat, weight in zip(feature, weights):
			total_sum += feat * weight
		total_sum += bias
		probab = 1/(1 + math.exp(-total_sum))
		probabilities.append(probab)
		total = 0.0
	for prob, label in zip(probabilities, labels):
		total += (prob - label) ** 2
	mse = total / N
	return probabilities, mse