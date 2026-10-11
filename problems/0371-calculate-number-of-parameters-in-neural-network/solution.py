def calculate_parameters(layers: list[dict]) -> int:
	total = 0
	for layer in layers:
		lyType = layer['type']
		if(lyType == 'dense'):
			total += layer['input_size'] * layer['output_size']
			if 'bias' not in layer or layer['bias']:
				total += layer['output_size']
		if(lyType == 'conv2d'):
			inOutChannels = layer['in_channels'] * layer['out_channels']
			total += inOutChannels * (layer['kernel_size']**2) 
			if 'bias' not in layer or layer['bias']:
				total += layer['out_channels']
	return total


	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	# Your code here
	pass