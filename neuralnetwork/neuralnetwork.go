package neuralnetwork

import (
	"gobraincells/activation"
	"gobraincells/matrix"
)

type NeuralNetwork struct {
	Weights         []matrix.Matrix
	Biases          []matrix.Matrix
	hiddenLayers    []int
	inputs, outputs int
}

func New(inputs, outputs int, hiddenLayers []int) NeuralNetwork {
	nn := NeuralNetwork{
		inputs:       inputs,
		outputs:      outputs,
		hiddenLayers: hiddenLayers,
	}

	nn.Biases = make([]matrix.Matrix, len(hiddenLayers)+1)
	nn.Weights = make([]matrix.Matrix, len(hiddenLayers)+1)

	if len(hiddenLayers) == 0 {
		nn.Biases[0] = matrix.New(1, outputs, true)
		nn.Weights[0] = matrix.New(inputs, hiddenLayers[0], true)
	} else {
		nn.Biases[0] = matrix.New(1, hiddenLayers[0], true)
		nn.Weights[0] = matrix.New(inputs, hiddenLayers[0], true)

		for i := 1; i < len(hiddenLayers); i++ {
			nn.Biases[i] = matrix.New(1, hiddenLayers[i], true)
			nn.Weights[i] = matrix.New(hiddenLayers[i-1], hiddenLayers[i], true)
		}

		nn.Biases[len(hiddenLayers)] = matrix.New(outputs, 1, true)
		nn.Weights[len(hiddenLayers)] = matrix.New(hiddenLayers[len(hiddenLayers)-1], outputs, true)
	}

	return nn
}

func (nn *NeuralNetwork) Forward(inputs, expected matrix.Matrix) matrix.Matrix {
	if inputs.Cols != nn.inputs {
		panic("input matrix cols should be same as the no. of inputs of neural network")
	}

	if expected.Cols != nn.outputs {
		panic("output matrix cols should be same as the no. of outputs of neural network")
	}

	if inputs.Rows != expected.Rows {
		panic("input and output matrices should have same no. of rows")
	}

	cost := matrix.New(expected.Rows, expected.Cols, false)

	for i := 0; i < inputs.Rows; i++ {
		output := inputs.GetRow(i)

		for j := 0; j < len(nn.Weights); j++ {
			output = matrix.Multiply(output, nn.Weights[j])

			b := nn.Biases[j]

			output.Add(b)
			output.Apply(activation.Sigmoid)
		}

		d := expected.GetRow(i)
		d.Subtract(output)
		d.Apply(func(f float64) float64 { return f * f })

		cost.SetRow(i, d)
	}

	return cost
}

func (nn *NeuralNetwork) Train(inputs, expected matrix.Matrix, eps, learningRate float64) {
	// TODO: Implement training
}
