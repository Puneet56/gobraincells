package neuralnetwork

import (
	"fmt"
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

		nn.Biases[len(hiddenLayers)] = matrix.New(1, outputs, true)
		nn.Weights[len(hiddenLayers)] = matrix.New(hiddenLayers[len(hiddenLayers)-1], outputs, true)
	}

	return nn
}

func (nn *NeuralNetwork) Forward(inputs, expected matrix.Matrix) float64 {
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

	c := 0.0

	for i := 0; i < cost.Rows; i++ {
		for j := 0; j < cost.Cols; j++ {
			c += cost.Get(i, j)
		}
	}

	return c
}

func (nn *NeuralNetwork) Train(inputs, expected matrix.Matrix, epochs int, eps, learningRate float64) {
	for i := 0; i < epochs; i++ {
		for w := 0; w < len(nn.Weights); w++ {
			wg := matrix.New(nn.Weights[w].Rows, nn.Weights[w].Cols, false)
			for j := 0; j < nn.Weights[w].Rows; j++ {
				for k := 0; k < nn.Weights[w].Cols; k++ {
					s := nn.Weights[w].Get(j, k)
					nn.Weights[w].Set(j, k, s+eps)
					c1 := nn.Forward(inputs, expected)
					nn.Weights[w].Set(j, k, s-eps)
					c2 := nn.Forward(inputs, expected)
					nn.Weights[w].Set(j, k, s)
					wg.Set(j, k, (c1-c2)/(2*eps))
				}
			}
			wg.Apply(func(x float64) float64 { return x * learningRate })
			nn.Weights[w].Subtract(wg)
		}

		for b := 0; b < len(nn.Biases); b++ {
			bg := matrix.New(nn.Biases[b].Rows, nn.Biases[b].Cols, false)
			for j := 0; j < nn.Biases[b].Rows; j++ {
				for k := 0; k < nn.Biases[b].Cols; k++ {
					s := nn.Biases[b].Get(j, k)
					nn.Biases[b].Set(j, k, s+eps)
					c1 := nn.Forward(inputs, expected)
					nn.Biases[b].Set(j, k, s-eps)
					c2 := nn.Forward(inputs, expected)
					nn.Biases[b].Set(j, k, s)
					bg.Set(j, k, (c1-c2)/(2*eps))
				}
			}
			bg.Apply(func(x float64) float64 { return x * learningRate })
			nn.Biases[b].Subtract(bg)
		}
		fmt.Println("i: ", i, "cost:", nn.Forward(inputs, expected))
	}

	fmt.Println("--------Final--------")

	for i := 0; i < inputs.Rows; i++ {
		output := inputs.GetRow(i)

		for j := 0; j < len(nn.Weights); j++ {
			output = matrix.Multiply(output, nn.Weights[j])

			b := nn.Biases[j]

			output.Add(b)
			output.Apply(activation.Sigmoid)
		}

		fmt.Printf("Expected: %f  |  %f  =  %f  |  %f \n", inputs.Get(i, 0), inputs.Get(i, 1), expected.Get(i, 0), expected.Get(i, 1))
		fmt.Printf("Output:   %f  |  %f  =  %f  |  %f \n", inputs.Get(i, 0), inputs.Get(i, 1), output.Get(0, 0), output.Get(0, 1))

		fmt.Println()
	}
}
