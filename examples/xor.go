package examples

import (
	"fmt"
	"gobraincells/matrix"
	"gobraincells/neuralnetwork"
)

// 2 inputs, 1 output, one hidden layer with 2 neurons.
func XorGate() {
	nn := neuralnetwork.New(2, 1, []int{2})

	fmt.Println("Weights: ")
	for _, w := range nn.Weights {
		w.Print()
	}

	fmt.Println("Biases: ")
	for _, b := range nn.Biases {
		b.Print()
	}

	inputs := matrix.New(4, 2, false)
	inputs.Set(0, 0, 0)
	inputs.Set(0, 1, 0)
	inputs.Set(1, 0, 0)
	inputs.Set(1, 1, 1)
	inputs.Set(2, 0, 1)
	inputs.Set(2, 1, 0)
	inputs.Set(3, 0, 1)
	inputs.Set(3, 1, 1)

	expected := matrix.New(4, 1, false)
	expected.Set(0, 0, 0)
	expected.Set(1, 0, 1)
	expected.Set(2, 0, 1)
	expected.Set(3, 0, 0)

	nn.Train(inputs, expected, 20000, 1e-3, 1e-1)
}
