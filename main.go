package main

import (
	"gobraincells/matrix"
	"gobraincells/neuralnetwork"
)

func main() {
	nn := neuralnetwork.New(2, 2, []int{3, 5, 4})

	// fmt.Println("Weights:")
	// for _, w := range nn.Weights {
	// 	w.Print()
	// }

	// fmt.Println("Biases:")
	// for _, b := range nn.Biases {
	// 	b.Print()
	// }

	a0 := matrix.New(4, 2, false)
	a0.Set(0, 0, 0)
	a0.Set(0, 1, 0)
	a0.Set(1, 0, 0)
	a0.Set(1, 1, 1)
	a0.Set(2, 0, 1)
	a0.Set(2, 1, 0)
	a0.Set(3, 0, 1)
	a0.Set(3, 1, 1)

	expected := matrix.New(4, 2, false)
	expected.Set(0, 0, 0)
	expected.Set(0, 0, 0)
	expected.Set(1, 0, 0)
	expected.Set(1, 1, 1)
	expected.Set(2, 0, 1)
	expected.Set(2, 1, 0)
	expected.Set(3, 0, 0)
	expected.Set(3, 1, 0)

	nn.Train(a0, expected, 20000, 1e-3, 1e-1)
}
