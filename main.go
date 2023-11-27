package main

import (
	"gobraincells/matrix"
	"gobraincells/neuralnetwork"
)

func main() {
	nn := neuralnetwork.New(2, 1, []int{3, 5, 2})

	a0 := matrix.New(4, 2, false)
	a0.Set(0, 0, 0)
	a0.Set(0, 0, 0)
	a0.Set(1, 0, 0)
	a0.Set(1, 1, 1)
	a0.Set(2, 0, 1)
	a0.Set(2, 1, 0)
	a0.Set(3, 0, 1)
	a0.Set(3, 1, 1)

	expected := matrix.New(4, 1, false)
	expected.Set(0, 0, 0)
	expected.Set(1, 0, 1)
	expected.Set(2, 0, 1)
	expected.Set(3, 0, 0)

	c := nn.Forward(a0, expected)

	c.Print()
}
