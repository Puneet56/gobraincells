package main

import (
	"gobraincells/neuralnetwork"
)

func main() {
	nn := neuralnetwork.New(8, 5, []int{10, 10, 10})

	nn.Visualize()
}
