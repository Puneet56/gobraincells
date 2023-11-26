package main

import (
	"gobraincells/matrix"
)

func main() {
	input := matrix.NewMatrix(4, 2, true)
	input.SetElement(0, 0, 0)
	input.SetElement(0, 1, 0)
	input.SetElement(1, 0, 1)
	input.SetElement(1, 1, 0)
	input.SetElement(2, 0, 0)
	input.SetElement(2, 1, 1)
	input.SetElement(3, 0, 1)
	input.SetElement(3, 1, 0)

	expectedOutput := matrix.NewMatrix(4, 1, true)

	expectedOutput.SetElement(0, 0, 0)
	expectedOutput.SetElement(1, 0, 1)
	expectedOutput.SetElement(2, 0, 1)
	expectedOutput.SetElement(3, 0, 1)
}
