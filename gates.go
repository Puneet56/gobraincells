package main

import (
	"fmt"
	"math"
	"math/rand"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

var or_data = [][]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 1},
}

var and_data = [][]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 1},
}

var nand_data = [][]float64{
	{0, 0, 1},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

var xor_data = [][]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

func gate() {
	data := or_data

	w1 := rand.Float64()
	w2 := rand.Float64()
	b := rand.Float64()

	eps := 1e-3
	lr := 1e-1

	for i := 0; i < 100000; i++ {
		c := cost(data, w1, w2, b)

		dw1 := (cost(data, w1+eps, w2, b) - c) / eps
		dw2 := (cost(data, w1, w2+eps, b) - c) / eps
		db := (cost(data, w1, w2, b+eps) - c) / eps

		w1 -= lr * dw1
		w2 -= lr * dw2
		b -= lr * db
		fmt.Printf("w1 %f | w2 %f | bias %f | cost %f \n", w1, w2, b, cost(data, w1, w2, b))
	}

	fmt.Println("----------FINAL RESULTS-----------")

	for _, d := range data {
		fmt.Printf("  %f  |  %f  |  %f\n", d[0], d[1], sigmoid(d[0]*w1+d[1]*w2+b))
	}

}

func cost(data [][]float64, w1 float64, w2 float64, b float64) float64 {
	result := 0.0
	for _, d := range data {
		x1 := d[0]
		x2 := d[1]
		y := sigmoid(x1*w1 + x2*w2 + b)

		dw := y - d[2]
		result += dw * dw
	}

	result = result / float64(len(data))

	return result
}
