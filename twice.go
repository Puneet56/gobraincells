package main

import (
	"fmt"
	"math/rand"
)

func twice() {
	data := [][]float64{
		{0, 0},
		{1, 2},
		{2, 4},
		{3, 6},
		{4, 8},
	}
	w := rand.Float64() * 10
	b := rand.Float64() * 5

	eps := 1e-3
	lr := 1e-2

	for i := 0; i < 500; i++ {
		c := cost_twice(data, w, b)

		dw := (cost_twice(data, w+eps, b) - c) / eps
		// db := (cost(data, w, b+eps) - c) / eps

		w -= lr * dw
		// b -= lr * db
		fmt.Printf("cost %f | weight %f\n", c, w)
	}

	fmt.Println("----------FINAL RESULTS-----------")

	for _, d := range data {
		fmt.Printf("input %f | expected %f | predicted %f \n", d[0], d[1], d[0]*w)
	}
}

func cost_twice(data [][]float64, w float64, b float64) float64 {
	result := 0.0
	for _, d := range data {
		x := d[0]
		y := x * w

		dw := y - d[1]
		result += dw * dw
	}

	result = result / float64(len(data))

	return result
}
