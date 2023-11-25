package main

import (
	"fmt"
	"math/rand"
)

func main() {
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
	lr := 1e-3

	fmt.Printf("%f \n", cost(data, w, b))

	for i := 0; i < 1000; i++ {
		dw := (cost(data, w+eps, b) - cost(data, w, b)) / eps
		db := (cost(data, w, b+eps) - cost(data, w, b)) / eps

		w -= lr * dw
		b -= lr * db
		fmt.Printf("cost %f | weight %f | bias %f \n", w, b, cost(data, w, b))
	}

	fmt.Println("-------------------------------")
	fmt.Printf("final weight %f and bias %f \n", w, b)
}

func cost(data [][]float64, w float64, b float64) float64 {
	result := 0.0
	for _, d := range data {
		x := d[0]
		y := x*w + b

		dw := y - d[1]
		result += dw * dw

		// fmt.Printf("expected %f | got %f \n", d[1], y)
	}

	result = result / float64(len(data))

	return result
}
