package gates

import (
	"fmt"
	"math/rand"
)

func Twice(derivative bool) {
	data := [][]float64{
		{0, 0},
		{1, 2},
		{2, 4},
		{3, 6},
		{4, 8},
	}
	w := rand.Float64()

	lr := 1e-1

	if derivative {
		for i := 0; i < 100; i++ {
			dw := cost_twice_derivative(data, w)
			w -= lr * dw
			fmt.Printf("cost %f | weight %f\n", cost_twice_derivative(data, w), w)
		}

		fmt.Println("-----------FINAL RESULTS------------")
		for _, d := range data {
			fmt.Printf("input %f | expected %f | predicted %f \n", d[0], d[1], d[0]*w)
		}
	} else {
		eps := 1e-3
		for i := 0; i < 100; i++ {
			c := cost_twice_eps(data, w)
			dw := (cost_twice_eps(data, w+eps) - c) / eps

			w -= lr * dw
			fmt.Printf("cost %f | weight %f\n", cost_twice_derivative(data, w), w)
		}

		fmt.Println("-----------FINAL RESULTS------------")
		for _, d := range data {
			fmt.Printf("input %f | expected %f | predicted %f \n", d[0], d[1], d[0]*w)
		}
	}
}

func cost_twice_derivative(data [][]float64, w float64) float64 {
	result := 0.0
	for _, d := range data {
		x := d[0]
		y := d[1]

		dw := 2 * (x*w - y) * x
		result += dw
	}

	result = result / float64(len(data))

	return result
}

func cost_twice_eps(data [][]float64, w float64) float64 {
	result := 0.0
	for _, d := range data {
		x := d[0]
		y := x * w

		dw := y * y
		result += dw
	}

	result = result / float64(len(data))

	return result
}
