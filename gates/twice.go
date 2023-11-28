package gates

import (
	"fmt"
	"math/rand"
)

func Twice() {
	data := [][]float64{
		{0, 0},
		{1, 2},
		{2, 4},
		{3, 6},
		{4, 8},
	}
	w := rand.Float64()

	lr := 1e-1

	for i := 0; i < 10; i++ {
		dw := cost_twice(data, w)
		w -= lr * dw
		fmt.Printf("cost %f | weight %f\n", cost_twice(data, w), w)
	}

	fmt.Println("-----------FINAL RESULTS------------")

	for _, d := range data {
		fmt.Printf("input %f | expected %f | predicted %f \n", d[0], d[1], d[0]*w)
	}
}

func cost_twice(data [][]float64, w float64) float64 {
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
