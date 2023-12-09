package gates

import (
	"fmt"
	"gobraincells/activation"
	"gobraincells/matrix"
	"math/rand"
)

var or_data = [][]float64{{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}}

var and_data = [][]float64{{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}}

var nand_data = [][]float64{{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}

func OrGate(derivative bool) {
	gate(or_data, derivative)
}

func AndGate(derivative bool) {
	gate(and_data, derivative)
}

func NandGate(derivative bool) {
	gate(nand_data, derivative)
}

func gate(data [][]float64, derivative bool) {
	w1 := rand.Float64()
	w2 := rand.Float64()
	b := rand.Float64()

	if derivative {
		for i := 0; i < 1000; i++ {
			dc := gcost_gate(data, w1, w2, b)

			w1 -= dc[0]
			w2 -= dc[1]
			b -= dc[2]

			fmt.Printf("w1 %f | w2 %f | bias %f | cost %f \n", w1, w2, b, cost_gate(data, w1, w2, b))
		}
	} else {
		eps := 1e-3
		lr := 1e-1

		for i := 0; i < 1000; i++ {
			c := cost_gate(data, w1, w2, b)

			dw1 := (cost_gate(data, w1+eps, w2, b) - c) / eps
			dw2 := (cost_gate(data, w1, w2+eps, b) - c) / eps
			db := (cost_gate(data, w1, w2, b+eps) - c) / eps

			w1 -= lr * dw1
			w2 -= lr * dw2
			b -= lr * db
			fmt.Printf("w1 %f | w2 %f | bias %f | cost %f \n", w1, w2, b, cost_gate(data, w1, w2, b))
		}
	}
	fmt.Println("----------FINAL RESULTS-----------")

	for _, d := range data {
		fmt.Printf("  %f  |  %f  |  %f\n", d[0], d[1], activation.Sigmoid(d[0]*w1+d[1]*w2+b))
	}
}

func gcost_gate(data [][]float64, w1 float64, w2 float64, b float64) []float64 {
	cost := make([]float64, 3)

	for _, d := range data {
		x1 := d[0]
		x2 := d[1]
		z := d[2]
		a := activation.Sigmoid(x1*w1 + x2*w2 + b)

		cost[0] += 2 * (a - z) * a * (1 - a) * x1
		cost[1] += 2 * (a - z) * a * (1 - a) * x2
		cost[2] += 2 * (a - z) * a * (1 - a)
	}

	for i := range cost {
		cost[i] /= float64(len(data))
	}

	return cost
}

func cost_gate(data [][]float64, w1 float64, w2 float64, b float64) float64 {
	result := 0.0
	for _, d := range data {
		x1 := d[0]
		x2 := d[1]
		y := activation.Sigmoid(x1*w1 + x2*w2 + b)

		dw := y - d[2]
		result += dw * dw
	}

	result = result / float64(len(data))

	return result
}

func XorGate() {
	// xor
	// [
	// 	[0, 0],
	// 	[0, 1],
	// 	[1, 0],
	// 	[1, 1]
	// ]
	a0 := matrix.New(4, 2, false)
	a0.Set(0, 0, 0)
	a0.Set(0, 0, 0)
	a0.Set(1, 0, 0)
	a0.Set(1, 1, 1)
	a0.Set(2, 0, 1)
	a0.Set(2, 1, 0)
	a0.Set(3, 0, 1)
	a0.Set(3, 1, 1)

	// [
	// 	[0],
	// 	[1],
	// 	[1],
	// 	[0]
	// ]
	expected := matrix.New(4, 1, false)
	expected.Set(0, 0, 0)
	expected.Set(1, 0, 1)
	expected.Set(2, 0, 1)
	expected.Set(3, 0, 0)

	w1 := matrix.New(2, 2, true)
	b1 := matrix.New(1, 2, false)

	w2 := matrix.New(2, 1, true)
	b2 := matrix.New(1, 1, false)

	eps := 1e-1
	lr := 1e-1

	for i := 0; i < 100000; i++ {
		w1g := matrix.New(w1.Rows, w1.Cols, false)
		for j := 0; j < w1.Rows; j++ {
			for k := 0; k < w1.Cols; k++ {
				s := w1.Get(j, k)
				w1.Set(j, k, s+eps)
				c1 := forward_xor(a0, expected, w1, b1, w2, b2)
				w1.Set(j, k, s-eps)
				c2 := forward_xor(a0, expected, w1, b1, w2, b2)
				w1.Set(j, k, s)
				w1g.Set(j, k, (c1-c2)/(2*eps))
			}
		}

		b1g := matrix.New(b1.Rows, b1.Cols, false)
		for j := 0; j < b1.Rows; j++ {
			for k := 0; k < b1.Cols; k++ {
				s := b1.Get(j, k)
				b1.Set(j, k, s+eps)
				c1 := forward_xor(a0, expected, w1, b1, w2, b2)
				b1.Set(j, k, s-eps)
				c2 := forward_xor(a0, expected, w1, b1, w2, b2)
				b1.Set(j, k, s)
				b1g.Set(j, k, (c1-c2)/(2*eps))
			}
		}

		w2g := matrix.New(w2.Rows, w2.Cols, false)
		for j := 0; j < w2.Rows; j++ {
			for k := 0; k < w2.Cols; k++ {
				s := w2.Get(j, k)
				w2.Set(j, k, s+eps)
				c1 := forward_xor(a0, expected, w1, b1, w2, b2)
				w2.Set(j, k, s-eps)
				c2 := forward_xor(a0, expected, w1, b1, w2, b2)
				w2.Set(j, k, s)
				w2g.Set(j, k, (c1-c2)/(2*eps))
			}
		}

		b2g := matrix.New(b2.Rows, b2.Cols, false)
		for j := 0; j < b2.Rows; j++ {
			for k := 0; k < b2.Cols; k++ {
				s := b2.Get(j, k)
				b2.Set(j, k, s+eps)
				c1 := forward_xor(a0, expected, w1, b1, w2, b2)
				b2.Set(j, k, s-eps)
				c2 := forward_xor(a0, expected, w1, b1, w2, b2)
				b2.Set(j, k, s)
				b2g.Set(j, k, (c1-c2)/(2*eps))
			}
		}

		w1g.Apply(func(x float64) float64 { return x * lr })
		b1g.Apply(func(x float64) float64 { return x * lr })
		w2g.Apply(func(x float64) float64 { return x * lr })
		b2g.Apply(func(x float64) float64 { return x * lr })

		w1.Subtract(w1g)
		b1.Subtract(b1g)
		w2.Subtract(w2g)
		b2.Subtract(b2g)

		fmt.Println("cost:", forward_xor(a0, expected, w1, b1, w2, b2))
	}

	fmt.Println("--------Final--------")

	for i := 0; i < a0.Rows; i++ {
		a1 := matrix.Multiply(a0.GetRow(i), w1)
		a1.Add(b1)
		a1.Apply(activation.Sigmoid)

		a2 := matrix.Multiply(a1, w2)
		a2.Add(b2)
		a2.Apply(activation.Sigmoid)

		fmt.Printf("%f  |  %f  |  %f\n", a0.Get(i, 0), a0.Get(i, 1), a2.Get(0, 0))
	}
}

func forward_xor(a0, expected, w1, b1, w2, b2 matrix.Matrix) float64 {
	cost := 0.0
	for i := 0; i < a0.Rows; i++ {
		a1 := matrix.Multiply(a0.GetRow(i), w1)
		a1.Add(b1)
		a1.Apply(activation.Sigmoid)

		a2 := matrix.Multiply(a1, w2)
		a2.Add(b2)
		a2.Apply(activation.Sigmoid)

		d := a2.Get(0, 0) - expected.Get(i, 0)

		cost += d * d
	}
	return cost / float64(a0.Rows)
}
