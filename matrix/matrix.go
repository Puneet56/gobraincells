package matrix

import (
	"fmt"
	"math/rand"
)

type Matrix struct {
	Data       *[]float64
	Rows, Cols int
}

func New(rows, cols int, randomize bool) Matrix {
	data := make([]float64, rows*cols)

	m := Matrix{
		Rows: rows,
		Cols: cols,
		Data: &data,
	}

	if randomize {
		m.Randomize()
	}

	return m
}

func (m *Matrix) Index(row, col int) int {
	return row*m.Cols + col
}

func (m *Matrix) Randomize() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			index := m.Index(i, j)
			(*m.Data)[index] = rand.Float64()
		}
	}
}

func (m *Matrix) Set(row, col int, value float64) {
	if row < 0 || row >= m.Rows || col < 0 || col >= m.Cols {
		panic("index out of bounds")
	}

	index := m.Index(row, col)
	(*m.Data)[index] = value
}

func (m *Matrix) Get(row, col int) float64 {
	if row < 0 || row >= m.Rows || col < 0 || col >= m.Cols {
		panic("index out of bounds")
	}

	index := m.Index(row, col)
	return (*m.Data)[index]
}

func (m *Matrix) GetRow(row int) Matrix {
	data := make([]float64, m.Cols)

	r := Matrix{
		Rows: 1,
		Cols: m.Cols,
		Data: &data,
	}

	for i := 0; i < m.Cols; i++ {
		r.Set(0, i, m.Get(row, i))
	}
	return r
}

func (m *Matrix) Print() {

	fmt.Println("[")

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			fmt.Printf("  %f ", (*m.Data)[m.Index(i, j)])
		}
		fmt.Printf("\n")
	}

	fmt.Printf("]\n")
}

func (m *Matrix) Add(n Matrix) {
	if m.Rows != n.Rows || m.Cols != n.Cols {
		panic("matrices must be the same size")
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			index := m.Index(i, j)
			(*m.Data)[index] += (*n.Data)[index]
		}
	}
}

func (m *Matrix) Subtract(n Matrix) {
	if m.Rows != n.Rows || m.Cols != n.Cols {
		panic("matrices must be the same size")
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			index := m.Index(i, j)
			(*m.Data)[index] -= (*n.Data)[index]
		}
	}
}

func Multiply(left Matrix, right Matrix) Matrix {
	if left.Cols != right.Rows {
		panic("left matrix must have the same number of columns as right matrix has rows")
	}

	m := New(left.Rows, right.Cols, false)

	for i := 0; i < left.Rows; i++ {
		for j := 0; j < right.Cols; j++ {
			sum := 0.0
			for k := 0; k < left.Cols; k++ {
				leftVal := left.Get(i, k)
				rightVal := right.Get(k, j)
				sum += leftVal * rightVal
			}
			m.Set(i, j, sum)
		}
	}
	return m
}

type ApplyFunc func(float64) float64

func (m *Matrix) Apply(f ApplyFunc) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			index := m.Index(i, j)
			(*m.Data)[index] = f((*m.Data)[index])
		}
	}
}
