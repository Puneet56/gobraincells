package matrix

import (
	"fmt"
	"math/rand"
)

type Matrix struct {
	Data       *[]float64
	Rows, Cols int
}

func NewMatrix(rows, cols int, randomize bool) Matrix {
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

func (m *Matrix) SetElement(row, col int, value float64) {
	if row >= 0 && row < m.Rows && col >= 0 && col < m.Cols {
		index := m.Index(row, col)
		(*m.Data)[index] = value
	}
}

func (m *Matrix) GetElement(row, col int) (float64, error) {
	if row < 0 || row >= m.Rows || col < 0 || col >= m.Cols {
		return 0, fmt.Errorf("row or column out of bounds")
	}

	index := m.Index(row, col)
	return (*m.Data)[index], nil
}

func (m *Matrix) GetRow(row int) []float64 {
	result := make([]float64, m.Cols)
	for i := 0; i < m.Cols; i++ {
		result[i], _ = m.GetElement(row, i)
	}
	return result
}

func (m *Matrix) PrintMatrix() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			fmt.Printf("%f ", (*m.Data)[m.Index(i, j)])
		}
		fmt.Println()
	}
}

func (m *Matrix) Add(n Matrix) {
	if m.Rows != n.Rows || m.Cols != n.Cols {
		fmt.Println("Matrices are not the same size")
		return
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			index := m.Index(i, j)
			(*m.Data)[index] += (*n.Data)[index]
		}
	}
}

func MultiplyMatrices(left Matrix, right Matrix) (*Matrix, error) {
	if left.Cols != right.Rows {
		return nil, fmt.Errorf("cols of left matrix must equal rows of right matrix")
	}

	result := NewMatrix(left.Rows, right.Cols, false)

	for i := 0; i < left.Rows; i++ {
		for j := 0; j < right.Cols; j++ {
			sum := 0.0
			for k := 0; k < left.Cols; k++ {
				leftVal, _ := left.GetElement(i, k)
				rightVal, _ := right.GetElement(k, j)
				sum += leftVal * rightVal
			}
			result.SetElement(i, j, sum)
		}
	}
	return &result, nil
}
