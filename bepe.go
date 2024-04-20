package main

import (
	"strconv"
)

type Value struct {
	value    float64
	op       string
	grad     float64
	children []*Value
}

func (v *Value) String() string {
	return strconv.FormatFloat(v.value, 'g', -1, 64)
}

func (v *Value) Value() interface{} {
	return v.value
}

func (v *Value) Add(v2 *Value) *Value {
	return &Value{
		value:    v.value + v2.value,
		op:       "+",
		children: []*Value{v, v2},
	}
}

func (v *Value) Mul(v2 *Value) *Value {
	return &Value{
		value:    v.value * v2.value,
		op:       "*",
		children: []*Value{v, v2},
	}
}

func (v *Value) Sub(v2 *Value) *Value {
	return v.Add(v2.Mul(&Value{value: -1}))
}
