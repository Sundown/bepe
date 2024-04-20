package main

import (
	"strconv"
)

type Value struct {
	value    float64
	op       string
	grad     float64
	children [2]*Value
	back     func()
}

func MakeVal(value float64, children [2]*Value, op string) *Value {
	return &Value{value: value, children: children, op: op, back: nil, grad: 0}
}

func (v *Value) String() string {
	return strconv.FormatFloat(v.value, 'g', -1, 64)
}

func (v *Value) Value() interface{} {
	return v.value
}

func (v *Value) Add(v2 *Value) *Value {
	out := &Value{
		value:    v.value + v2.value,
		op:       "+",
		children: [2]*Value{v, v2},
	}

	out.back = func() {
		v.grad += out.grad
		v2.grad += out.grad
	}
	return out
}

func (v *Value) Mul(v2 *Value) *Value {
	out := &Value{
		value:    v.value * v2.value,
		op:       "*",
		children: [2]*Value{v, v2},
	}

	out.back = func() {
		v.grad += v2.value * out.grad
		v2.grad += v.value * out.grad
	}

	return out
}

func (v *Value) Sub(v2 *Value) *Value {
	return v.Add(v2.Mul(&Value{value: -1}))
}
