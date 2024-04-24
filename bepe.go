package main

import (
	"fmt"
	"math/rand"
	"strconv"
)

type Module interface {
	Zero()
	Parameters() []*Value
	String() string
}

type Value struct {
	value    float64
	op       string
	grad     float64
	children [2]*Value
	back     func()
}

func Val(value float64) *Value {
	return &Value{value: value, back: func() {
	}, grad: 0}
}

func MakeVal(value float64, children [2]*Value, op string) *Value {
	return &Value{value: value, children: children, op: op, back: func() {
	}, grad: 0}
}

func (v *Value) String() string {
	return strconv.FormatFloat(v.value, 'g', -1, 64)
}

func (v *Value) Value() interface{} {
	return v.value
}

func (v *Value) Add(v2 *Value) *Value {
	out := MakeVal(v.value+v2.value, [2]*Value{v, v2}, "+")

	out.back = func() {
		v.grad += out.grad
		v2.grad += out.grad
	}
	return out
}

func (v *Value) Mul(v2 *Value) *Value {
	out := MakeVal(v.value*v2.value, [2]*Value{v, v2}, "*")
	out.back = func() {
		v.grad += v2.value * out.grad
		v2.grad += v.value * out.grad
	}

	return out
}

func (v *Value) Sub(v2 *Value) *Value {
	return v.Add(v2.Mul(&Value{value: -1}))
}

func (v *Value) Pow(p float64) *Value {
	out := MakeVal(v.value, [2]*Value{v, nil}, "pow")
	out.back = func() {
		v.grad += p * out.grad * v.value
	}

	return out
}

func (v *Value) Relu() *Value {
	val := 0.0
	if v.value > 0 {
		val = v.value
	}

	out := MakeVal(val, [2]*Value{v, nil}, "relu")
	out.back = func() {
		if v.value > 0 {
			v.grad += out.grad
		}
	}

	return out
}

func (v *Value) Backward() {
	topo := []*Value{}
	visited := map[*Value]bool{}
	var build_topo func(*Value)

	build_topo = func(v *Value) {
		if visited[v] {
			return
		}

		visited[v] = true
		for _, c := range v.children {
			if c != nil {
				build_topo(c)
			}
		}

		topo = append(topo, v)
	}

	build_topo(v)

	v.grad = 1

	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].back()
	}
}

func (v *Value) Zero() {
	v.grad = 0
}

type Neuron struct {
	weights []*Value
	bias    *Value
	nonlin  bool
}

func MakeNeuron(n int, nonlin bool) *Neuron {
	weights := make([]*Value, n)
	for i := 0; i < n; i++ {
		weights[i] = &Value{value: rand.Float64()}
	}

	return &Neuron{weights: weights, bias: &Value{value: 0}, nonlin: nonlin}
}

func (n *Neuron) Parameters() []*Value {
	return append([]*Value{n.bias}, n.weights...)
}

func (n *Neuron) Call(x []*Value) *Value {
	out := n.bias
	for i := range min(len(n.weights), len(x)) {
		out = out.Add(n.weights[i].Mul(x[i]))
	}

	if n.nonlin {
		out = out.Relu()
	}

	return out
}

func (n *Neuron) String() string {
	t := "Linear"
	if n.nonlin {
		t = "Relu"
	}

	return fmt.Sprintf("%vNeuron(%v)", t, len(n.weights))
}

func (n *Neuron) Zero() {
	n.bias.Zero()
	for _, w := range n.weights {
		w.Zero()
	}
}

type Layer struct {
	neurons []*Neuron
}

func MakeLayer(nin int, nout int) *Layer {
	neurons := make([]*Neuron, nout)
	for i := 0; i < nout; i++ {
		neurons[i] = MakeNeuron(nin, true)
	}

	return &Layer{neurons: neurons}
}

func (l *Layer) Parameters() []*Value {
	out := []*Value{}
	for _, n := range l.neurons {
		out = append(out, n.Parameters()...)
	}

	return out
}

func (l *Layer) Call(x []*Value) []*Value {
	out := make([]*Value, len(l.neurons))
	for i, n := range l.neurons {
		out[i] = n.Call(x)
	}

	return out
}

func (l *Layer) String() string {
	return fmt.Sprintf("Layer(%v, %v)", len(l.neurons[0].weights), len(l.neurons))
}

func (l *Layer) Zero() {
	for _, n := range l.neurons {
		n.Zero()
	}
}

type Model struct {
	layers []*Layer
}

func MakeModel(nin int, nouts []int) *Model {
	layers := make([]*Layer, len(nouts))
	for i, nout := range nouts {
		if i == 0 {
			layers[i] = MakeLayer(nin, nout)
		} else {
			layers[i] = MakeLayer(nouts[i-1], nout)
		}
	}

	return &Model{layers: layers}
}

func (m *Model) Parameters() []*Value {
	out := []*Value{}
	for _, l := range m.layers {
		out = append(out, l.Parameters()...)
	}

	return out
}

func (m *Model) Call(x []*Value) []*Value {
	for _, l := range m.layers {
		x = l.Call(x)
	}

	return x
}

func (m *Model) Zero() {
	for _, l := range m.layers {
		l.Zero()
	}
}

func (m *Model) String() string {
	return fmt.Sprintf("Model(%v)", len(m.layers))
}

func main() {
	x := Val(-4.0)
	z := x.Mul(Val(2)).Add(Val(2)).Add(x)
	q := z.Relu().Add(z.Mul(x))
	h := z.Mul(z).Relu()
	y := h.Add(q).Add(q.Mul(x))
	fmt.Println(x.grad)
	y.Backward()

	fmt.Println(x, y)
	fmt.Println(x.grad, y.grad)

}
