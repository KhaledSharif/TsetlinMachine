## Tsetlin Machine implementation in Rust

A "Tsetlin Machine" solves complex pattern recognition problems with easy-to-interpret
propositional formulas, and is composed of a collective of
[Tsetlin Automata](https://en.wikipedia.org/wiki/Learning_automata). The idea of
the machine was proposed in
[a paper by Ole-Christoffer Granmo](https://arxiv.org/abs/1804.01508).

## Example code

```rust
let mut tm = tsetlin_machine();
tm.create(2, 2, 10);

let mut rng = thread_rng();
let mut average_error : f32 = 1.0;

for e in 0..1000
{
    let input_vector = &inputs[e % 4];
    {
        let output_vector = tm.activate(input_vector.to_vec());
        let mut correct = false;
        if (input_vector[0] == input_vector[1]) && (!output_vector[0] && output_vector[1])
        {
            correct = true;
        }
        else if output_vector[0] && !output_vector[1]
        {
            correct = true;
        }
        average_error = 0.99 * average_error + 0.01 * (if !correct {1.0} else {0.0});
        println!("{} {} -> {} {} | {}", input_vector[0], input_vector[1], output_vector[0], output_vector[1], average_error);
    }
    tm.learn(&outputs[e % 4], 4.0, 4.0, &mut rng);
}
```

## Example output

```
true true -> false true   | 0.00007643679
false false -> false true | 0.00007567242
false true -> true false  | 0.0000749157
true false -> true false  | 0.00007416654
true true -> false true   | 0.000073424875
```

## Original implementation

This repository is [a translation of this repository](https://github.com/222464/TsetlinMachine),
which is an implementation of the Tsetlin Machine in C++.

