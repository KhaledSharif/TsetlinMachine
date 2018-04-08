extern crate rand;
use rand::thread_rng;

mod tsetlin_machine;
use tsetlin_machine::tsetlin_machine;

fn main()
{
    let inputs : Vec<Vec<bool>> = 
    [
        [ 0, 0 ],
        [ 0, 1 ],
        [ 1, 0 ],
        [ 1, 1 ]
    ]
    .iter()
    .map(|x| x.iter().map(|&y| y == 1).collect::<Vec<bool>>().to_vec())
    .collect();

    let outputs : Vec<Vec<bool>> = 
    [
        [ 0, 1 ],
        [ 1, 0 ],
        [ 1, 0 ],
        [ 0, 1 ],
    ]
    .iter()
    .map(|x| x.iter().map(|&y| y == 1).collect::<Vec<bool>>().to_vec())
    .collect();

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
}

