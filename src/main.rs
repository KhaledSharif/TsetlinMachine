extern crate rand;
use rand::thread_rng;

mod tsetlin_machine;
use tsetlin_machine::tsetlin_machine;

extern crate csv;
use csv::Reader;

struct LabelWithData
{
    label : Vec<bool>,
    data  : Vec<bool>,
}

fn main()
{
    let file_path = "/home/khaled/mnist/train.csv";
    let rdr_with_error = Reader::from_path(file_path);
    assert!(!rdr_with_error.is_err());

    let mut rdr = rdr_with_error.ok().unwrap();

    let mut training_data : Vec<LabelWithData> = Vec::new();
    for result in rdr.records()
    {
        let record = result.ok().unwrap();
        let vector : Vec<u16> = record
            .iter()
            .map(|x| x.parse::<u16>().unwrap())
            .collect();

        {
            let label = one_hot_encoder(vector[0]);
            let data = vector[1..].to_vec().iter().map(|&x| x > 128).collect();

            {
                let lwv = LabelWithData
                {
                    label : label,
                    data  : data,
                };
                training_data.push(lwv);
            }
        }
    }
    let (inputs, outputs) = converter(training_data);

    println!("Inputs  length: {}, Inputs[0]  length: {}", inputs.len(),  inputs[0].len());
    println!("Outputs length: {}, Outputs[0] length: {}", outputs.len(), outputs[0].len());

    let mut tm = tsetlin_machine();
    tm.create(inputs[0].len(), outputs[0].len(), 1000);

    let mut rng = thread_rng();
    let mut average_error : f32 = 1.0;

    for e in 0..
    {
        let input_vector           = &inputs[e % inputs.len()];
        let expected_output_vector = &outputs[e % outputs.len()];
        {
            let output_vector = tm.activate(input_vector.to_vec());
            let correct       = check_two_vectors(expected_output_vector, output_vector);
            average_error     = 0.99 * average_error + 0.01 * (if !correct {1.0} else {0.0});
            if e % 10 == 0
            {
                println!("{} | {}%", e, (1.0 - average_error) * 100.0);
            }
        }
        tm.learn(expected_output_vector, 4.0, 4.0, &mut rng);
    }
}

fn check_two_vectors(y_true : &Vec<bool>, y_pred : &Vec<bool>) -> bool
{
    assert_eq!(y_true.len(), y_pred.len());
    assert!(y_true.len() > 0);

    for i in 0..y_true.len()
    {
        if (y_true[i] && !y_pred[i]) || (!y_true[i] && y_pred[i])
        {
            return false;
        }
    }
    return true;
}

fn one_hot_encoder(n: u16) -> Vec<bool>
{
    let mut m : Vec<bool> = (0..).take(10).map(|_x| false).collect();
    m[n as usize] = true;
    return m;
}

fn converter(list: Vec<LabelWithData>) -> (Vec<Vec<bool>>, Vec<Vec<bool>>)
{
    let mut inputs : Vec<Vec<bool>> = Vec::new();
    let mut outputs : Vec<Vec<bool>> = Vec::new();

    for l in list
    {
        inputs.push(l.data);
        outputs.push(l.label);
    }

    (inputs, outputs)
}
