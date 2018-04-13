extern crate rand;
use rand::thread_rng;

extern crate tsetlin_machine;
use tsetlin_machine::tsetlin_machine;

extern crate csv;
use csv::Reader;

fn main()
{
    let training_file_path = "mnist/train.csv";
    let testing_file_path  = "mnist/test.csv";

    let (training_inputs, training_outputs) = read(training_file_path);

    println!("Training dataset");
    println!("Inputs  length: {}, Inputs[0]  length: {}", training_inputs.len(),  training_inputs[0].len());
    println!("Outputs length: {}, Outputs[0] length: {}", training_outputs.len(), training_outputs[0].len());

    let (testing_inputs, testing_outputs)   = read(testing_file_path);

    println!("Testing dataset");
    println!("Inputs  length: {}, Inputs[0]  length: {}", testing_inputs.len(),  testing_inputs[0].len());
    println!("Outputs length: {}, Outputs[0] length: {}", testing_outputs.len(), testing_outputs[0].len());

    let mut tm = tsetlin_machine();
    tm.create(training_inputs[0].len(), training_outputs[0].len(), 10);

    let mut rng = thread_rng();
    let mut average_training_error : f32 = 1.0;
    let mut average_testing_error  : f32 = 1.0;

    loop
    {
        for e in 0 .. (training_inputs.len() - 1)
        {
            let expected_output_vector = &training_outputs[e];
            {
                {
                    let input_vector       = &training_inputs[e];
                    let correct            = check_two_vectors(expected_output_vector, tm.activate(input_vector.to_vec()));
                    average_training_error = 0.99 * average_training_error + 0.01 * (if !correct {1.0} else {0.0});
                }

                if e % 10 == 0
                {
                    println!(
                        "{}% of training dataset | {}% training accuracy",
                        (e as f32 / training_inputs.len() as f32) * 100.0,
                        (1.0 - average_training_error) * 100.0,
                    );
                }
            }
            tm.learn(expected_output_vector, 4.0, 4.0, &mut rng);
        }

        for f in 0 .. (testing_inputs.len() - 1)
        {
            let expected_output_vector = &testing_outputs[f];
            {
                {
                    let input_vector      = &testing_inputs[f];
                    let correct           = check_two_vectors(expected_output_vector, tm.activate(input_vector.to_vec()));
                    average_testing_error = 0.99 * average_testing_error + 0.01 * (if !correct {1.0} else {0.0});
                }

                if f % 10 == 0
                {
                    println!(
                        "{}% of testing dataset | {}% testing accuracy",
                        (f as f32 / testing_inputs.len() as f32) * 100.0,
                        (1.0 - average_testing_error) * 100.0,
                    );
                }
            }
        }
    }
}

struct LabelWithData
{
    label : Vec<bool>,
    data  : Vec<bool>,
}

type BooleanMatrix = Vec<Vec<bool>>;
type BooleanMatrixTuple = (BooleanMatrix, BooleanMatrix);

fn read(file_path : &str) -> BooleanMatrixTuple
{
    fn one_hot_encoder(n: u16) -> Vec<bool>
    {
        let mut m : Vec<bool> = (0..).take(10).map(|_x| false).collect();
        m[n as usize] = true;
        return m;
    }

    fn converter(list: Vec<LabelWithData>) -> BooleanMatrixTuple
    {
        let mut inputs  : BooleanMatrix = Vec::new();
        let mut outputs : BooleanMatrix = Vec::new();

        for l in list
        {
            inputs.push(l.data);
            outputs.push(l.label);
        }

        (inputs, outputs)
    }

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

    converter(training_data)
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
