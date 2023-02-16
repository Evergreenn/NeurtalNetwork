use std::io::stdin;

use rand::*;

mod afunctions;
mod models;
mod neurals;

use crate::neurals::network::NeuralNetwork;

//100k - 0.5

const TRAINING_LOOP: usize = 500_000;
const DATASET_SIZE: usize = 1;
const LEARNING_RATE: f64 = 1.2;

fn generate_dataset(size: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rng = thread_rng();

    (0..size)
        .map(|_| {
            let input: Vec<f64> = (0..10).map(|_| rng.gen_range(-1.0, 1.0)).collect();
            let target: Vec<f64> = (0..1).map(|_| rng.gen_range(-1.0, 1.0)).collect();
            // let input: Vec<f64> = (0..10).map(|_| rng.gen_range(0.0, 1.0)).collect();
            // let target: Vec<f64> = (0..1).map(|_| rng.gen_range(0.0, 1.0)).collect();
            (input, target)
        })
        .collect()
}

fn main() {
    let mut nn = NeuralNetwork::new(6, &[100], 1);
    let mut rng = thread_rng();
    // let dataset_gen = generate_dataset(DATASET_SIZE);
    let mut dataset = vec![
        (vec![1.0, 1.0, 1.0], vec![1.0]),
        (vec![1.0, 1.0, -1.0], vec![1.0]),
        (vec![1.0, -1.0, 1.0], vec![1.0]),
        (vec![1.0, -1.0, -1.0], vec![-1.0]),
        (vec![-1.0, 1.0, 1.0], vec![1.0]),
        (vec![-1.0, 1.0, -1.0], vec![-1.0]),
        (vec![-1.0, -1.0, 1.0], vec![-1.0]),
        (vec![-1.0, -1.0, -1.0], vec![-1.0]),
    ];

    for _ in 0..TRAINING_LOOP {
        // let (inputs, targets) = rng.choose(&dataset).unwrap();
        // println!("inputs: {:#?} - targets: {:#?}", inputs, targets);
        for i in dataset.iter_mut() {
            let (inputs, targets) = i;

            nn.train(inputs, targets, LEARNING_RATE);
        }
        // dataset.into_iter().map(|(i, t)| {
        // });
    }

    // let mut user_input = String::new();

    // let stdin = stdin();
    // println!("Please enter the data set you want to be analysed...");
    // stdin.read_line(&mut user_input).unwrap();

    // println!("user input: {}", user_input);

    let predictions = nn.predict(&[-1.0, -1.0, 1.0]);
    println!("Prediction: {:?}", predictions);
}
