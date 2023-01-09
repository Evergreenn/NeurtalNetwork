use rand::*;

mod models;
mod neurals;

use crate::neurals::network::NeuralNetwork;

const TRAINING_LOOP: usize = 1_000;
const DATASET_SIZE: usize = 10_000;

fn generate_dataset(size: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rng = thread_rng();

    (0..size)
        .map(|_| {
            let input: Vec<f64> = (0..10).map(|_| rng.gen_range(-1.0, 1.0)).collect();
            let target: Vec<f64> = (0..1).map(|_| rng.gen_range(-1.0, 1.0)).collect();
            (input, target)
        })
        .collect()
}

fn main() {
    let mut nn = NeuralNetwork::new(6, &[10], 1);
    let mut rng = thread_rng();
    let dataset = generate_dataset(DATASET_SIZE);

    for _ in 0..TRAINING_LOOP {
        let (inputs, targets) = rng.choose(&dataset).unwrap();
        nn.train(inputs, targets, 0.1);
    }

    let predictions = nn.predict(&[0.0, 1.0]);
    println!("Prediction: {:?}", predictions);
}
