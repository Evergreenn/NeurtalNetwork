use crate::afunctions::activation_functions::{ActivationFunction, ActivationFunctionType};
use rand::*;

pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    activation_function: ActivationFunctionType,
}

impl Perceptron {
    pub fn new(n: usize, activation_function_name: ActivationFunctionType) -> Perceptron {
        let mut rng = thread_rng();
        let weights = (0..n).map(|_| rng.gen_range(0.0, 1.0)).collect();
        Perceptron {
            weights,
            bias: rng.gen_range(-1.0, 1.0),
            activation_function: activation_function_name,
        }
    }

    pub fn predict(&self, inputs: &[f64]) -> f64 {
        // assert_eq!(inputs.len(), self.weights.len());
        let activation = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias;
        // println!("activation: {}", activation);

        match self.activation_function {
            ActivationFunctionType::Heaviside => ActivationFunction().heaviside(activation),
            ActivationFunctionType::Sigmoid => ActivationFunction().sigmoid(activation),
            ActivationFunctionType::Tanh => ActivationFunction().tanh(activation),
            ActivationFunctionType::Relu => ActivationFunction().relu(activation),
        }
    }

    pub fn train(&mut self, inputs: &[f64], target: &f64, learning_rate: f64) {
        let prediction = self.predict(inputs);
        let error = target - prediction;
        self.weights = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w + error * learning_rate * x)
            .collect();
        self.bias += error * learning_rate;
    }
}
