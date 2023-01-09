use crate::{
    afunctions::activation_functions::ActivationFunctionType, models::perceptrons::Perceptron,
};

pub struct NeuralNetwork {
    layers: Vec<Vec<Perceptron>>,
}

impl NeuralNetwork {
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        // af: ActivationFunctionType,
    ) -> NeuralNetwork {
        let mut layers = Vec::new();
        let mut input_size = input_size;
        for &hidden_size in hidden_sizes {
            let layer = (0..hidden_size)
                .map(|_| Perceptron::new(input_size, ActivationFunctionType::Heaviside))
                .collect();
            layers.push(layer);
            input_size = hidden_size;
        }
        layers.push(
            (0..output_size)
                .map(|_| Perceptron::new(input_size, ActivationFunctionType::Relu))
                .collect(),
        );
        NeuralNetwork { layers }
    }

    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let mut activations = inputs.to_vec();
        for layer in &self.layers {
            activations = layer
                .iter()
                .map(|perceptron| perceptron.predict(&activations))
                .collect();
        }
        activations
    }

    pub fn train(&mut self, inputs: &[f64], targets: &[f64], learning_rate: f64) {
        // assert_eq!(inputs.len(), targets.len());
        let mut activations = inputs.to_vec();
        let mut activations_history = vec![activations.clone()];
        for layer in &mut self.layers {
            activations = layer
                .iter_mut()
                .zip(targets)
                .map(|(perceptron, target)| {
                    let prediction = perceptron.predict(&activations);
                    perceptron.train(&activations, target, learning_rate);
                    prediction
                })
                .collect();
            activations_history.push(activations.clone());
        }
    }
}
