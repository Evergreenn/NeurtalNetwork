use rand::*;

pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    pub fn new(n: usize) -> Perceptron {
        let mut rng = thread_rng();
        let weights = (0..n).map(|_| rng.gen_range(-1.0, 1.0)).collect();
        Perceptron {
            weights,
            bias: rng.gen_range(-1.0, 1.0),
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
        if activation >= 0.0 {
            1.0
        } else {
            -1.0
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
