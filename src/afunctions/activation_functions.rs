pub enum ActivationFunctionType {
    Heaviside,
    Sigmoid,
    Tanh,
    Relu,
}

pub struct ActivationFunction();

impl ActivationFunction {
    /// Binary step function
    ///
    /// It define a threshold and compare the input to it. If it is below the neuron is not
    /// activated.
    ///
    pub fn heaviside(&self, x: f64) -> f64 {
        if x >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    ///Output range : 0 -> 1
    ///
    pub fn sigmoid(&self, x: f64) -> f64 {
        use std::f64::consts::E;

        1. / 1. + E.powf(-x)
    }

    /// Output range : -1.0 -> 1.0
    ///
    /// Output of the tanh function is 0 centred. As it means that we can see if value is strongly
    /// positive or strongly negative.
    ///
    pub fn tanh(&self, x: f64) -> f64 {
        use std::f64::consts::E;

        (E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x))
    }

    ///The neurons will only be deactivated if the output of the linear transformation is less than 0.
    ///
    pub fn relu(&self, x: f64) -> f64 {
        use std::cmp;

        cmp::max(0.0 as i64, x as i64) as f64
    }
}
