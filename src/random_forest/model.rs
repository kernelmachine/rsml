#[allow(non_snake_case)]
use ndarray::{Ix, ArrayView, OwnedArray};
use rand::distributions::{IndependentSample, Range};

use rand::StdRng;
use traits::SupervisedLearning;
use tree::model::{DecisionTree, DecisionTreeConfig};
use util::util::{noncontig_1d_slice, noncontig_2d_slice};
use rayon::prelude::*;

/// Rectangular matrix.
pub type Mat<A> = OwnedArray<A, (Ix, Ix)>;

/// Feature view
pub type Feature<'a, A> = ArrayView<'a, A, Ix>;

/// Feature view
pub type Sample<'a, A> = ArrayView<'a, A, Ix>;


/// Col matrix.
pub type Col<A> = OwnedArray<A, Ix>;



/// This represents the Random Forest.
pub struct RandomForest {
    /// Vector of Decision Trees
    pub trees: Vec<DecisionTree>,
}

impl RandomForest {
    /// Create a new random forest
    /// # Arguments:
    ///
    /// * `n_estimators` - number of decision trees you want
    pub fn new(n_estimators: usize) -> RandomForest {

        RandomForest {
            trees: vec![DecisionTree::from_config(DecisionTreeConfig::default(), 32); n_estimators],
        }

    }

    /// get a random set of indices to map random samples to a decision tree in your forest
    /// # Arguments:
    ///
    /// * `num_indices` - number of samples in the training data
    pub fn bootstrap(num_samples: usize) -> Vec<usize> {

        let range = Range::new(0, num_samples);

        (0..num_samples)
            .map(|_| {
                range.ind_sample(&mut StdRng::new().expect("Error with random number generator"))
            })
            .collect::<Vec<_>>()

    }
}


impl SupervisedLearning<Mat<f64>, Col<f64>> for RandomForest {
    fn fit(&mut self, train: &Mat<f64>, target: &Col<f64>) {
        for tree in self.trees.iter_mut() {

            // get random set of indices
            let indices = RandomForest::bootstrap(train.rows());
            // println!("{:?}", indices);
            // sample data
            let train_subset = noncontig_2d_slice(train, &indices);
            let target_subset = noncontig_1d_slice(target, &indices);

            // train decision tree
            tree.fit(&train_subset, &target_subset);

        }

    }

    fn predict(&self, test: &Mat<f64>) -> Result<Col<f64>, &'static str> {
        // prediction in a random forest is just taking the average of the output of a set of
        // decision trees trained on random subsets of the data.


        let mut df = OwnedArray::zeros(test.shape()[0]);
        for tree in self.trees.iter().cloned() {
            if let Ok(pred) = tree.predict(test) {
                df = df + pred;
            }
        }

        df = df / (self.trees.len() as f64);
        Ok(df)


    }
}
