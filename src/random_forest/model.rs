#![allow(non_snake_case)]
extern crate serde_json;
extern crate serde;
extern crate rustc_serialize;

use ndarray::{Ix, ArrayView, Array, Axis};
use rand::distributions::{IndependentSample, Range};

use rand::StdRng;
use traits::SupervisedLearning;
use tree::model::{DecisionTree, DecisionTreeConfig};
use ndarray_rand::RandomExt;
/// Rectangular matrix.
pub type Mat<A> = Array<A, (Ix, Ix)>;

/// Feature view
pub type Feature<'a, A> = ArrayView<'a, A, Ix>;

/// Feature view
pub type Sample<'a, A> = ArrayView<'a, A, Ix>;


/// Col matrix.
pub type Col<A> = Array<A, Ix>;



/// This represents the Random Forest.
#[derive(Debug, RustcDecodable, RustcEncodable)]
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
            trees: vec![DecisionTree::from_config(DecisionTreeConfig::default()); n_estimators],
        }

    }

    /// get a random set of indices to map random samples to a decision tree in your forest
    /// # Arguments:
    ///
    /// * `num_indices` - number of samples in the training data
    pub fn bootstrap(range: Range<usize>, num_samples: usize) -> Vec<usize> {

        (0..num_samples)
            .map(|_| {
                range.ind_sample(&mut StdRng::new().expect("Error with random number generator"))
            })
            .collect::<Vec<_>>()

    }
}


impl SupervisedLearning<Mat<f64>, Col<f64>> for RandomForest {
    fn fit(&mut self, train: &Mat<f64>, target: &Col<f64>) {
        let index_matrix = Array::random((train.rows(), self.trees.len()),
                                         Range::new(0, train.rows()));


        for i in 0..self.trees.len() {
            // get random set of indices
            // let indices = RandomForest::bootstrap(train.rows());
            // sample data
            let ind_col = index_matrix.column(i);
            let indices = ind_col.to_owned();
            let train_subset = train.select(Axis(0), indices.as_slice().unwrap());


            let target_subset = target.select(Axis(0), indices.as_slice().unwrap());


            // train decision tree
            self.trees[i].fit(&train_subset, &target_subset);

        }

    }

    fn predict(&self, test: &Mat<f64>) -> Result<Col<f64>, &'static str> {
        // prediction in a random forest is just taking the average of the output of a set of
        // decision trees trained on random subsets of the data.


        let mut df = Array::zeros(test.shape()[0]);
        for tree in self.trees.iter().cloned() {
            if let Ok(pred) = tree.predict(test) {
                df = df + pred;
            }
        }
        df = df / (self.trees.len() as f64);
        Ok(df)


    }
}
