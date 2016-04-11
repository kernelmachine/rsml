#[allow(unused_imports)]
#[allow(dead_code)]
#[allow(non_snake_case)]
use ndarray::{RcArray,Ix, Axis, ArrayView, ViewRepr, arr2, stack};
use rand::distributions::{IndependentSample, Range};

use rand::StdRng;
use traits::SupervisedLearning;
use tree::model::DecisionTree;

/// Rectangular matrix.
pub type Mat<A> = RcArray<A, (Ix, Ix)>;

/// Feature view
pub type Feature<'a,A> = ArrayView<'a,A, Ix>;

/// Col matrix.
pub type Col<A> = RcArray<A, Ix>;




pub struct RandomForest {
    trees : Vec<DecisionTree>,
}

impl RandomForest {

    pub fn new(n_estimators : usize) -> RandomForest {
        let trees = vec![DecisionTree::new(); n_estimators];
        RandomForest {
            trees :  trees
        }
    }

    pub fn bootstrap_indices(num_indices: usize) -> Vec<usize> {
        let range = Range::new(0, num_indices);

        (0..num_indices)
            .map(|_| range.ind_sample(&mut StdRng::new().unwrap()))
            .collect::<Vec<_>>()
    }
}


impl SupervisedLearning<Mat<f64>, Col<f64>> for RandomForest {
    fn fit(&mut self, X: &Mat<f64>, y: &Col<f64>) {


        for tree in self.trees.iter_mut() {
            let indices = RandomForest::bootstrap_indices(X.rows());
            let x_subset = indices.iter().map(|&x| X.row(x).into_shape((1,X.shape()[1])).ok().unwrap()).collect::<Vec<_>>();
            let x_subset = stack(Axis(0), x_subset.as_slice()).ok().unwrap();

            let y_subset = RcArray::from_vec(indices.iter().cloned().collect::<Vec<_>>().iter().map(|&x| y[x]).collect::<Vec<_>>());

            tree.fit(&x_subset.to_shared(), &y_subset);
        }


    }

    fn predict(&mut self, X: &Mat<f64>) -> Result<Col<f64>, &'static str> {

        let mut df = RcArray::zeros(X.shape()[0]);


        for mut tree in self.trees.iter().cloned() {
            df = df + tree.predict(X).ok().unwrap();
        }

        df = df / (self.trees.len() as f64);

        Ok(df)

    }
}
