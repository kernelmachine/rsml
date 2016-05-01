#![feature(test)]

extern crate ndarray;
extern crate typenum;
extern crate rand;
extern crate ndarray_rand;
extern crate rayon;
pub mod tree;
pub mod traits;
pub mod random_forest;
pub mod util;


use random_forest::model::*;
use ndarray::OwnedArray;
use ndarray_rand::RandomExt;
use traits::SupervisedLearning;
use rand::distributions::Range;
use rand::{thread_rng, Rng};

fn main() {

    let rows = 2000;
    let cols = 5;

    let X = OwnedArray::random((rows, cols), Range::new(0., 10.));
    let mut rng = thread_rng();
    let y = OwnedArray::from_vec((0..rows)
                                     .map(|_| *rng.choose(&vec![0.0, 1.0][..]).unwrap())
                                     .collect::<Vec<_>>());

    let mut rf = RandomForest::new(10);

    rf.fit(&X, &y);
    let pred = rf.predict(&X).ok().unwrap();

    assert!(y.all_close(&pred, 0.3));

}
