use ndarray::{OwnedArray,Ix};


/// Rectangular matrix.
pub type Mat<A> = OwnedArray<A, (Ix, Ix)>;
/// Col matrix.
pub type Col<A> = OwnedArray<A, Ix>;

/// Trait for supervised learning.
pub trait SupervisedLearning<A,B> {
    /// Fit training data to decision tree
    /// # Arguments:
    ///
    /// * `X` - training data data
    /// * `y` - target data
    fn fit(&mut self, X : &A, y: &B);

    /// Predict on test data
    /// # Arguments:
    ///
    /// * `X` - test data
    fn predict(&self, X :&A)-> Result<Col<f64>, &'static str>;
}
