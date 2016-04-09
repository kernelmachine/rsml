use ndarray::{RcArray,Ix};


/// Rectangular matrix.
pub type Mat<A> = RcArray<A, (Ix, Ix)>;
/// Col matrix.
pub type Col<A> = RcArray<A, Ix>;


pub trait SupervisedLearning<A,B> {
    fn fit(&mut self, X : &A, y: &B);
    fn decision(&mut self, X : A);
    fn predict(&mut self, X :A)-> Result<Col<f64>, &'static str>;
}
