
use ndarray::{RcArray,Ix, Axis, ArrayView, stack};


/// Rectangular matrix.
pub type Mat<A> = RcArray<A, (Ix, Ix)>;

/// Feature view
pub type Feature<'a,A> = ArrayView<'a,A, Ix>;

/// Col matrix.
pub type Col<A> = RcArray<A, Ix>;




pub fn noncontig_2d_slice(mat : &Mat<f64>, indices : &Vec<usize>)  -> Mat<f64> {
    let mat = indices.iter()
                        .map(|&x| mat.row(x).into_shape((1,mat.shape()[1]))
                        .ok().unwrap()).collect::<Vec<_>>();
    stack(Axis(0), mat.as_slice()).ok().unwrap().to_shared()
}

pub fn noncontig_1d_slice(mat : &Col<f64>, indices : &Vec<usize>) -> Col<f64>{
    RcArray::from_vec(indices
                        .iter().cloned().collect::<Vec<_>>()
                        .iter().map(|&x| mat[x]).collect::<Vec<_>>())

}
