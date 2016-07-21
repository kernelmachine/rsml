
use ndarray::{Array, Ix, Axis, ArrayView, stack};


/// Rectangular matrix.
pub type Mat<A> = Array<A, (Ix, Ix)>;

/// Feature view
pub type Feature<'a, A> = ArrayView<'a, A, Ix>;

/// Col matrix.
pub type Col<A> = Array<A, Ix>;




/// Get a noncontinguous slice of a 1-dimensional ndarray.
/// # Arguments:
///
/// * `mat` - 1 dimensional matrix
/// * `indices` - row indices you want to slice on
///
/// # Example:
/// ```
/// extern crate rsml;
/// extern crate ndarray;
/// use rsml::util::util::*;
/// use ndarray::Array;
/// fn main(){
/// let y = Array::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
/// let s = noncontig_1d_slice(&y,&vec![0,2,4,6]);
/// let target = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
/// assert!(s.all_close(&target,1e-8))
/// }
/// ```

pub fn noncontig_1d_slice(mat: &Col<f64>, indices: &[usize]) -> Col<f64> {
    Array::from_vec(indices.iter()
                           .cloned()
                           .collect::<Vec<_>>()
                           .iter()
                           .map(|&x| mat[x])
                           .collect::<Vec<_>>())

}

/// Get a noncontinguous slice of a 2-dimensional ndarray.
/// # Arguments:
///
/// * `mat` - 2 dimensional matrix
/// * `indices` - row indices you want to slice on
///
/// # Example:
/// ```
/// extern crate rsml;
/// extern crate ndarray;
/// use rsml::util::util::*;
/// use ndarray::arr2;
/// fn main(){
/// let x = arr2(&[[0.0, 1.0], [1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[0.0, 1.0],[0.0, 1.0]]);
/// let s = noncontig_2d_slice(&x,&vec![1,3,5]);
/// let target = arr2(&[[1.0,0.0],[1.0,0.0],[0.0, 1.0]]);
/// assert!(s.all_close(&target,1e-8))
/// }
/// ```
pub fn noncontig_2d_slice(mat: &Mat<f64>, indices: &[usize]) -> Mat<f64> {
    let mat = indices.iter()
                     .map(|&x| {
                         mat.row(x)
                            .into_shape((1, mat.shape()[1]))
                            .unwrap_or_else(|e| panic!("Indexing Error {:?}", e))
                     })
                     .collect::<Vec<_>>();
    stack(Axis(0), mat.as_slice()).unwrap_or_else(|e| panic!("Row stacking error {:?}", e))
}
