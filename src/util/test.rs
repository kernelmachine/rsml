#![allow(non_snake_case)]
#[cfg(test)]

mod tests {
    use util::util::*;
    use ndarray::{RcArray, rcarr2};
    use ndarray_rand::RandomExt;
    use traits::SupervisedLearning;
    use rand::distributions::Range;
    use test::Bencher;
    use rand::{thread_rng, Rng};


    #[test]
     fn test_noncontig_1d_slice() {
        let y = RcArray::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let s = noncontig_1d_slice(&y,&vec![0,2,4,6]);
        assert!(s.all_close(&RcArray::from_vec(vec![1.0, 1.0, 1.0, 1.0]),1e-8))
        }
        
    #[test]
     fn test_noncontig_2d_slice() {
        let X = rcarr2(&[[0.0, 1.0], [1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[0.0, 1.0],[0.0, 1.0]]);
        let s = noncontig_2d_slice(&X,&vec![1,3,5]);
        let target = rcarr2(&[[1.0,0.0],[1.0,0.0],[0.0, 1.0]]);
        assert!(s.all_close(&target,1e-8))
        }
    }
