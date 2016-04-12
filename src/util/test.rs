#[cfg(test)]

mod tests {
    use util::util::*;
    use ndarray::{OwnedArray, arr2};

    #[test]
     fn test_noncontig_1d_slice() {
        let y = OwnedArray::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let s = noncontig_1d_slice(&y,&vec![0,2,4,6]);
        assert!(s.all_close(&OwnedArray::from_vec(vec![1.0, 1.0, 1.0, 1.0]),1e-8))
        }

    #[test]
     fn test_noncontig_2d_slice() {
        let x = arr2(&[[0.0, 1.0], [1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[0.0, 1.0],[0.0, 1.0]]);
        let s = noncontig_2d_slice(&x,&vec![1,3,5]);
        let target = arr2(&[[1.0,0.0],[1.0,0.0],[0.0, 1.0]]);
        assert!(s.all_close(&target,1e-8))
        }
    }
