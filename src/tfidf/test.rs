#![allow(non_snake_case)]
#[cfg(test)]

mod tests {
    extern crate test;
    use tfidf::model::*;
    use ndarray::{OwnedArray, arr2};
    use ndarray_rand::RandomExt;
    use traits::SupervisedLearning;
    use rand::distributions::Range;
    use test::Bencher;
    use rand::{thread_rng, Rng};


    #[test]
     fn test_tfidf() {
         let d = vec!["Hi, my name is Suchin.".to_string(), "Woohoo.".to_string()];
         println!("{:?}", fit(d));
        }
}
