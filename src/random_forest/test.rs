#[cfg(test)]

mod tests {
    extern crate test;
    use random_forest::model::RandomForest;
    use ndarray::{RcArray, rcarr2};
    use ndarray_rand::RandomExt;
    use traits::SupervisedLearning;
    use rand::distributions::Range;
    use test::Bencher;
    use rand::{thread_rng, Rng};


    #[test]
    fn test_forest_building() {

        let train = rcarr2(&[[0.0, 1.0], [1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[0.0, 1.0],[0.0, 1.0]]);

        let target = RcArray::from_vec(vec![1.0, 0.0,0.0,0.0,0.0,1.0,1.0]);

        let mut rf = RandomForest :: new(5);

        rf.fit(&train, &target);

        let pred = rf.predict(&train).ok().unwrap();

        assert!( target.all_close(&pred,0.1));

    }


    #[bench]
    fn bench_rf(b: &mut Bencher) {

        let rows = 5;
        let cols = 10;

        let x = RcArray::random((rows,cols), Range::new(0.,10.));
        let mut rng = thread_rng();
        let y = RcArray::from_vec((0..rows)
                                .map(|_| *rng.choose(&vec![0.0, 1.0][..]).unwrap())
                                .collect::<Vec<_>>());

        let mut rf = RandomForest :: new(30);


        b.iter(|| {
            rf.fit(&x, &y);
        });
    }
}
