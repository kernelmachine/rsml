#![allow(non_snake_case)]

#[cfg(test)]

mod tests {
    extern crate test;
    use tree::model::*;
    use ndarray::{OwnedArray, arr2};
    use ndarray_rand::RandomExt;
    use traits::SupervisedLearning;
    use rand::distributions::Range;
    use self::test::Bencher;
    use rand::{thread_rng, Rng};


    #[test]
    fn test_gini_impurity() {
        let impurity = DecisionTree::gini_impurity(0.5, 0.5, 0.5);
        let expected = 0.5;
        assert!(impurity == expected);

        let impurity = DecisionTree::gini_impurity(0.8, 0.4, 0.6);
        let expected = 0.48;
        assert!(impurity == expected);

        let impurity = DecisionTree::gini_impurity(0.3, 1.0, 0.0);
        let expected = 0.0;
        assert!(impurity == expected);
    }


    #[test]
    fn test_split() {
        let X = OwnedArray::random((10, 5), Range::new(0., 10.));
        let feature_idx = 4;
        let value = 4.0;
        let (left, right) = DecisionTree::split(X.column(feature_idx), value);
        assert!(left.iter().all(|&x| X.get((x, feature_idx)).unwrap() <= &value));
        assert!(right.iter().all(|&x| X.get((x, feature_idx)).unwrap() > &value))
    }

    #[test]
    fn test_calculate_split() {

        let X = arr2(&[[-8.0], [-2.0], [0.0], [0.0], [0.0], [0.2], [1.0]]);

        let y = OwnedArray::from_vec(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let (threshold, split_impurity) = DecisionTree::find_optimal_split(X.column(0), &y);
        assert!(threshold == -2.0);
        assert!(split_impurity == 0.0);

    }

    #[test]
    fn test_calculate_split_1() {

        let X = arr2(&[[-8.0], [-2.0], [0.0], [0.0], [0.0], [0.2], [1.0]]);

        let y = OwnedArray::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);

        let (threshold, split_impurity) = DecisionTree::find_optimal_split(X.column(0), &y);

        assert!(threshold == 0.0);
        assert!(split_impurity == 0.0);

    }

    #[test]
    fn test_tree_building() {

        let X = arr2(&[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                       [0.0, 1.0]]);

        let y = OwnedArray::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

        let mut dt = DecisionTree::from_config(DecisionTreeConfig::default());

        dt.fit(&X, &y);

        let pred = dt.predict(&X).ok().unwrap();

        assert!(y.all_close(&pred, 1e-8));

    }

    #[test]
    fn break_tree() {
        let rows = 500;
        let cols = 20;

        let X = OwnedArray::random((rows, cols), Range::new(0., 10.));
<<<<<<< HEAD

        let mut rng = thread_rng();
        let y = OwnedArray::from_vec((0..rows)
                                         .map(|_| *rng.choose(&vec![0.0, 1.0][..]).unwrap())
                                         .collect::<Vec<_>>());

        let mut dt = DecisionTree::from_config(DecisionTreeConfig::default());
        dt.fit(&X, &y);
    }


    #[bench]
    fn bench_predict_tree(b: &mut Bencher) {

        let rows = 20;
        let cols = 20;

        let X = OwnedArray::random((rows, cols), Range::new(0., 10.));

=======
>>>>>>> 95176b96e5bd41e7ebe0d51b380c7558dd6cf237
        let mut rng = thread_rng();
        let y = OwnedArray::from_vec((0..rows)
                                         .map(|_| *rng.choose(&vec![0.0, 1.0][..]).unwrap())
                                         .collect::<Vec<_>>());

        let mut dt = DecisionTree::from_config(DecisionTreeConfig::default());
        dt.fit(&X, &y);
    }


    #[bench]
    fn bench_tree(b: &mut Bencher) {

        let rows = 20;
        let cols = 20;

        let X = OwnedArray::random((rows, cols), Range::new(0., 10.));
        let mut rng = thread_rng();
        let y = OwnedArray::from_vec((0..rows)
                                         .map(|_| *rng.choose(&vec![0.0, 1.0][..]).unwrap())
                                         .collect::<Vec<_>>());

        let mut dt = DecisionTree::from_config(DecisionTreeConfig::default());


        b.iter(|| {
            dt.fit(&X, &y);
        });
    }

}
