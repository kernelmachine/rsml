#![allow(non_snake_case)]
#[cfg(test)
]
mod tests {
    use tree::model::*;
    use ndarray::{RcArray, rcarr2};
    use ndarray_rand::RandomExt;
    use traits::SupervisedLearning;
    use rand::distributions::Range;

    #[test]
     fn test_gini_impurity() {
         let impurity = DecisionTree::gini_impurity(0.5, 0.5, 0.5);
         let expected = 0.5;
         assert!(impurity == expected);

         let impurity = DecisionTree::gini_impurity(0.5, 1.0, 0.0);
         let expected = 0.0;
         assert!(impurity == expected);

         let impurity = DecisionTree::gini_impurity(0.2, 1.0, 0.5);
         let expected = 0.8 * 0.5;
         assert!(impurity == expected);
        }


    #[test]
    fn test_split_data(){
        let X = RcArray::random((10,5), Range::new(0.,10.));
        let indices = RcArray::from_vec(vec![0,1,2,3,4,5,6,7,8,9]);
        let feature_idx = 4;
        let value = 4.0;
        let (left, right) = DecisionTree::split_data(&X,&indices, feature_idx, value);
        assert!(left.iter().all(|&x| X.get((x,feature_idx)).unwrap() <= &value));
        assert!(right.iter().all(|&x| X.get((x,feature_idx)).unwrap() > &value))
    }

    #[test]
    fn test_calculate_split(){

        let X = rcarr2(&[[-1.0], [-0.5], [0.0], [0.0],[0.0],[0.5],[1.0]]);

        let y = RcArray::from_vec(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let indices = RcArray::from_vec(vec![0,1,2,3,4,5,6]);
        let (threshold, split_impurity) = DecisionTree::calculate_split(&X, 0, &y, &indices);

        assert!(threshold == -0.5);
        assert!(split_impurity == 0.0);

    }

    #[test]
    fn test_calculate_split_1(){
        let X = rcarr2(&[[-1.0], [-0.5], [0.0], [0.0],[0.0],[0.5],[1.0]]);
        let y = RcArray::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        let indices = RcArray::from_vec(vec![0,1,2,3,4,5,6]);
        let (threshold, split_impurity) = DecisionTree::calculate_split(&X, 0, &y, &indices);
        assert!(threshold == 0.0);
        assert!(split_impurity == 0.0);

    }

    #[test]
    fn test_tree_building() {

        let X = rcarr2(&[[0.0, 1.0], [1.0,0.0] ]);

        let y = RcArray::from_vec(vec![0.0, 1.0]);

        let mut dt = DecisionTree::new();

        dt.fit(&X, &y);

        let pred = dt.predict(X).ok().unwrap();
        assert!( y.all_close(&pred, 0.5));

    }

}
