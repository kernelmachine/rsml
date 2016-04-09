#![allow(non_snake_case)]
#[cfg(test)
]
mod tests {
    use ensemble::model::*;
    use ndarray::{RcArray};
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
    fn test_basic_tree_building() {
        let X = RcArray::random((7, 5), Range::new(0., 10.));
        let y = RcArray::from_vec(vec![1.0, 1.0,1.0,1.0,0.0,0.0,0.0]);

        let mut dt = DecisionTree::new();
        dt.fit(&X, &y);
        let pred = dt.predict(X);
        println!("{:?}", pred.ok().unwrap())

    }
}
