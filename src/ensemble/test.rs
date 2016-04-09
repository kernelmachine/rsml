#![allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use ensemble::model::*;
    use ndarray::{rcarr2, RcArray};
    use traits::SupervisedLearning;
    #[test]
     fn test_gini_impurity() {
         let impurity = DecisionTree::proxy_gini_impurity(0.5, 0.5, 0.5);
         let expected = 0.5;
         assert!(impurity == expected);

         let impurity = DecisionTree::proxy_gini_impurity(0.5, 1.0, 0.0);
         let expected = 0.0;
         assert!(impurity == expected);

         let impurity = DecisionTree::proxy_gini_impurity(0.2, 1.0, 0.5);
         let expected = 0.8 * 0.5;
         assert!(impurity == expected);
        }
    #[test]
    fn test_basic_tree_building() {

        let X = rcarr2(&[[1.,1.,1.,1.,1.],
                        [10., 1.,200.,-3.,4.],
                        [10., 1.,200.,-3.,4.],
                        [10., 1.,200.,-3.,4.],
                        [10., 1.,200.,-3.,4.],
                        [10., 1.,200.,-3.,4.],
                        [20., 0.,10.,3.,8.],
                        [1., 0.,10.,3.,8.],
                        [0., 1.,2.,3.,4.],
                        [1., 1.,2.,3.,4.]]);
        let y = RcArray::from_vec(vec![1.0, 0.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0]);

        let mut dt = DecisionTree::new();
        dt.fit(&X, &y);
        let pred = dt.predict(X);
        println!("{:?}", pred.ok().unwrap())

    }
}
