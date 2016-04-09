#![allow(unused_imports)]
#![allow(dead_code)]

#![allow(non_snake_case)]
#![allow(unused_variables)]

use ndarray::{RcArray,Ix, Axis, ArrayBase, arr2};
use std::cmp;
use rand::{thread_rng, Rng};
use traits::SupervisedLearning;
/// Rectangular matrix.
pub type Mat<A> = RcArray<A, (Ix, Ix)>;
/// Col matrix.
pub type Col<A> = RcArray<A, Ix>;



#[derive(Debug)]
pub enum Node {
    Interior {
        feature : usize,
        threshold : f64,
        children : Box <(Node, Node)>
    },
    Leaf {
        probability : f64
    }
}

#[derive(Debug)]
pub struct DecisionTree{
    criterion : String,
    splitter : String,
    max_depth : i32,
    min_samples_split : i32,
    min_samples_leaf : i32,
    min_weight_fraction_leaf : f64,
    max_features : usize,
    max_leaf_nodes : i32,
    n_features : usize,
    n_outputs : usize,
    classes : Col<f64>,
    n_classes : usize,
    root: Option<Node>
}

impl DecisionTree {
    pub fn new() -> DecisionTree {
        DecisionTree {
            criterion : "gini".to_string(),
            splitter : "best".to_string(),
            max_depth : 50 ,
            min_samples_split : 2,
            min_samples_leaf : 0  ,
            min_weight_fraction_leaf : 0.0,
            max_features : 0,
            max_leaf_nodes : 0,
            n_features : 0,
            n_outputs : 0,
            classes : RcArray::from_vec(vec![0.0]),
            n_classes : 0,
            root : None
        }
    }
    pub fn count_positives(y : &Col<f64>, indices : &Col<usize> ) -> i32{
        let mut count = 0;

        let data = y.iter().cloned().collect::<Vec<f64>>();

        for row_idx in indices.iter().cloned() {
            if data[row_idx] > 0.0 {
                count += 1;
            }
        }
        // println!("{:?}", count);
        count

    }

    pub fn split_data(X : &Mat<f64>, column : usize, value : f64) -> (Col<f64>,Col<f64>, Col<usize>,Col<usize>){
        let mut a = Vec :: new();
        let mut b = Vec :: new();
        let mut a_idx = Vec :: new();
        let mut b_idx = Vec :: new();

        for (idx, elem) in X.column(column).indexed_iter(){
            match elem {
                x if x <= &value =>{ a.push(*elem); a_idx.push(idx)}
                x if x > &value => {b.push(*elem); b_idx.push(idx)}
                _ => continue
            }
        }
        (RcArray::from_vec(a),RcArray::from_vec(b),RcArray::from_vec(a_idx),RcArray::from_vec(b_idx))
    }




    pub fn build_tree(X: &Mat<f64>, y: &Col<f64>,
                 indices : &Col<usize>,
                depth: usize, max_depth: i32, min_samples_split: i32,
                ) -> Node {

        let num_positives : i32 = DecisionTree::count_positives(&y, &indices);
        let probability = num_positives as f64 / indices.len() as f64;
        if probability == 0.0
            || probability == 1.0
            || depth > max_depth as usize
            || indices.len() < min_samples_split as usize {

                return Node::Leaf{probability: probability};
            }

            let mut best_feature_idx = 0;
            let mut best_feature_threshold = 0.0 as f64;
            let mut best_impurity = 1.0f64 / 0.0f64;

            for feature_idx in 0..X.shape()[1] {

                let (threshold, impurity) = DecisionTree::calculate_split(X, feature_idx, &y, indices);

                if impurity < best_impurity {
                    best_feature_idx = feature_idx;
                    best_feature_threshold = threshold;
                    best_impurity = impurity;
                }
            }

            let (left_data, right_data, left_data_idx, right_data_idx) = DecisionTree::split_data(X, best_feature_idx, best_feature_threshold);
            if left_data.len() > 0 && right_data.len() > 0 {

                    let left = DecisionTree::build_tree(X, &y,
                                             &left_data_idx,
                                             depth + 1, max_depth, min_samples_split);
                     let right = DecisionTree::build_tree(X, &y,
                                              &right_data_idx,
                                               depth + 1, max_depth, min_samples_split);
                    println!("{:?}", depth );
                    return Node::Interior {feature: best_feature_idx,
                                       threshold: best_feature_threshold,
                                       children: Box::new((left,
                                                           right))}

            }


        Node::Leaf {probability: probability}

    }

    pub fn calculate_split(X : &Mat<f64>,feature_idx : usize, y : &Col<f64>, indices : &Col<usize>) -> (f64, f64) {

            let mut values = Vec :: new();
            let features = X.column(feature_idx);
            let data = y.iter().cloned().collect::<Vec<f64>>();

            for row_idx in indices.iter().cloned() {
                values.push(data[row_idx])
            }

            let sorted_values = values.sort_by(|a, b| a.partial_cmp(&b).unwrap());
            let max_value = values.last().unwrap();

            let mut split_impurity = 1.0f64 / 0.0f64;
            let mut split_x = 0.0;

            let total_count = values.len() as f64;
            let total_y = data.iter().fold(0.0,|a, &b| a + b);

            let mut cumulative_count = 0.0;
            let mut cumulative_y = 0.0;

            for (&x, &y) in features.iter().zip(values.iter()) {


                cumulative_count += 1.0;
                cumulative_y += y;


                if x == *max_value {
                    continue;
                }

                let left_child_proportion = cumulative_count / total_count;
                let left_child_positive_probability = cumulative_y / cumulative_count;
                let right_child_positive_probability = (total_y - cumulative_y) /
                                                       (total_count - cumulative_count);

                let impurity = DecisionTree::proxy_gini_impurity(left_child_proportion,
                                                                 left_child_positive_probability,
                                                                 right_child_positive_probability);

                // It's important that this is less than or equal rather
                // than less than: subject to no decrease in impurity
                // it's always good to move to a split at a higher value.
                if impurity <= split_impurity {
                    split_impurity = impurity;
                    split_x = x;
                }

            }
            (split_x, split_impurity)
        }

        pub fn proxy_gini_impurity(left_child_proportion: f64,
                               left_child_probability: f64,
                               right_child_probability: f64)
                               -> f64 {

            let right_child_proportion = 1.0 - left_child_proportion;

            let left_impurity = 1.0 - left_child_probability.powi(2) -
                                (1.0 - left_child_probability).powi(2);
            let right_impurity = 1.0 - right_child_probability.powi(2) -
                                 (1.0 - right_child_probability).powi(2);

            left_child_proportion * left_impurity + right_child_proportion * right_impurity
        }

        pub fn query_tree(&self, node: &Node, X: &Mat<f64>, row_idx: usize) -> f64 {
           match node {
               &Node::Interior {feature,
                                threshold,
                                ref children} => {
                   match X.get((row_idx, feature)).unwrap() <= &threshold {
                       true => self.query_tree(&children.0, X, row_idx),
                       false => self.query_tree(&children.1, X, row_idx),
                   }
               }
               &Node::Leaf {probability} => probability,
           }
       }


}

impl SupervisedLearning<Mat<f64>, Col<f64>> for DecisionTree{


    fn fit(&mut self, X : &Mat<f64>, y : &Col<f64>)  {
        let n_features = X.shape()[1];
        let n_outputs = y.shape()[0];

        let mut y_cloned = y.iter().cloned().collect::<Vec<f64>>();

        y_cloned.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        y_cloned.dedup();
        let classes = RcArray::from_vec(y_cloned);
        let n_classes = classes.shape()[0];

        let x: i32 = 2;

        let max_depth = 1000;

        let max_leaf_nodes = -1;
        let min_samples_leaf = 1;
        let min_samples_split = 2;

        let max_features = n_features;
        let min_weight_fraction_leaf = 0.0;
        self.criterion =  "gini".to_string();
        self.splitter = "best".to_string();
        self.max_depth =max_depth;
        self.min_samples_split =min_samples_split;
        self.min_samples_leaf=min_samples_leaf;
        self.min_weight_fraction_leaf=min_weight_fraction_leaf;
        self.max_features=max_features;
        self.max_leaf_nodes=max_leaf_nodes;
        self.n_features=n_features;
        self.n_outputs=n_outputs;
        self.classes=classes;
        self.n_classes=n_classes;

        self.root= Some(DecisionTree::build_tree(&X, &y,
                    & RcArray::from_vec((0..X.shape()[1]).collect()), 1, max_depth, min_samples_split));


    }
    fn decision(&mut self, X : Mat<f64>){
        unimplemented!();
    }
    fn predict(&mut self, X : Mat<f64>) -> Result<Col<f64>, &'static str>{
        match self.root {
            Some(ref node) => {
                let mut data = Vec::with_capacity(X.shape()[0]);
                for row_idx in 0..X.shape()[0] {
                    data.push(self.query_tree(&node, &X, row_idx));
                }
                Ok(RcArray::from_vec(data))
            }
            None => Err("Tree must be built before predicting"),
        }
    }



}
