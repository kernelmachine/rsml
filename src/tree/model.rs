
use ndarray::{Ix, ArrayView, Array, Axis};
use traits::SupervisedLearning;

/// Rectangular matrix.
pub type Mat<A> = Array<A, (Ix, Ix)>;

/// Feature view
pub type Feature<'a, A> = ArrayView<'a, A, Ix>;

/// Sample view
pub type Sample<'a, A> = ArrayView<'a, A, Ix>;


/// Col matrix.
pub type Col<A> = Array<A, Ix>;



#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
/// Node represents a node of the decision tree: Internal or Leaf Node.
pub enum Node {
    Internal {
        /// Feature split
        feature: usize,
        /// Feature split threshold
        threshold: f64,
        /// node left and right children, placed on heap.
        children: Box<(Node, Node)>,
    },
    Leaf {
        /// terminal probability that the sample is of class 1.
        probability: f64,
    },
}

/// `DecisionTree` represents the full decision tree model
pub struct DecisionTreeConfig {
    /// maximum depth of the tree
    pub max_depth: u32,
    /// minimum number of samples to split on
    pub min_samples_split: u32,
    /// number of features
    pub n_features: usize,
    /// number of outputs
    pub n_outputs: usize,
    /// array of classes
    pub classes: Vec<f64>,
    /// number of classes
    pub n_classes: usize,
    /// root node of tree
    pub root: Option<Node>,
}

/// `DecisionTree` represents the full decision tree model
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct DecisionTree {
    /// maximum depth of the tree
    max_depth: u32,
    /// minimum number of samples to split on
    min_samples_split: u32,
    /// number of features
    n_features: usize,
    /// number of outputs
    n_outputs: usize,
    /// array of classes
    classes: Vec<f64>,
    /// number of classes
    n_classes: usize,
    /// root node of tree
    root: Option<Node>,
}

impl Default for DecisionTreeConfig {
    fn default() -> DecisionTreeConfig {
        DecisionTreeConfig {
            max_depth: 1024,
            min_samples_split: 2,
            n_features: 2,
            n_outputs: 2,
            classes: vec![0.0, 1.0],
            n_classes: 2,
            root: None,
        }
    }
}

impl DecisionTree {
    /// create new decision tree
    pub fn from_config(cgf: DecisionTreeConfig) -> DecisionTree {
        // initializing this with dummy values. Probably doesn't matter.
        DecisionTree {
            max_depth: cgf.max_depth,
            min_samples_split: cgf.min_samples_split,
            n_features: cgf.n_features,
            n_outputs: cgf.n_outputs,
            classes: cgf.classes,
            n_classes: cgf.n_classes,
            root: cgf.root,
        }
    }

    /// Split a feature column in training data.
    /// # Arguments:
    ///
    /// * `X` - original training data
    /// * `indices` - indices of data subset fed to node
    /// * `feature_idx` - index of current feature to split on
    /// * `threshold` - threshold to apply to current split
    ///
    /// # Example:
    /// ```
    /// extern crate rsml;
    /// extern crate ndarray;
    /// extern crate rand;
    /// extern crate ndarray_rand;
    /// use ndarray_rand::RandomExt;
    /// use ndarray::Array;
    /// use rand::distributions::Range;
    /// use rsml::tree::model::DecisionTree;
    /// fn main(){
    ///     let z = Array::random((10,5), Range::new(0.,10.));
    ///     let feature_idx = 4;
    ///     let value = 4.0;
    ///     let (left, right) = DecisionTree::split(z.column(feature_idx),value);
    ///     assert!(left.iter().all(|&x| z.get((x,feature_idx)).unwrap() <= &value));
    ///     assert!(right.iter().all(|&x| z.get((x,feature_idx)).unwrap() > &value))
    /// }
    /// ```
    pub fn split(feature: Feature<f64>, threshold: f64) -> (Vec<usize>, Vec<usize>) {

        let mut right_idx = Vec::new();
        let mut left_idx = Vec::new();

        for (idx, &elem) in feature.iter().enumerate() {
            match elem {
                x if x <= threshold => right_idx.push(idx),
                x if x > threshold => left_idx.push(idx),
                _ => continue,
            }
        }

        (right_idx, left_idx)
    }

    /// Determine optimal threshold to split data
    /// # Arguments:
    ///
    /// * `X` - original training data
    /// * `feature_idx` - index of current feature to split on
    /// * `y` - original target data
    /// * `indices` - indices of data subset fed to node
    ///
    /// # Example:
    /// ```
    /// extern crate ndarray;
    /// extern crate rsml;
    /// use ndarray :: {arr2, Array};
    /// use rsml::tree::model::*;
    /// fn main(){
    ///     let x = arr2(&[[-1.0], [-0.5], [0.0], [0.0],[0.0],[0.5],[1.0]]);
    ///     let y = Array::from_vec(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    ///     let (threshold, split_impurity) = DecisionTree::find_optimal_split(x.column(0), &y);
    ///     assert!(threshold == -0.5);
    ///     assert!(split_impurity == 0.0);
    /// }
    /// ```
    pub fn find_optimal_split(feature: Feature<f64>, target: &Col<f64>) -> (f64, f64) {
        // This function finds the optimal split for a binary classification problem by
        // sorting the feature vector and then computing gini impurity variables. We sort
        // the vector because it's easier to keep track of left/right child impurities.
        // However, note that this algorithm is tailored to binary classification.
        // If we want to use these decision trees for regression/multiclass stuff,
        // this function will probably have to change.
        let mut split_impurity = 1.0f64 / 0.0f64;
        let mut threshold = 0.0;
        let mut cumulative_y = 0.0;
        let mut cumulative_count = 0.0;
        let mut xy_pairs = feature.iter().zip(target.iter()).collect::<Vec<_>>();

        xy_pairs.sort_by(|a, b| a.0.partial_cmp(b.0).expect("error with sorting x,y pairs"));

        let target_sum = target.scalar_sum();
        for (&x, &y) in xy_pairs {
            cumulative_count += 1.0;
            cumulative_y += y;

            let p_left = cumulative_y / target_sum;
            let p_right = 1.0 - p_left;
            let left_proportion = cumulative_count / target.len() as f64;
            let impurity = DecisionTree::gini_impurity(left_proportion, p_left, p_right);
            if impurity < split_impurity {
                split_impurity = impurity;
                threshold = x;
            }
        }
        (threshold, split_impurity)

    }

    /// Determine the information gain for a split during learning.
    /// # Arguments:
    ///
    /// * `left_proportion` - proportion of target data that map to left child data
    /// * `p_left` - proportion of training data that contain left child data
    /// * `p_right` - proportion of training data that contain right child data
    ///
    /// # Example:
    /// ```
    /// extern crate rsml;
    /// use rsml::tree::model::DecisionTree;
    /// fn main(){
    ///     let impurity = DecisionTree::gini_impurity(0.2, 1.0, 0.5);
    ///     let expected = 0.8 * 0.5;
    ///     assert!(impurity == expected);
    /// }
    /// ```
    pub fn gini_impurity(left_proportion: f64, p_left: f64, p_right: f64) -> f64 {

        let right_proportion = 1.0 - left_proportion;

        let left_impurity = 1.0 - p_left.powi(2) - (1.0 - p_left).powi(2);
        let right_impurity = 1.0 - p_right.powi(2) - (1.0 - p_right).powi(2);

        left_proportion * left_impurity + right_proportion * right_impurity
    }



    /// Build a decision tree on training data.
    /// # Arguments:
    ///
    /// * `X` - training data
    /// * `y` - target data
    /// * `indices` - indices of data subset fed to node
    /// * `depth` - current depth of tree
    pub fn build_tree(&mut self, train: &Mat<f64>, target: &Col<f64>, depth: usize) -> Node {
        // here we calculate the probability that the sample maps to class 1.
        let num_plus = target.scalar_sum();
        let probability = num_plus as f64 / target.len() as f64;

        // if any of these conditions are met, then we terminate the tree.
        if probability.round() == 0.0 || probability.round() == 1.0 ||
           depth > self.max_depth as usize ||
           target.len() < self.min_samples_split as usize {
            return Node::Leaf { probability: probability };
        }

        // otherwise, we try to find the optimal feature and threshold to split on at this node.
        let mut best_feature_idx = 0;
        let mut best_feature_threshold = 0.0 as f64;
        let mut best_impurity = 1.0f64 / 0.0f64;
        for (feature_idx, feature) in train.axis_iter(Axis(1)).enumerate() {
            let (threshold, impurity) = DecisionTree::find_optimal_split(feature, target);

            if impurity < best_impurity {
                best_feature_idx = feature_idx;
                best_feature_threshold = threshold;
                best_impurity = impurity;
            }

        }

        let best_feature = train.column(best_feature_idx);

        // now we split on the feature.
        let (left_data_idx, right_data_idx) = DecisionTree::split(best_feature,
                                                                  best_feature_threshold);
        if !left_data_idx.is_empty() && !right_data_idx.is_empty() {

            let left_train = train.select(Axis(0), &left_data_idx);
            let left_target = target.select(Axis(0), &left_data_idx);
            let right_train = train.select(Axis(0), &right_data_idx);
            let right_target = target.select(Axis(0), &right_data_idx);

            // now we recursively build further nodes on left/right children.
            let left = self.build_tree(&left_train, &left_target, depth + 1);
            let right = self.build_tree(&right_train, &right_target, depth + 1);
            return Node::Internal {
                feature: best_feature_idx,
                threshold: best_feature_threshold,
                children: Box::new((left, right)),
            };

        }


        Node::Leaf { probability: probability }

    }


    /// Traverse tree to retrieve terminal classification probability on new data.
    /// # Arguments:
    ///
    /// * `X` - test data
    /// * `y` - target data
    /// * `row_idx` - row index of test sample in test data
    pub fn query_tree(&self, node: &Node, test: &Sample<f64>) -> f64 {
        match *node {
            Node::Internal { feature, threshold, ref children } => {
                if test[feature] <= threshold {
                    self.query_tree(&children.0, test)
                } else {
                    self.query_tree(&children.1, test)
                }
            }
            Node::Leaf { probability } => probability,
        }
    }
}

impl SupervisedLearning<Mat<f64>, Col<f64>> for DecisionTree {
    fn fit(&mut self, train: &Mat<f64>, target: &Col<f64>) {
        let n_features = train.shape()[1];
        let n_outputs = target.shape()[0];

        // ndarray doesn't have support for uniquing a vector, so here's how I do it.
        let mut target_cloned = target.iter().cloned().collect::<Vec<f64>>();
        target_cloned.sort_by(|a, b| a.partial_cmp(b).unwrap());
        target_cloned.dedup();

        let classes = target_cloned;

        let n_classes = classes.len();

        // you can probably play around with these values.
        let max_depth = 4200;

        let min_samples_split = 4;

        self.max_depth = max_depth;
        self.min_samples_split = min_samples_split;
        self.n_features = n_features;
        self.n_outputs = n_outputs;
        self.classes = classes;
        self.n_classes = n_classes;
        self.root = Some(self.build_tree(train, target, 1));
    }


    fn predict(&self, test: &Mat<f64>) -> Result<Col<f64>, &'static str> {

        match self.root {
            Some(ref node) => {
                // query tree on each row of test data
                let data = test.inner_iter()
                               .map(|x| self.query_tree(node, &x))
                               .collect::<Vec<_>>();
                Ok(Array::from_vec(data))
            }
            None => Err("Fit your tree to some data first!"),
        }

    }
}
