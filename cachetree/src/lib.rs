#![allow(dead_code, unused_variables)]
// Used to track whether a bool exists

#[derive(Debug, PartialOrd, PartialEq)]
pub enum Node<T: PartialOrd> {
    Branch {
        value: T,
    },
    None,
}

impl<T: PartialOrd> Node<T> {
    pub fn new(value: T) -> Node<T> {
        Node::Branch { value: value }
    }
}

#[allow(match_same_arms)]
impl<T: PartialOrd> Node<T> {
    pub fn get_value(&self) -> &T {
        match *self {
            Node::Branch { ref value, .. } => value,
            Node::None => panic!("Tried to get value of a None node"),
        }
    }
}

#[derive(Debug)]
pub struct FlatTree<T: PartialOrd> {
    nodes: Vec<Node<T>>,
}

impl<T: PartialOrd> FlatTree<T> {
    #[allow(new_without_default)]
    pub fn new(max_depth: usize, root: T) -> FlatTree<T> {
        let max_depth = max_depth.pow(2) - 1;
        let mut nodes = Vec::with_capacity(max_depth);

        for _ in 0..max_depth {
            nodes.push(Node::None);
        }
        let root = Node::new(root);

        nodes[0] = root;

        FlatTree { nodes: nodes }
    }

    pub fn add_node(&mut self, new_node: T) {
        let parent_index = self.find_parent(&new_node);
        let new_node = Node::new(new_node);
        if new_node.get_value() < self.nodes[parent_index].get_value() {
            self.nodes[parent_index + 1] = new_node;
        } else {
            self.nodes[parent_index + 2] = new_node;
        }
    }

    // fn remove_node(&mut self, node: Node<f32>) {
    //     unimplemented!();
    // }

    pub fn find_parent(&self, new_value: &T) -> usize {
        let mut i = 0;
        loop {
            match self.nodes[i] {
                Node::Branch { ref value } => {
                    if new_value < value {
                        match self.nodes[i + 1] {
                            Node::None => break,
                            _ => i += 1,

                        }
                    } else {
                        match self.nodes[i + 2] {
                            Node::None => break,
                            _ => i += 2,
                        }
                    }
                }
                Node::None => panic!("Node::None is not a valid parent"),
            }
        }
        i
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let mut tree = FlatTree::new(8, 1.0);

        tree.add_node(3.0);
        tree.add_node(4.0);
        tree.add_node(2.0);

        tree.add_node(5.0);
        println!("{:?}", tree);
    }
}
