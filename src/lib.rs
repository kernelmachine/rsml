#![feature(custom_derive, plugin)]
#![feature(test)]
#[macro_use(s)]

extern crate ndarray;
extern crate typenum;
extern crate rand;
extern crate ndarray_rand;
extern crate rustc_serialize;
extern crate test;
extern crate rayon;
extern crate tfidf;
extern crate serde;

pub mod tree;
pub mod traits;
pub mod random_forest;
pub mod util;
pub mod tfidf_helper;
pub mod accuracy;
