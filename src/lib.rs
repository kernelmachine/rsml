#![feature(custom_derive)]
#![feature(test)]
#[macro_use(s)]

extern crate ndarray;
extern crate typenum;
extern crate rand;
extern crate ndarray_rand;
extern crate test;

pub mod tree;
pub mod traits;
pub mod random_forest;
pub mod util;
