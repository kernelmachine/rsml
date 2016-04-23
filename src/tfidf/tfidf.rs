
use ndarray::{Ix, ArrayView,OwnedArray};
use regex::Regex;

/// Rectangular matrix.
pub type Mat<A> = OwnedArray<A, (Ix, Ix)>;



/// Feature view
pub type Feature<'a,A> = ArrayView<'a,A, Ix>;

/// Sample view
pub type Sample<'a,A> = ArrayView<'a,A, Ix>;


/// Col matrix.
pub type Col<A> = OwnedArray<A, Ix>;



// get the tfidf score from a vector of strings representing documents
pub fn fit(split_document : Vec<String>) -> Mat<f64> {
    let n_samples = split_document.len();
    let df = &document_frequency(split_document);
    let idf = (n_samples as f64 / df).mapv(|x| x.log(10.0)) + 1.0;
    df.dot(&idf.t())
}

pub fn document_frequency(split_document : Vec<String>) -> Mat<f64> {
    let document = split_document.join(" ");
    let mut freqs = OwnedArray::zeros((split_document.len(),1));
    for ix in 0..split_document.len() {
        let iter = split_document[ix].split_whitespace();
        for word in iter.collect::<Vec<&str>>().iter() {
            let re = Regex::new(word).expect("regex error");
            let freq : usize = re.captures_iter(&document).collect::<Vec<_>>().len();
            freqs[(ix,0)] = freq as f64;
        }
    }
    freqs
}
