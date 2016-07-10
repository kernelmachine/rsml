
#[cfg(test)]
mod tests {
    use tfidf_helper::*;
    use tfidf::{TfIdf, TfIdfDefault};

    #[test]
    fn it_works() {
        let strs = vec!["the quick brown fox", "the black brown fox"];
        let unique_word_list = get_unique_word_list(&strs[..]);

        let docs: Vec<_> = strs.iter().map(|s| str_to_doc(s)).collect();

        let all_docs = docs.clone();

        for doc in docs.into_iter() {
            for word in &unique_word_list {
                let x = TfIdfDefault::tfidf(word, &doc, all_docs.iter());
                println!("{} - {:?}", word, x);
            }
        }
    }
}
