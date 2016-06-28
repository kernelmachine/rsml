extern crate tfidf;

use std::collections::BTreeMap;

pub fn str_to_doc(string: &str) -> Vec<(&str, usize)> {
    let split_str: Vec<&str> = string.split_whitespace().collect();
    let mut word_map = BTreeMap::new();

    for word in split_str {
        *word_map.entry(word).or_insert(0) += 1;
    }
    word_map.into_iter().map(|(word, count)| (word, count)).collect()
}

fn get_words(sentence: &str) -> Vec<&str> {
    sentence.split_whitespace().collect()
}

pub fn get_unique_word_list<'a>(sentence_list: &'a [&str]) -> Vec<&'a str> {
    let mut words: Vec<&str> = sentence_list.iter()
                                            .map(|sentence| get_words(sentence))
                                            .fold(vec![], |mut acc, s| {
                                                acc.extend_from_slice(&s[..]);
                                                acc
                                            });

    words.sort();
    words.dedup();
    words
}


#[cfg(test)]
mod tests {
    use super::*;
    use tfidf::{TfIdf, TfIdfDefault};

    #[test]
    fn it_works() {
        let strs = vec!["the quick brown fox", "the only fox fox red fox the", "quick black dog"];
        let unique_word_list = get_unique_word_list(&strs[..]);

        let docs: Vec<_> = strs.iter().map(|s| str_to_doc(s)).collect();

        let all_docs = docs.clone();

        for doc in docs.into_iter() {
            for word in &unique_word_list {
                let x = TfIdfDefault::tfidf(word, &doc, all_docs.iter());
            }
        }
    }
}
