extern crate tfidf;

use std::collections::BTreeMap;

use std::ascii::AsciiExt;


pub fn str_to_doc(string: &str) -> Vec<(String, usize)> {
    let words = get_words(string);
    let mut word_map = BTreeMap::new();

    for word in words {
        *word_map.entry(word).or_insert(0) += 1;
    }
    word_map.into_iter().map(|(word, count)| (word, count)).collect()
}

fn get_words(sentence: &str) -> Vec<String> {
    let cleaned = sentence;
    let cleaned = cleaned.replace(".", " ");
    let cleaned = cleaned.replace("?", " ");
    let cleaned = cleaned.replace("!", " ");
    let cleaned = cleaned.replace(",", " ");
    let cleaned = cleaned.replace(":", " ");
    let cleaned = cleaned.replace(";", " ");
    let cleaned = cleaned.replace("'", "");
    let cleaned = cleaned.replace("\"", "");
    let cleaned = cleaned.replace("(", " ");
    let cleaned = cleaned.replace(")", " ");
    let cleaned = cleaned.replace("`", "");
    let cleaned = cleaned.replace("{", " ");
    let cleaned = cleaned.replace("}", " ");
    let cleaned = cleaned.replace("]", " ");
    let cleaned = cleaned.replace("[", " ");
    let cleaned = cleaned.replace("-", "");
    let cleaned = cleaned.replace("_", "");
    let cleaned = cleaned.replace("&amp", "");
    let cleaned = cleaned.replace("&gt", "");
    let cleaned = cleaned.replace("&lt", "");
    let cleaned = cleaned.replace("*", "");;
    let cleaned = cleaned.replace("/", " ");
    let cleaned = cleaned.replace("=", " ");
    let cleaned = cleaned.replace("|", " ");
    let cleaned = cleaned.replace("~", " ");
    let cleaned = cleaned.to_lowercase();
    let cleaned: String = cleaned.chars()
                                 .filter(|c| c.is_ascii())
                                 .collect();

    cleaned.split_whitespace()
           .filter(|s| 3 < s.len() && s.len() < 7)
           .filter(|s| s.chars().all(is_letter))
           .map(depluralize)
           .filter(|s| 3 < s.len())
           .map(String::from)
           .collect()
}

fn depluralize<'a>(s: &'a str) -> &'a str {
    if s.chars().last().unwrap() == 's' {
        &s[..s.len() - 2]
    } else {
        s
    }
}

fn is_letter(c: char) -> bool {
    match c as u8 {
        97...122 => true,
        _ => false,
    }
}

pub fn get_unique_word_list(sentence_list: &[&str]) -> Vec<String> {
    let mut words: Vec<String> = sentence_list.iter()
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

    #[test]
    fn test_depluralize() {
        assert_eq!("dog", depluralize("dogs"));
        assert_eq!("jersey", depluralize("jersey"));
        assert_eq!("jersey", depluralize("dress"));
    }
}
