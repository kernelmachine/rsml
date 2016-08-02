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

fn should_replace(c: u8) -> bool {
    match c {
        b'.' => true,
        b'?' => true,
        b'!' => true,
        b',' => true,
        b':' => true,
        b';' => true,
        b'(' => true,
        b')' => true,
        b'{' => true,
        b'}' => true,
        b']' => true,
        b'[' => true,
        b'/' => true,
        b'=' => true,
        b'|' => true,
        b'~' => true,
        _ => false,
    }
}

fn should_drop(c: u8) -> bool {
    match c {
        b'\'' => true,
        b'\"' => true,
        b'`' => true,
        b'-' => true,
        b'_' => true,
        b'*' => true,
        b'&' => true,
        _ => false,
    }
}

pub fn get_words(sentence: &str) -> Vec<String> {
    // This whole function could easily be optimized by turning the sentence into a Vec<u8>.
    // We can, fo rour purposes, simply strip out all non-ascii characters, and then do in-place
    // replacements. This would incur only a single copy for the function..
    let cleaned = sentence;

    let cleaned: String = cleaned.chars()
                                 .filter(|c| c.is_ascii())
                                 .collect();

    let mut cleaned: Vec<u8> = cleaned.bytes().collect();

    for c in cleaned.as_mut_slice() {
        if should_replace(*c) {
            *c = b' ';
        }
    }

    let cleaned: Vec<_> = cleaned.into_iter()
                                 .filter(|c| !should_drop(*c))
                                 .collect();

    // We take in a &str and filter all non-ascii out, this is safe
    let cleaned = unsafe { String::from_utf8_unchecked(cleaned) };
    let cleaned = cleaned.to_lowercase();


    cleaned.split_whitespace()
           .filter(|s| 2 < s.len() && s.len() < 10)
           .filter(|s| s.chars().all(is_letter))
           .map(depluralize)
           .filter(|s| 2 < s.len())
           .map(String::from)
           .collect()
}

fn depluralize(s: &str) -> &str {
    if s.chars().last().unwrap() == 's' {
        &s[..s.len() - 1]
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
        assert_eq!("dres", depluralize("dress"));
    }
}
