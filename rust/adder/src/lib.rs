/// Adds two numbers.
///
/// # Examples
///
/// ```
/// let answer = adder::add(1, 2);
/// assert_eq!(3, answer);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

pub fn do_panic() {
    panic!("This function always panics");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    #[ignore]
    fn it_fails() {
        let result = add(2, 2);
        // assert with custom error message
        assert_eq!(
            result, 5,
            "Expected 2 + 2 to equal 5, but it equals {}",
            result
        );
    }

    #[test]
    #[should_panic(expected = "always panics")] // substring of the message
    fn it_panics() {
        do_panic();
    }
}
