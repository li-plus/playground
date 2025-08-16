use adder::add;

#[test]
fn it_adds_two_with_two() {
    let result = add(2, 2);
    assert_eq!(result, 4);
}
