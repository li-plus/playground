fn add(x: i32, y: i32) -> i32 {
    x + y
}

fn basic_functions() {
    println!("===== functions =====");

    // https://doc.rust-lang.org/book/ch03-03-how-functions-work.html
    // Expressions do not include ending semicolons. If you add a semicolon to the end of an expression, you turn it into a statement, and it will then not return a value.
    let y = {
        let x = 3;
        x + 1
    };
    println!("y: {y}");

    println!("add(1, 2): {}", add(1, 2));
}

fn basic_control_flow() {
    println!("===== control flow =====");

    let number = 6;

    if number % 4 == 0 {
        println!("number is divisible by 4");
    } else if number % 3 == 0 {
        println!("number is divisible by 3");
    } else if number % 2 == 0 {
        println!("number is divisible by 2");
    } else {
        println!("number is not divisible by 4, 3, or 2");
    }

    let x = -3;
    let abs_x = if x < 0 { -x } else { x };
    println!("abs(-3): {abs_x}");

    // equivalent to while (true)
    let mut counter = 0;

    let result = loop {
        counter += 1;

        if counter == 10 {
            break counter * 2;
        }
    };

    println!("The result is {result}");

    // loop labels
    let result = 'outer_loop: loop {
        loop {
            loop {
                break 'outer_loop 123;
            }
        }
    };
    println!("result: {result}");

    // while loop
    let mut cnt = 0;
    while cnt < 3 {
        println!("while loop count: {cnt}");
        cnt += 1;
    }

    // for loop
    let a = [10, 20, 30];
    for elem in a {
        println!("for loop elem: {elem}");
    }

    // reverse
    for number in (1..4).rev() {
        println!("reverse: {number}");
    }
}

fn calculate_length(s: &String) -> usize {
    // s is a reference to a String
    s.len()
    // s is not dropped when it goes out of scope
}

fn update_string(s: &mut String) {
    s.push_str(" world");
}

// fn dangle() -> &String { // dangle returns a reference to a String
//     let s = String::from("hello"); // s is a new String
//     &s // we return a reference to the String, s
// } // Here, s goes out of scope, and is dropped. Its memory goes away.
//   // Danger!

fn basic_ownership() {
    println!("===== ownership =====");

    // like a unique_ptr and assignment means being moved
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved into s2, and is invalidated, cannot use s1 any more
    println!("s1 is invalidated, s2 = {s2}");

    // to make a deep copy, should explicitly call clone()
    let s1 = String::from("hello");
    let s2 = s1.clone();
    println!("s1 = {s1}, s2 = {s2}");

    // pass by reference to preserve ownership
    // We call the action of creating a reference borrowing
    let len = calculate_length(&s1);
    println!("len('{s1}') = {len}");

    // mutable reference
    let mut s = String::from("hello");
    update_string(&mut s);
    println!("updated s = '{s1}'");

    let mut s = String::from("hello");
    let _r1 = &mut s;
    // Mutable references have one big restriction: if you have a mutable reference to a value, you can have no other references to that value.
    // let r2 = &mut s;
    // println!("{}, {}", r1, r2);
}

fn first_word(s: &str) -> &str {
    for (i, &item) in s.as_bytes().iter().enumerate() {
        if item == b' ' {
            return &s[..i];
        }
    }
    &s[..]
}

fn basic_slice() {
    // https://doc.rust-lang.org/book/ch04-03-slices.html
    println!("===== slice =====");
    let s = "hello world";
    let w = first_word(&s);
    // s.clear();   // error
    println!("first_word(\"{s}\") = \"{w}\"");
}

struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64
}

fn basic_struct() {
    let user1 = User {
        active: true,
        username: String::from("user1"),
        email: String::from("user1@example.com"),
        sign_in_count: 1
    };
    println!("user1: name={} email={} active={} sign_in_count={}", user1.username, user1.email, user1.active, user1.sign_in_count);

    // field init shorthand
    let username = String::from("user2");
    let email = String::from("user2@example.com");
    let user2 = User {
        active: true,
        username, 
        email, 
        sign_in_count: 1
    };
    println!("user2: name={} email={} active={} sign_in_count={}", user2.username, user2.email, user2.active, user2.sign_in_count);

    // Struct Update Syntax
    let user3 = User {
        username: String::from("user3"),
        email: String::from("user3@example.com"),
        ..user2
    };
    println!("user3: name={} email={} active={} sign_in_count={}", user3.username, user3.email, user3.active, user3.sign_in_count);
}

fn main() {
    basic_functions();
    basic_control_flow();
    basic_ownership();
    basic_slice();
    basic_struct();
}
