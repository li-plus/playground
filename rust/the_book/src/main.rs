fn basic_variables() {
    println!("===== variables =====");

    let x = 5; // const int = 5;
    println!("The value of immutable x is: {x}");

    let mut x = 5; // int x = 5; (shadow)
    println!("The value of mutable x is: {x}");
    x = 6;
    println!("The value of mutable x is: {x}");
    {
        let x = 20;
        println!("The value of inner x is: {x}");
    }
    println!("The value of outer x is: {x}");

    const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3; // constexpr in c++
    println!("THREE_HOURS_IN_SECONDS: {THREE_HOURS_IN_SECONDS}");

    // integer
    let dec_x = 1_000_000; // i32 by default
    let hex_x = 0xff;
    let oct_x = 0o77;
    let bin_x = 0b1111_0000;
    let byte_x: u8 = b'A';
    println!("dec_x: {dec_x}, hex_x: {hex_x}, oct_x: {oct_x}, bin_x: {bin_x}, byte_x: {byte_x}");

    // floating point
    let x_f64 = 2.0; // f64 by default
    let x_f32: f32 = 3.0; // f32
    println!("x_f64: {x_f64}, x_f32: {x_f32}");

    // boolean type
    let t = true; // bool
    let f: bool = false;
    println!("t: {t}, f: {f}");

    // char type (unicode)
    let c = 'z';
    let z: char = '😻'; // unicode
    let heart_eyed_cat = '😻';
    println!("c: {c}, z: {z}, heart_eyed_cat: {heart_eyed_cat}");

    // string literal
    let s: &str = "hello world";
    println!("s = {s}");

    // tuple
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    let (x, y, z) = tup; // destructure a tuple

    // access a tuple element
    let tup_0 = tup.0;
    let tup_1 = tup.1;
    let tup_2 = tup.2;
    println!("tup: {tup:?}, access: ({tup_0}, {tup_1}, {tup_2}), destructure: ({x}, {y}, {z})");

    // array
    let a: [i32; 5] = [1, 2, 3, 4, 5];
    println!("a: {a:?}",);
    let a0 = a[0]; // index
    println!("a[0]: {a0}");
    let a = [3; 5]; // [3] * 5
    println!("a: {a:?}");
}

fn basic_functions() {
    println!("===== functions =====");

    fn add(x: i32, y: i32) -> i32 {
        x + y
    }

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

fn basic_ownership() {
    println!("===== ownership =====");

    fn calculate_length(s: &String) -> usize {
        // s is a reference to a String
        s.len()
        // s is not dropped when it goes out of scope
    }

    fn update_string(s: &mut String) {
        s.push_str(" world");
    }

    // https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html#dangling-references
    // fn dangle() -> &String { // dangle returns a reference to a String
    //     let s = String::from("hello"); // s is a new String
    //     &s // we return a reference to the String, s
    // } // Here, s goes out of scope, and is dropped. Its memory goes away.
    //   // Danger!

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
    // let _r2 = &s;
    // println!("{_r1} {_r2}");
}

fn basic_slice() {
    // https://doc.rust-lang.org/book/ch04-03-slices.html

    println!("===== slice =====");

    fn first_word(s: &str) -> &str {
        for (i, &item) in s.as_bytes().iter().enumerate() {
            if item == b' ' {
                return &s[..i];
            }
        }
        &s[..]
    }

    let s = String::from("hello world");
    let w = first_word(&s);
    // s.clear();   // error
    println!("first_word(\"{s}\") = \"{w}\"");
}

fn basic_struct() {
    println!("===== struct =====");

    #[derive(Debug)]
    struct User {
        active: bool,
        username: String,
        email: String,
        sign_in_count: u64,
    }

    let username = String::from("user1");
    let user1 = User {
        active: true,
        username, // field init shorthand
        email: String::from("user1@example.com"),
        sign_in_count: 1,
    };
    println!(
        "user1: active={} username={} email={} sign_in_count={}",
        user1.active, user1.username, user1.email, user1.sign_in_count
    );

    // Struct Update Syntax
    let user2 = User {
        email: String::from("user2@example.com"),
        ..user1
    };
    println!("user2 debug print: {user2:?}"); // Debug trait
    println!("user2 pretty debug print: {user2:#?}"); // expanded struct
    dbg!(&user2); // debug macro

    // Cannot use user1 or user1.username since it's been moved to user2
    // println!("user1: {user1:?}");   // error
    // println!("user1.username: {}", user1.username);  // error

    // However user1.email is not moved and still valid
    println!("user1.email: {}", user1.email);

    struct Color(i32, i32, i32);
    let color = Color(255, 0, 0); // RGB color
    let Color(x, y, z) = color; // destructure
    println!("Color: ({x}, {y}, {z})");

    // methods
}

fn basic_method() {
    println!("===== methods =====");

    #[derive(Debug)]
    struct Rectangle {
        width: u32,
        height: u32,
    }

    impl Rectangle {
        fn area(&self) -> u32 {
            self.width * self.height
        }

        fn square(size: u32) -> Self {
            Self {
                width: size,
                height: size,
            }
        }
    }

    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };
    println!(
        "The area of the rectangle {rect1:?} is {} square pixels.",
        rect1.area()
    );

    let sq = Rectangle::square(3);
    println!("Rectangle square {sq:?}");
}

fn basic_enum() {
    println!("===== enum =====");

    enum IpAddr {
        V4(u8, u8, u8, u8),
        V6(String),
    }

    let ipv4 = IpAddr::V4(127, 0, 0, 1);
    let ipv6 = IpAddr::V6(String::from("::1"));

    fn print_ip(ip: &IpAddr) {
        // match statement
        match ip {
            IpAddr::V4(a, b, c, d) => println!("IPv4: {a}.{b}.{c}.{d}"),
            IpAddr::V6(s) => println!("IPv6: {s}"),
        }
    }
    print_ip(&ipv4);
    print_ip(&ipv6);

    fn plus_one(x: Option<i32>) -> Option<i32> {
        match x {
            None => None,
            Some(i) => Some(i + 1),
        }
    }
    assert_eq!(plus_one(Some(5)), Some(6));
    assert_eq!(plus_one(None), None);

    let dice_roll = 6;
    match dice_roll {
        1 => println!("You rolled a one!"),
        6 => println!("You rolled a six!"),
        x => println!("You rolled a {x}!"),
    }
    match dice_roll {
        1 => println!("You rolled a one!"),
        2 => println!("You rolled a two!"),
        _ => (), // do nothing
    }

    let value = Some(5);
    if let Some(x) = value {
        println!("The value is: {x}");
    } else {
        println!("The value is: None")
    }

    fn minus_one(x: Option<i32>) -> Option<i32> {
        // let else syntax
        let Some(i) = x else {
            return None; // must return here
        };
        Some(i - 1)
    }
    assert_eq!(minus_one(Some(5)), Some(4));
    assert_eq!(minus_one(None), None);
}

fn basic_collections() {
    // https://doc.rust-lang.org/book/ch08-00-common-collections.html
    println!("===== collections =====");

    let v: Vec<i32> = Vec::new();
    assert!(v.is_empty());
    let mut v = vec![1, 2, 3];
    println!("{v:?}");
    v.push(4);
    v.push(5);
    println!("{v:?}");

    // indexing
    let v1 = &v[1];
    // let v100 = &v[100];     // panic
    let v100 = v.get(100);
    println!("&v[1] = {v1}");
    println!("v.get(100) = {v100:?}");

    for i in &mut v {
        *i += 1;
    }

    for i in &v {
        println!("{i}")
    }

    // strings
    let s1 = String::from("tic");
    let s2 = String::from("tac");
    let s3 = String::from("toe");
    let s_add = s1 + "-" + &s2 + "-" + &s3; // s1 is moved

    let s1 = String::from("tic");
    let s_fmt = format!("{s1}-{s2}-{s3}");
    print!("s_add = {s_add}, s_fmt = {s_fmt}\n");

    let x = String::from("hello 世界");
    println!("&x[0..6] = {}", &x[0..6]);
    // println!("{}", &x[0..7]);   // panic
    println!("&x[0..9] = {}", &x[0..9]);
    for c in x.chars() {
        print!("{c}");
    }
    println!();
    for c in x.bytes() {
        print!("{c} ");
    }
    println!();

    // hash map
    use std::collections::HashMap;
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);

    let team_name = String::from("Blue");
    let score = scores.get(&team_name).copied().unwrap_or(0);
    assert!(score == 10);

    for (key, value) in &scores {
        println!("{key}: {value}");
    }

    scores.insert(String::from("Blue"), 25); // update
    println!("Updated scores: {scores:?}");

    scores.entry(String::from("Blue")).or_insert(10); // nothing happens
    scores.entry(String::from("Green")).or_insert(10); // Green inserted with value 10
    println!("Scores after entry: {scores:?}");
}

fn basic_error_handling() {
    println!("===== error handling =====");

    // {
    //     panic!("This is a panic message"); // panic! macro
    // }

    // {
    //     let v = vec![1, 2, 3];
    //     v[99]; // panic
    // }

    fn read_file() -> Result<String, std::io::Error> {
        use std::fs::File;
        use std::io::Read;

        let mut f = match File::open("not_found.rs") {
            Ok(file) => file,
            Err(e) => return Err(e),
        };
        let mut content = String::new();
        f.read_to_string(&mut content)?; // same as above but shorter
        Ok(content)
    }
    let result = read_file();
    println!("read_file: {result:?}");
}

fn basic_generic() {
    println!("===== generic =====");

    struct Point<T> {
        x: T,
        y: T,
    }

    impl<T> Point<T> {
        fn x(&self) -> &T {
            &self.x
        }

        fn y(&self) -> &T {
            &self.y
        }
    }

    impl Point<f32> {
        fn distance(&self) -> f32 {
            (self.x * self.x + self.y * self.y).sqrt()
        }
    }

    let p = Point { x: 3.0, y: 4.0 };
    assert!(*p.x() == 3.0);
    assert!(*p.y() == 4.0);
    assert!(p.distance() == 5.0);
}

fn basic_trait() {
    println!("===== trait =====");

    // trait is like interface in other languages
    trait Summary {
        // fn summarize(&self) -> String;

        // default implementation
        fn summarize(&self) -> String {
            String::from("(Read more...)")
        }
    }

    #[allow(unused)]
    struct NewsArticle {
        headline: String,
        location: String,
        author: String,
        content: String,
    }

    impl Summary for NewsArticle {
        fn summarize(&self) -> String {
            format!("{}, by {} ({})", self.headline, self.author, self.location)
        }
    }

    #[allow(unused)]
    struct SocialPost {
        username: String,
        content: String,
        reply: bool,
        repost: bool,
    }

    impl Summary for SocialPost {
        fn summarize(&self) -> String {
            format!("{}: {}", self.username, self.content)
        }
    }

    let post = SocialPost {
        username: String::from("horse_ebooks"),
        content: String::from("of course, as you probably already know, people"),
        reply: false,
        repost: false,
    };
    println!("1 new post: {}", post.summarize());

    let article = NewsArticle {
        headline: String::from("Penguins win the Stanley Cup Championship!"),
        location: String::from("Pittsburgh, PA, USA"),
        author: String::from("Iceburgh"),
        content: String::from(
            "The Pittsburgh Penguins once again are the best \
             hockey team in the NHL.",
        ),
    };

    println!("New article available! {}", article.summarize());

    fn notify1(item: &impl Summary) {
        println!("Breaking news! {}", item.summarize());
    }
    notify1(&article);

    // bound syntax
    fn notify2<T: Summary>(item: &T) {
        println!("Breaking news! {}", item.summarize());
    }
    notify2(&article);

    // multiple trait bounds
    impl ToString for NewsArticle {
        fn to_string(&self) -> String {
            format!(
                "NewsArticle(headline={:?}, author={:?}, location={:?}, content={:?})",
                self.headline, self.author, self.location, self.content
            )
        }
    }
    fn notify3<T: Summary + ToString>(item: &T) {
        println!("TLDR: {}. Object: {}", item.summarize(), item.to_string());
    }
    notify3(&article);
}

fn basic_lifetime() {
    println!("===== lifetime =====");

    // https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#lifetime-annotations-in-function-signatures
    fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
        if x.len() > y.len() {
            x
        } else {
            y
        }
    }

    let string1 = String::from("abcd");
    let result;
    {
        let string2 = String::from("xyz");
        result = longest(&string1, &string2);
        println!("The longest string is {}", result); // ok
    }
    // println!("The longest string is {}", result); // error[E0597]: `string2` does not live long enough

    // life time elision rules: https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#lifetime-elision

    let s: &'static str = "I have a static lifetime."; // static lifetime
    println!("{s}");
}

fn main() {
    basic_variables();
    basic_functions();
    basic_control_flow();
    basic_ownership();
    basic_slice();
    basic_struct();
    basic_method();
    basic_enum();
    basic_collections();
    basic_error_handling();
    basic_generic();
    basic_trait();
    basic_lifetime();
}
