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

fn basic_closure() {
    println!("===== closure =====");

    let mut list = vec![1, 2, 3];

    // immutable
    let immutable_closure = || println!("immutable closure: {list:?}");
    immutable_closure();

    // mutable
    println!("Before mutable closure: {list:?}");
    let mut mutable_closure = || list.push(4);
    // println!("Before mutable closure: {list:?}"); // error[E0502]: cannot borrow `list` as immutable because it is also borrowed as mutable
    mutable_closure();
    println!("After mutable closure: {list:?}");
    // mutable_closure(); // error[E0502]: cannot borrow `list` as immutable because it is also borrowed as mutable

    // move
    use std::thread;
    thread::spawn(move || println!("Thread spawned with list: {list:?}"))
        .join()
        .unwrap();
}

fn basic_iterator() {
    println!("===== iterator =====");

    let v1 = vec![1, 2, 3];

    // iter
    let mut v1_iter = v1.iter();

    assert_eq!(v1_iter.next(), Some(&1));
    assert_eq!(v1_iter.next(), Some(&2));
    assert_eq!(v1_iter.next(), Some(&3));
    assert_eq!(v1_iter.next(), None);
    assert_eq!(v1_iter.next(), None);

    // iter_mut
    let mut v1 = vec![1, 2, 3];
    for v in v1.iter_mut() {
        *v += 1;
    }
    assert_eq!(v1, vec![2, 3, 4]);

    // into_iter
    let v1 = vec![1, 2, 3];
    let v1_into: Vec<_> = v1.into_iter().collect(); // v1 is moved
    assert_eq!(v1_into, vec![1, 2, 3]);

    // consuming iter
    let v1 = vec![1, 2, 3];
    let v1_iter = v1.iter();
    let total: i32 = v1_iter.sum();
    assert_eq!(total, 6);

    // map
    let v2: Vec<_> = v1.iter().map(|x| x + 1).collect();
    assert_eq!(v2, vec![2, 3, 4]);

    // filter
    let v2: Vec<_> = v1.into_iter().filter(|x| x % 2 == 1).collect();
    assert_eq!(v2, vec![1, 3]);
}

fn basic_smart_pointers() {
    println!("===== smart pointers =====");

    let x = 5;
    let y = Box::new(x); // smart pointer with data on heap
    assert_eq!(x, *y); // deref

    struct MyBox<T>(T);

    impl<T> MyBox<T> {
        fn new(x: T) -> MyBox<T> {
            MyBox(x)
        }
    }

    use std::ops::Deref;

    impl<T> Deref for MyBox<T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            return &self.0;
        }
    }

    let y = MyBox::new(5);
    assert_eq!(x, *y); // equivalent to *(y.deref())

    // deref coercion
    fn hello(name: &str) {
        println!("Hello, {name}!");
    }
    let m = MyBox::new(String::from("Rust"));
    hello(&m); // &MyBox<String> -> &String -> str

    // Rc: shared ptr with reference counter
    use std::rc::Rc;

    #[derive(Debug)]
    #[allow(unused)]
    enum List {
        Cons(i32, Rc<List>),
        Nil,
    }
    use List::{Cons, Nil};

    let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
    println!("a = {a:?}");
    println!("count after creating a = {}", Rc::strong_count(&a));
    let _b = Cons(3, Rc::clone(&a));
    println!("count after creating b = {}", Rc::strong_count(&a));
    {
        let _c = Cons(4, Rc::clone(&a));
        println!("count after creating c = {}", Rc::strong_count(&a));
    }
    println!("count after c goes out of scope = {}", Rc::strong_count(&a));

    // RefCell<T>: borrow rule at runtime
    use std::cell::RefCell;
    let x = Rc::new(RefCell::new(5));
    let a = Rc::clone(&x);
    let b = Rc::clone(&x);
    *a.borrow_mut() += 1;
    *b.borrow_mut() += 1;
    println!("x = {}", x.borrow());

    // Weak<T>: avoid reference cycles
    use std::rc::Weak;

    #[derive(Debug)]
    #[allow(unused)]
    struct Node {
        value: i32,
        parent: RefCell<Weak<Node>>,
        children: Vec<Rc<Node>>,
    }

    let leaf = Rc::new(Node {
        value: 5,
        parent: RefCell::new(Weak::new()),
        children: vec![],
    });
    {
        let branch = Rc::new(Node {
            value: 10,
            parent: RefCell::new(Weak::new()),
            children: vec![Rc::clone(&leaf)],
        });
        *leaf.parent.borrow_mut() = Rc::downgrade(&branch);

        println!("leaf = {leaf:?}");
        println!("branch = {branch:?}");
        println!("leaf parent = {:?}", leaf.parent.borrow().upgrade());
        println!(
            "leaf strong = {}, weak = {}",
            Rc::strong_count(&leaf),
            Rc::weak_count(&leaf)
        );
        println!(
            "branch strong = {}, weak = {}",
            Rc::strong_count(&branch),
            Rc::weak_count(&branch)
        );
    }
    println!(
        "after branch dropped: leaf strong = {}, weak = {}",
        Rc::strong_count(&leaf),
        Rc::weak_count(&leaf)
    );
}

fn basic_concurrency() {
    println!("===== concurrency =====");

    // channel
    use std::sync::mpsc;
    use std::thread;
    let (tx, rx) = mpsc::channel();

    let tx1 = tx.clone();

    thread::spawn(move || {
        let val = String::from("hello");
        tx.send(val).unwrap();
    });
    thread::spawn(move || {
        let val = String::from("world");
        tx1.send(val).unwrap();
    });

    for received in rx {
        println!("Got: {received}");
    }

    // mutex
    use std::sync::{Arc, Mutex};
    let m = Mutex::new(5);
    {
        let mut num = m.lock().unwrap();
        *num = 6;
    } // MutexGuard dropped, unlock automatically

    println!("m = {m:?}");

    // Atomic Reference Counting with Arc<T>
    let counter = Arc::new(Mutex::new(0));
    let mut handles = Vec::with_capacity(10);
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut x;
            {
                let num = counter.lock().unwrap();
                x = num;
            }
            *x += 1;
        });
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap();
    }
    println!("Result: {}", counter.lock().unwrap());
}

fn basic_async() {
    println!("===== async =====");

    // trpl: The Rust Programming Language
    use std::thread;
    use std::time::Duration;
    use trpl::Html;

    async fn page_title(url: &str) -> Option<String> {
        let response = trpl::get(url).await;
        let response_text = response.text().await;
        Html::parse(&response_text)
            .select_first("title")
            .map(|title_element| title_element.inner_html())
    }

    trpl::run(async {
        let url = "https://www.rust-lang.org";
        match page_title(url).await {
            Some(title) => println!("The title for {url} was {title}"),
            None => println!("{url} had no title"),
        }
    });

    trpl::run(async {
        let fut1 = async {
            for i in 0..5 {
                println!("hi number {i} from the first task!");
                trpl::sleep(Duration::from_millis(10)).await;
            }
        };
        let fut2 = async {
            for i in 0..3 {
                println!("hi number {i} from the second task!");
                trpl::sleep(Duration::from_millis(10)).await;
            }
        };
        // similar to asyncio.create_task
        let fut3 = trpl::spawn_task(async {
            for i in 0..3 {
                println!("hi number {i} from the third task!");
                trpl::sleep(Duration::from_millis(10)).await;
            }
        });
        trpl::sleep(Duration::from_millis(20)).await;
        // at this point, fut1 & fut2 are not running, fut3 is running
        // if not joining the future, task3 will be terminated when this main async block is done
        trpl::join(fut1, fut2).await;
        fut3.await.unwrap();
    });

    trpl::run(async {
        // similar to asyncio.Queue
        let (tx, mut rx) = trpl::channel();
        let tx1 = tx.clone();

        let tx_fut = async move {
            let vals = vec![
                String::from("hi"),
                String::from("from"),
                String::from("the"),
                String::from("future"),
            ];
            for val in vals {
                tx.send(val).unwrap();
                trpl::sleep(Duration::from_millis(10)).await;
            }
        };
        let tx1_fut = async move {
            let vals = vec![
                String::from("more"),
                String::from("messages"),
                String::from("for"),
                String::from("you"),
            ];
            for val in vals {
                tx1.send(val).unwrap();
                trpl::sleep(Duration::from_millis(10)).await;
            }
        };
        let rx_fut = async {
            // when tx and tx1 is dropped, rx.recv() returns None
            while let Some(val) = rx.recv().await {
                println!("received: '{val}'");
            }
        };
        trpl::join!(tx_fut, tx1_fut, rx_fut);
    });

    // yield control to executor
    trpl::run(async {
        let slow = async {
            println!("'slow' started.");
            trpl::sleep(Duration::from_millis(10)).await;
            println!("'slow' finished."); // won't print
        };

        let fast = async {
            println!("'fast' started.");
            trpl::sleep(Duration::from_millis(5)).await;
            println!("'fast' finished.");
        };

        trpl::race(slow, fast).await;
    });

    trpl::run(async {
        let one_ns = Duration::from_nanos(1);
        use std::time::Instant;

        let start = Instant::now();
        async {
            for _ in 1..1000 {
                trpl::sleep(one_ns).await;
            }
        }
        .await;
        let time = Instant::now() - start;
        println!(
            "'sleep' version finished after {} seconds.",
            time.as_secs_f32()
        );

        let start = Instant::now();
        async {
            for _ in 1..1000 {
                trpl::yield_now().await;
            }
        }
        .await;
        let time = Instant::now() - start;
        println!(
            "'yield' version finished after {} seconds.",
            time.as_secs_f32()
        );
    });

    trpl::run(async {
        use trpl::StreamExt;

        let values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let iter = values.iter().map(|n| n * 2);
        let mut stream = trpl::stream_from_iter(iter);

        // stream is like async iterator
        while let Some(value) = stream.next().await {
            println!("The value was: {value}");
        }
    });

    trpl::run(async {
        use std::pin::pin;
        use trpl::{ReceiverStream, Stream, StreamExt};

        fn get_messages() -> impl Stream<Item = String> {
            let (tx, rx) = trpl::channel();

            trpl::spawn_task(async move {
                let messages = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
                for (index, message) in messages.into_iter().enumerate() {
                    let time_to_sleep = if index % 2 == 0 { 10 } else { 30 };
                    trpl::sleep(Duration::from_millis(time_to_sleep)).await;
                    tx.send(format!("Message: '{message}'")).unwrap();
                }
            });

            ReceiverStream::new(rx)
        }

        let mut messages = pin!(get_messages().timeout(Duration::from_millis(20)));

        while let Some(result) = messages.next().await {
            match result {
                Ok(message) => println!("{message}"),
                Err(reason) => eprintln!("Problem: {reason:?}"),
            }
        }
    });

    {
        // thread and as
        let (tx, mut rx) = trpl::channel();

        thread::spawn(move || {
            for i in 1..11 {
                tx.send(i).unwrap();
                thread::sleep(Duration::from_millis(10));
            }
        });

        trpl::run(async {
            while let Some(message) = rx.recv().await {
                println!("{message}");
            }
        });
    }
}

fn basic_patterns() {
    println!("===== patterns =====");

    struct Point {
        x: i32,
        y: i32,
    }

    let p = Point { x: 0, y: 7 };

    // Destructuring Structs
    let Point { x: a, y: b } = p;
    assert_eq!(0, a);
    assert_eq!(7, b);

    // Destructuring Structs
    let Point { x, y } = p;
    assert_eq!(0, x);
    assert_eq!(7, y);

    let p = Point { x: 0, y: 7 };

    match p {
        Point { x, y: 0 } => println!("On the x axis at {x}"),
        Point { x: 0, y } => println!("On the y axis at {y}"),
        Point { x, y } => {
            println!("On neither axis: ({x}, {y})");
        }
    }

    #[allow(dead_code)]
    enum Message {
        Quit,
        Move { x: i32, y: i32 },
        Write(String),
        ChangeColor(i32, i32, i32),
    }

    let msg = Message::ChangeColor(0, 160, 255);

    match msg {
        Message::Quit => {
            println!("The Quit variant has no data to destructure.");
        }
        Message::Move { x, y } => {
            println!("Move in the x direction {x} and in the y direction {y}");
        }
        Message::Write(text) => {
            println!("Text message: {text}");
        }
        Message::ChangeColor(r, g, b) => {
            println!("Change color to red {r}, green {g}, and blue {b}");
        }
    }

    {
        // Remaining Parts of a Value with ..
        #[allow(dead_code)]
        struct Point {
            x: i32,
            y: i32,
            z: i32,
        }

        let origin = Point { x: 0, y: 0, z: 0 };

        match origin {
            Point { x, .. } => println!("x is {x}"),
        }

        let numbers = (2, 4, 8, 16, 32);

        match numbers {
            (first, .., last) => {
                println!("Some numbers: {first}, {last}");
            }
        }
    }

    // Extra Conditionals with Match Guards
    let num = Some(4);

    match num {
        Some(x) if x % 2 == 0 => println!("The number {x} is even"),
        Some(x) => println!("The number {x} is odd"),
        None => (),
    }

    // @ Bindings
    {
        enum Message {
            Hello { id: i32 },
        }

        let msg = Message::Hello { id: 5 };

        match msg {
            Message::Hello {
                id: id_variable @ 3..=7,
            } => println!("Found an id in range: {id_variable}"),
            Message::Hello { id: 10..=12 } => {
                println!("Found an id in another range")
            }
            Message::Hello { id } => println!("Found some other id: {id}"),
        }
    }
}

fn basic_unsafe() {
    println!("===== unsafe =====");

    // 1. Dereferencing a Raw Pointer
    let mut num = 5;

    let r1 = &raw const num;
    let r2 = &raw mut num;

    unsafe {
        println!("r1 is: {}", *r1);
        println!("r2 is: {}", *r2);
    }

    // 2. Calling an Unsafe Function or Method
    unsafe fn dangerous() {}

    unsafe {
        dangerous();
    }

    unsafe extern "C" {
        safe fn abs(input: i32) -> i32;
    }

    println!("Absolute value of -3 according to C: {}", abs(-3));

    // 3. Accessing or Modifying a Mutable Static Variable
    static mut COUNTER: u32 = 0;

    /// SAFETY: Calling this from more than a single thread at a time is undefined
    /// behavior, so you *must* guarantee you only call it from a single thread at
    /// a time.
    unsafe fn add_to_count(inc: u32) {
        unsafe {
            COUNTER += inc;
        }
    }

    unsafe {
        // SAFETY: This is only called from a single thread in `main`.
        add_to_count(3);
        println!("COUNTER: {}", *(&raw const COUNTER));
    }

    // 4. Implementing an Unsafe Trait
    #[allow(dead_code)]
    unsafe trait Foo {
        // methods go here
    }

    unsafe impl Foo for i32 {
        // method implementations go here
    }

    // 5. Accessing Fields of a Union
}

fn basic_function_pointer_and_closure() {
    println!("===== function pointer =====");

    fn add_one(x: i32) -> i32 {
        x + 1
    }

    fn do_twice(f: fn(i32) -> i32, arg: i32) -> i32 {
        f(arg) + f(arg)
    }

    let answer = do_twice(add_one, 5);
    println!("The answer is: {answer}");

    // return a trait
    {
        fn returns_closure() -> impl Fn(i32) -> i32 {
            |x| x + 1
        }

        let f = returns_closure();
        let answer = f(5);
        println!("The answer from closure is: {answer}");
    }

    // work with multiple functions with the same signature using trait object
    fn returns_closure() -> Box<dyn Fn(i32) -> i32> {
        Box::new(|x| x + 1)
    }

    fn returns_initialized_closure(init: i32) -> Box<dyn Fn(i32) -> i32> {
        Box::new(move |x| x + init)
    }

    let handlers = vec![returns_closure(), returns_initialized_closure(123)];
    for handler in handlers {
        let output = handler(5);
        println!("{output}");
    }
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
    basic_closure();
    basic_iterator();
    basic_smart_pointers();
    basic_concurrency();
    basic_async();
    basic_patterns();
    basic_unsafe();
    basic_function_pointer_and_closure();
}
