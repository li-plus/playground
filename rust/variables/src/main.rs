fn main() {
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
