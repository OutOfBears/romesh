use crate::RobloxMesh;
use std::{fs::File, io::Read};

#[test]
fn it_works() {
    let mut file = File::open("test.mesh").unwrap();
    let mut buffer = Vec::new();

    file.read_to_end(&mut buffer).unwrap();

    let mesh = RobloxMesh::from_buffer(buffer).expect("Failed to parse mesh");
    println!("{:?}", mesh);
}
