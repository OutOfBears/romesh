use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use std::io::{BufRead, Cursor, Read, Seek, SeekFrom};

use regex::Regex;

mod tests;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub type Vertex = [f32; 3];
pub type Color = [u8; 4];
pub type Normal = [f32; 3];
pub type Face = [u32; 3];
pub type UV = [f32; 2];
pub type Tangent = [f32; 4];

#[derive(Debug, Clone)]
pub struct Bone {
    pub name: String,
    pub parent_id: usize,
    pub lod_parent_id: usize,
    pub culling: f32,
    pub cframe: [f32; 12],
}

#[derive(Debug, Clone)]
pub struct RobloxMesh {
    pub version: String,
    pub faces: Vec<Face>,
    pub vertices: Vec<Vertex>,
    pub normals: Vec<Normal>,
    pub uvs: Vec<UV>,
    pub tangents: Vec<Tangent>,
    pub vertex_colors: Vec<Color>,
    pub skin_indices: Vec<[u8; 4]>,
    pub skin_weights: Vec<[f32; 4]>,
    pub bones: Vec<Bone>,
}

#[derive(Debug, Clone)]
pub struct RobloxMeshHeader {
    pub version: String,
    pub header_size: u16,
    pub vertex_size: u8,
    pub face_size: u8,
    pub lod_size: u16,
    pub name_table_size: u32,
    pub facs_data_size: u32,
    pub lod_count: u16,
    pub vertex_count: u32,
    pub face_count: u32,
    pub bone_count: u16,
    pub subset_count: u16,
}

fn read_string(reader: &mut Cursor<Vec<u8>>) -> Result<String> {
    let mut buffer = Vec::new();
    reader.read_until(0x00, &mut buffer)?;

    buffer.pop();
    String::from_utf8(buffer).map_err(|e| e.into())
}

fn read_float(value: &str) -> Result<f32> {
    value.trim().parse::<f32>().map_err(|e| e.into())
}

impl RobloxMesh {
    pub fn default() -> Self {
        Self {
            version: String::new(),
            faces: Vec::new(),
            vertices: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            tangents: Vec::new(),
            vertex_colors: Vec::new(),
            skin_indices: Vec::new(),
            skin_weights: Vec::new(),
            bones: Vec::new(),
        }
    }

    pub fn from_buffer(buffer: Vec<u8>) -> Result<Self> {
        let mut reader = Cursor::new(buffer);

        let mut version = [0u8; 8];
        let mut version_number = [0u8; 4];
        let checked_version = "version ";

        reader.read_exact(&mut version)?;

        assert!(
            &version[0..checked_version.len()] == checked_version.as_bytes(),
            "Invalid mesh version"
        );

        reader.read_exact(&mut version_number)?;

        let version = String::from_utf8(version_number.to_vec())?;
        match version.clone().as_str() {
            "1.00" | "1.01" => Self::from_v1(version, reader),
            _ => Self::from_v2(version, reader),
        }
    }

    fn from_v1(version: String, mut reader: Cursor<Vec<u8>>) -> Result<Self> {
        let default = Self::default();

        let mut header = String::new();
        let mut count = String::new();
        let mut buffer = String::new();

        reader.seek(SeekFrom::Start(0))?;
        reader.read_line(&mut header)?;
        reader.read_line(&mut count)?;
        reader.read_line(&mut buffer)?;

        header = header.trim().to_owned();
        count = count.trim().to_owned();
        buffer = buffer.trim().to_owned();

        assert!(&header[0..8] == "version ", "Invalid mesh version");
        assert!(header[8..] == version, "Invalid mesh version");

        let face_count = count.parse::<u32>()?;

        // split the data into vectors
        let vectors = buffer[1..buffer.len() - 1]
            .split("][")
            .collect::<Vec<&str>>();

        assert!(vectors.len() as u32 == face_count * 9, "Length mismatch");

        let scale_mult = if version == "1.00" { 0.5 } else { 1.0 };

        let mut vertices = vec![];
        let mut normals = vec![];
        let mut uvs = vec![];
        let mut faces = vec![];
        let mut index = 0;

        for face_data in vectors.chunks(9) {
            assert!(face_data.len() == 9, "Invalid face data length");

            for vertex_data in face_data.chunks(3) {
                assert!(vertex_data.len() == 3, "Invalid vertex data length");

                let vertex = vertex_data[0].split(",").collect::<Vec<&str>>();
                let normal = vertex_data[1].split(",").collect::<Vec<&str>>();
                let uv = vertex_data[2].split(",").collect::<Vec<&str>>();

                assert!(vertex.len() == 3, "Invalid vertex length");
                assert!(normal.len() == 3, "Invalid normal length");
                assert!(uv.len() == 3, "Invalid uv length");

                vertices.push([
                    read_float(vertex[0])? * scale_mult,
                    read_float(vertex[1])? * scale_mult,
                    read_float(vertex[2])? * scale_mult,
                ]);

                normals.push([
                    read_float(normal[0])?,
                    read_float(normal[1])?,
                    read_float(normal[2])?,
                ]);

                uvs.push([read_float(uv[0])?, read_float(uv[1])?]);
            }

            faces.push([index, index + 1, index + 2]);
            index += 3;
        }

        Ok(Self {
            version: String::from(version),
            faces,
            vertices,
            normals,
            uvs,
            ..default
        })
    }

    fn from_v2(version: String, mut reader: Cursor<Vec<u8>>) -> Result<Self> {
        let new_line = reader.read_u8()?;

        assert!(
            new_line == 0x0A || new_line == 0x0D && reader.read_u8()? == 0x0A,
            "Invalid mesh version 2 (Missing new line)"
        );

        let file_begin = reader.position();

        let header = Self::read_header(&version, &mut reader)?;
        println!("{:?}", header);

        let file_end = reader.position()
            + (header.vertex_size as u64 * header.vertex_count as u64)
            + (if header.bone_count > 0 {
                header.bone_count as u64 * 16
            } else {
                0
            })
            + (header.face_count as u64 * header.face_size as u64)
            + (header.lod_count as u64 * header.lod_size as u64)
            + (header.bone_count as u64 * 60)
            + header.name_table_size as u64
            + (header.subset_count as u64 * 72)
            + header.facs_data_size as u64;

        assert!(header.vertex_size >= 36, "Invalid vertex size");
        assert!(header.face_size >= 12, "Invalid face size");
        assert!(header.lod_size >= 4, "Invalid lod size");

        assert!(
            file_end <= reader.get_ref().len() as u64,
            "Invalid file size"
        );

        Self::read_mesh(&header, &mut reader, file_begin)
    }

    fn read_header(version: &String, reader: &mut Cursor<Vec<u8>>) -> Result<RobloxMeshHeader> {
        let mut vertex_size = 0;
        let mut face_size = 12;
        let mut lod_size = 4;
        let mut name_table_size = 0;
        let mut facs_data_size = 0;

        let mut lod_count = 0;
        let mut vertex_count = 0;
        let mut face_count = 0;
        let mut bone_count = 0;
        let mut subset_count = 0;

        let header_size = reader.read_u16::<LittleEndian>()?;

        if version.starts_with("2.") {
            assert_eq!(header_size, 12, "Invalid header size");

            vertex_size = reader.read_u8()?;
            face_size = reader.read_u8()?;
            vertex_count = reader.read_u32::<LittleEndian>()?;
            face_count = reader.read_u32::<LittleEndian>()?;
        } else if version.starts_with("3.") {
            assert_eq!(header_size, 16, "Invalid header size");

            vertex_size = reader.read_u8()?;
            face_size = reader.read_u8()?;
            lod_size = reader.read_u16::<LittleEndian>()?;
            lod_count = reader.read_u16::<LittleEndian>()?;
            vertex_count = reader.read_u32::<LittleEndian>()?;
            face_count = reader.read_u32::<LittleEndian>()?;
        } else if version.starts_with("4.") {
            assert_eq!(header_size, 24, "Invalid header size");

            reader.seek(SeekFrom::Current(2))?;
            vertex_count = reader.read_u32::<LittleEndian>()?;
            face_count = reader.read_u32::<LittleEndian>()?;
            lod_count = reader.read_u16::<LittleEndian>()?;
            bone_count = reader.read_u16::<LittleEndian>()?;
            name_table_size = reader.read_u32::<LittleEndian>()?;
            subset_count = reader.read_u16::<LittleEndian>()?;

            reader.seek(SeekFrom::Current(2))?;
            vertex_size = 40;
        } else if version.starts_with("5.") {
            assert!(header_size >= 32, "Invalid header size");

            reader.seek(SeekFrom::Current(2))?;
            vertex_count = reader.read_u32::<LittleEndian>()?;
            face_count = reader.read_u32::<LittleEndian>()?;
            lod_count = reader.read_u16::<LittleEndian>()?;
            bone_count = reader.read_u16::<LittleEndian>()?;
            name_table_size = reader.read_u32::<LittleEndian>()?;
            subset_count = reader.read_u16::<LittleEndian>()?;
            reader.seek(SeekFrom::Current(2))?;
            reader.seek(SeekFrom::Current(4))?;
            facs_data_size = reader.read_u32::<LittleEndian>()?;
        } else {
            return Err("Invalid version".into());
        }

        Ok(RobloxMeshHeader {
            version: version.clone(),
            header_size,
            vertex_size,
            face_size,
            lod_size,
            name_table_size,
            facs_data_size,
            lod_count,
            vertex_count,
            face_count,
            bone_count,
            subset_count,
        })
    }

    fn read_mesh(
        header: &RobloxMeshHeader,
        reader: &mut Cursor<Vec<u8>>,
        start: u64,
    ) -> Result<RobloxMesh> {
        let mut faces: Vec<[u32; 3]> = vec![[0u32; 3]; header.face_count as usize];
        let mut vertices = vec![[0f32; 3]; header.vertex_count as usize];
        let mut uvs = vec![[0f32; 2]; header.vertex_count as usize];
        let mut normals = vec![[0f32; 3]; header.vertex_count as usize];
        let mut tangents = vec![[0f32; 4]; header.vertex_count as usize];
        let mut lods = Vec::new();

        let has_vertex_colors = header.vertex_size >= 40;
        let mut vertex_colors = vec![];

        let mut skin_indices = Vec::new();
        let mut skin_weights = Vec::new();

        let mut bones = Vec::new();

        reader.seek(SeekFrom::Start(start + header.header_size as u64))?;

        for i in 0..header.vertex_count {
            let mut vertex = [0f32; 3];
            let mut normal = [0f32; 3];
            let mut uv = [0f32; 2];
            let mut tangent = [0u8; 4];

            reader.read_f32_into::<LittleEndian>(&mut vertex)?;
            reader.read_f32_into::<LittleEndian>(&mut normal)?;
            reader.read_f32_into::<LittleEndian>(&mut uv)?;
            reader.read_exact(&mut tangent)?;

            let mut real_tangents = [0f32; 4];
            uv[1] = 1.0 - uv[1];

            for i in 0..4 {
                real_tangents[i] = tangent[i] as f32 / 127.0 - 1.0;
            }

            if has_vertex_colors {
                let rgba = reader.read_u32::<LittleEndian>()?;
                let vertex_color = [
                    ((rgba >> 24) & 0xFF) as u8,
                    ((rgba >> 16) & 0xFF) as u8,
                    ((rgba >> 8) & 0xFF) as u8,
                    (rgba & 0xFF) as u8,
                ];

                vertex_colors.push(vertex_color);

                reader.seek(SeekFrom::Current(header.vertex_size as i64 - 40))?;
            } else {
                reader.seek(SeekFrom::Current(header.vertex_size as i64 - 36))?;
            }

            vertices[i as usize] = vertex;
            normals[i as usize] = normal;
            uvs[i as usize] = uv;
            tangents[i as usize] = real_tangents;
        }

        if header.bone_count > 0 {
            skin_indices = vec![[0u8; 4]; header.vertex_count as usize];
            skin_weights = vec![[0f32; 4]; header.vertex_count as usize];

            for i in 0..header.vertex_count {
                let mut skin_index = [0u8; 4];
                let mut skin_weight_bytes = [0u8; 4];
                let mut skin_weight_float = [0f32; 4];

                reader.read_exact(&mut skin_index)?;
                reader.read_exact(&mut skin_weight_bytes)?;

                skin_weight_float[0] = skin_weight_bytes[0] as f32 / 255.0;
                skin_weight_float[1] = skin_weight_bytes[1] as f32 / 255.0;
                skin_weight_float[2] = skin_weight_bytes[2] as f32 / 255.0;
                skin_weight_float[3] = skin_weight_bytes[3] as f32 / 255.0;

                skin_indices[i as usize] = skin_index;
                skin_weights[i as usize] = skin_weight_float;
            }
        }

        for i in 0..header.face_count {
            let mut face = [0u32; 3];

            reader.read_u32_into::<LittleEndian>(&mut face)?;
            reader.seek(SeekFrom::Current(header.face_size as i64 - 12))?;

            faces[i as usize] = face;
        }

        if header.lod_count <= 2 {
            lods.push(0);
            lods.push(header.face_count);
            reader.seek(SeekFrom::Current(
                header.lod_count as i64 * header.lod_size as i64,
            ))?;
        } else {
            for _ in 0..header.lod_count {
                lods.push(reader.read_u32::<LittleEndian>()?);
                reader.seek(SeekFrom::Current(header.lod_size as i64 - 4))?;
            }
        }

        if header.bone_count > 0 {
            let name_table_start = reader.position() + header.bone_count as u64 * 60;

            for _ in 0..header.bone_count {
                let name_index = reader.read_u32::<LittleEndian>()?;

                // read name
                let current_position = reader.position();
                reader.seek(SeekFrom::Start(name_table_start + name_index as u64 + 1))?;

                let bone_name = read_string(reader)?;
                reader.seek(SeekFrom::Start(current_position))?;

                let parent_id = reader.read_u16::<LittleEndian>()?;
                let lod_parent_id = reader.read_u16::<LittleEndian>()?;
                let culling = reader.read_f32::<LittleEndian>()?;

                let mut cframe = [0f32; 12];

                for i in 0..9 {
                    cframe[i + 3] = reader.read_f32::<LittleEndian>()?;
                }

                for i in 0..3 {
                    cframe[i] = reader.read_f32::<LittleEndian>()?;
                }

                bones.push(Bone {
                    name: bone_name,
                    parent_id: parent_id as usize,
                    lod_parent_id: lod_parent_id as usize,
                    culling,
                    cframe,
                });
            }
        }

        if header.name_table_size > 0 {
            reader.seek(SeekFrom::Current(header.name_table_size as i64))?;
        }

        if header.subset_count > 0 {
            let mut bone_indices = vec![0; 26];

            for _ in 0..header.subset_count {
                reader.read_u32::<LittleEndian>()?;
                reader.read_u32::<LittleEndian>()?;
                let verts_begin = reader.read_u32::<LittleEndian>()?;
                let verts_size = reader.read_u32::<LittleEndian>()?;
                reader.read_u32::<LittleEndian>()?;

                for i in 0..26 {
                    bone_indices[i as usize] = reader.read_u16::<LittleEndian>()? as usize;
                }

                let verts_end = verts_begin + verts_size;

                for i in verts_begin..verts_end {
                    skin_indices[i as usize][0] =
                        bone_indices[skin_indices[i as usize][0] as usize] as u8;

                    skin_indices[i as usize][1] =
                        bone_indices[skin_indices[i as usize][1] as usize] as u8;

                    skin_indices[i as usize][2] =
                        bone_indices[skin_indices[i as usize][2] as usize] as u8;

                    skin_indices[i as usize][3] =
                        bone_indices[skin_indices[i as usize][3] as usize] as u8;
                }
            }
        }

        if header.facs_data_size > 0 {
            reader.seek(SeekFrom::Current(header.facs_data_size as i64))?;
        }

        if lods.len() > 2 {
            faces = faces[lods[0] as usize..lods[1] as usize].to_vec();

            let mut max_vertices = 0;
            for i in faces.len()..0 {
                let face = faces[i];
                max_vertices = max_vertices.max(face[0]).max(face[1]).max(face[2]);
            }

            max_vertices += 1;

            if max_vertices > vertices.len() as u32 {
                if skin_indices.len() > 0 {
                    skin_indices.resize(max_vertices as usize, [0u8; 4]);
                    skin_weights.resize(max_vertices as usize, [0f32; 4]);
                }

                vertices.resize(max_vertices as usize, [0f32; 3]);
                normals.resize(max_vertices as usize, [0f32; 3]);
                uvs.resize(max_vertices as usize, [0f32; 2]);
                tangents.resize(max_vertices as usize, [0f32; 4]);

                if vertex_colors.len() > 0 {
                    vertex_colors.resize(max_vertices as usize, [0u8; 4]);
                }
            }
        }

        Ok(Self {
            version: header.version.clone(),
            faces,
            vertices,
            uvs,
            normals,
            tangents,
            vertex_colors,
            skin_indices,
            skin_weights,
            bones,
        })
    }
}
