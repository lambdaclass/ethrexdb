use std::sync::{Arc, OnceLock};

// TODO: I'm using the ethrex-trie crate directly because `BranchNode`, `LeafNode` and
// `ExtensionNode` are private in the `ethrex-trie` crate.
use ethrex_trie::{BranchNode, ExtensionNode, LeafNode, Nibbles, Node, NodeRef, TrieError};

// Two-node system tags (MVP format)
/// Tag for Branch node (16 children + 1 value)
const TAG_BRANCH: u8 = 0;
/// Tag for Extend node (combines Extension and Leaf)
const TAG_EXTEND: u8 = 1;

/// Serializes a Merkle Patricia Trie into a byte buffer using the two-node format.
///
/// This follows the ETHREX_DB_MVP.md specification with only two node types:
/// - Branch: 16 node offsets + 1 value offset
/// - Extend: 1 node offset + 1 value offset (combines Extension and Leaf)
///
/// The format uses offsets instead of inline data for better file management.
#[derive(Default)]
pub struct Serializer {
    /// Buffer where serialized data is accumulated
    buffer: Vec<u8>,
}

impl Serializer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serializes a trie using the two-node format
    pub fn serialize_tree(mut self, root: &Node) -> Result<Vec<u8>, TrieError> {
        self.serialize_node(root)?;
        Ok(self.buffer)
    }

    /// Serializes a node, converting from 3-node to 2-node system
    fn serialize_node(&mut self, node: &Node) -> Result<u64, TrieError> {
        let offset = self.buffer.len() as u64;

        match node {
            Node::Leaf(leaf) => {
                // Leaf becomes Extend with only value
                self.buffer.push(TAG_EXTEND);

                // Write nibbles with proper encoding
                let compact_nibbles = leaf.partial.encode_compact();
                self.write_bytes_with_len(&compact_nibbles);

                // Reserve space for offsets
                let value_offset_pos = self.buffer.len();
                self.buffer.extend_from_slice(&0u64.to_le_bytes()); // node offset = 0
                self.buffer.extend_from_slice(&0u64.to_le_bytes()); // value offset placeholder

                // Write value and get its offset
                // Even empty values need to be written to maintain the Extend node structure
                let value_offset = self.buffer.len() as u64;
                self.write_bytes_with_len(&leaf.value);

                // Go back and write the actual value offset
                self.buffer[value_offset_pos + 8..value_offset_pos + 16]
                    .copy_from_slice(&value_offset.to_le_bytes());

                Ok(offset)
            }
            Node::Extension(ext) => {
                // Extension becomes Extend with only child
                self.buffer.push(TAG_EXTEND);

                // Write nibbles
                let compact_prefix = ext.prefix.encode_compact();
                self.write_bytes_with_len(&compact_prefix);

                // Reserve space for offsets (we'll fill them later)
                let child_offset_pos = self.buffer.len();
                self.buffer.extend_from_slice(&0u64.to_le_bytes()); // child offset placeholder
                self.buffer.extend_from_slice(&0u64.to_le_bytes()); // value offset = 0

                // Now serialize child and get its offset
                let child_offset = match &ext.child {
                    NodeRef::Hash(hash) => {
                        if hash.is_valid() {
                            panic!("Hash references not supported in serialization");
                        }
                        0u64 // Empty child
                    }
                    NodeRef::Node(node, _) => self.serialize_node(node)?,
                };

                // Go back and write the actual child offset
                if child_offset > 0 {
                    self.buffer[child_offset_pos..child_offset_pos + 8]
                        .copy_from_slice(&child_offset.to_le_bytes());
                }

                Ok(offset)
            }
            Node::Branch(branch) => {
                // Branch stays Branch but with offsets
                self.buffer.push(TAG_BRANCH);

                // Reserve space for all offsets
                let offsets_start = self.buffer.len();
                // 16 child offsets + 1 value offset
                for _ in 0..17 {
                    self.buffer.extend_from_slice(&0u64.to_le_bytes());
                }

                // Serialize all children and collect their offsets
                let mut child_offsets = [0u64; 16];
                for (i, child) in branch.choices.iter().enumerate() {
                    child_offsets[i] = match child {
                        NodeRef::Hash(hash) => {
                            if hash.is_valid() {
                                panic!("Hash references not supported in serialization");
                            }
                            0u64
                        }
                        NodeRef::Node(node, _) => self.serialize_node(node)?,
                    };
                }

                // Serialize value if present
                let value_offset = if branch.value.is_empty() {
                    0u64
                } else {
                    let offset = self.buffer.len() as u64;
                    self.write_bytes_with_len(&branch.value);
                    offset
                };

                // Go back and write all the actual offsets
                let mut pos = offsets_start;
                for &child_offset in &child_offsets {
                    self.buffer[pos..pos + 8].copy_from_slice(&child_offset.to_le_bytes());
                    pos += 8;
                }
                self.buffer[pos..pos + 8].copy_from_slice(&value_offset.to_le_bytes());

                Ok(offset)
            }
        }
    }

    fn write_bytes_with_len(&mut self, bytes: &[u8]) {
        let len = bytes.len() as u32;
        self.buffer.extend_from_slice(&len.to_le_bytes());
        self.buffer.extend_from_slice(bytes);
    }
}

/// Deserializes a Merkle Patricia Trie from a byte buffer.
///
/// The deserializer reads the binary format produced by [`Serializer`].
/// It uses the two-node format and converts back to the standard 3-node format.
pub struct Deserializer<'a> {
    /// The byte buffer containing serialized trie data
    buffer: &'a [u8],
    /// Current reading position in the buffer
    pub position: usize,
}

impl<'a> Deserializer<'a> {
    /// Creates a new deserializer for the given buffer.
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }

    /// Deserializes a tree from the two-node format back to standard 3-node format
    pub fn decode_tree(&mut self) -> Result<Node, TrieError> {
        let node = self.decode_node()?;
        node.compute_hash();
        Ok(node)
    }

    /// Gets a value by path without deserializing the entire trie
    pub fn get_by_path(&mut self, path: &[u8]) -> Result<Option<Vec<u8>>, TrieError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let nibbles = Nibbles::from_raw(path, false);
        self.get_by_path_inner(nibbles)
    }

    /// Internal helper for get_by_path
    fn get_by_path_inner(&mut self, mut path: Nibbles) -> Result<Option<Vec<u8>>, TrieError> {
        let tag = self.read_byte()?;

        match tag {
            TAG_EXTEND => {
                // Read nibbles
                let compact_nibbles = self.read_bytes_with_len()?;
                let nibbles = Nibbles::decode_compact(&compact_nibbles);

                // Read node offset
                let mut node_offset_bytes = [0u8; 8];
                self.read_exact(&mut node_offset_bytes)?;
                let node_offset = u64::from_le_bytes(node_offset_bytes);

                // Read value offset
                let mut value_offset_bytes = [0u8; 8];
                self.read_exact(&mut value_offset_bytes)?;
                let value_offset = u64::from_le_bytes(value_offset_bytes);

                // Check if this is a leaf (only value, includes empty values)
                if node_offset == 0 && value_offset > 0 {
                    // This is a leaf node - check if paths match
                    let leaf_path_without_flag = if nibbles.is_leaf() {
                        nibbles.slice(0, nibbles.len() - 1)
                    } else {
                        nibbles
                    };

                    if path == leaf_path_without_flag {
                        let saved_pos = self.position;
                        self.position = value_offset as usize;
                        let value = self.read_bytes_with_len()?;
                        self.position = saved_pos;
                        Ok(Some(value))
                    } else {
                        Ok(None)
                    }
                } else if node_offset > 0 && value_offset == 0 {
                    // This is an extension node
                    if !path.skip_prefix(&nibbles) {
                        return Ok(None);
                    }

                    // Continue searching in child
                    let saved_pos = self.position;
                    self.position = node_offset as usize;
                    let result = self.get_by_path_inner(path)?;
                    self.position = saved_pos;
                    Ok(result)
                } else {
                    // Extend with both child and value not supported yet
                    panic!("Extend node with both child and value not yet supported");
                }
            }
            TAG_BRANCH => {
                if path.is_empty() {
                    // We want the branch value
                    // Skip to value offset (after 16 child offsets)
                    self.position += 16 * 8;

                    let mut value_offset_bytes = [0u8; 8];
                    self.read_exact(&mut value_offset_bytes)?;
                    let value_offset = u64::from_le_bytes(value_offset_bytes);

                    if value_offset > 0 {
                        let saved_pos = self.position;
                        self.position = value_offset as usize;
                        let value = self.read_bytes_with_len()?;
                        self.position = saved_pos;
                        Ok(Some(value))
                    } else {
                        Ok(None)
                    }
                } else {
                    // Get next nibble and find corresponding child
                    let next_nibble = match path.next_choice() {
                        Some(nibble) => nibble,
                        None => return Ok(None),
                    };

                    // Read child offset at position next_nibble
                    self.position += next_nibble * 8;
                    let mut child_offset_bytes = [0u8; 8];
                    self.read_exact(&mut child_offset_bytes)?;
                    let child_offset = u64::from_le_bytes(child_offset_bytes);

                    if child_offset > 0 {
                        let saved_pos = self.position;
                        self.position = child_offset as usize;
                        let result = self.get_by_path_inner(path)?;
                        self.position = saved_pos;
                        Ok(result)
                    } else {
                        Ok(None)
                    }
                }
            }
            _ => panic!("Invalid node tag: {}", tag),
        }
    }

    /// Decodes a node from the two-node format
    fn decode_node(&mut self) -> Result<Node, TrieError> {
        let tag = self.read_byte()?;

        match tag {
            TAG_EXTEND => {
                // Read nibbles
                let compact_nibbles = self.read_bytes_with_len()?;
                let nibbles = Nibbles::decode_compact(&compact_nibbles);

                // Read node offset
                let mut node_offset_bytes = [0u8; 8];
                self.read_exact(&mut node_offset_bytes)?;
                let node_offset = u64::from_le_bytes(node_offset_bytes);

                // Read value offset
                let mut value_offset_bytes = [0u8; 8];
                self.read_exact(&mut value_offset_bytes)?;
                let value_offset = u64::from_le_bytes(value_offset_bytes);

                // Determine node type based on what's present
                match (node_offset > 0, value_offset > 0) {
                    (false, true) => {
                        // Only value = Leaf node (includes empty values)
                        let saved_pos = self.position;
                        self.position = value_offset as usize;
                        let value = self.read_bytes_with_len()?;
                        self.position = saved_pos;
                        Ok(Node::Leaf(LeafNode::new(nibbles, value)))
                    }
                    (true, false) => {
                        // Only child = Extension node
                        let saved_pos = self.position;
                        self.position = node_offset as usize;
                        let child = self.decode_node()?;
                        self.position = saved_pos;

                        Ok(Node::Extension(ExtensionNode::new(
                            nibbles,
                            NodeRef::Node(Arc::new(child), OnceLock::new()),
                        )))
                    }
                    (true, true) => {
                        // Both child and value - this is a special case
                        // For now, treat as Extension with child that has the value
                        panic!("Extend node with both child and value not yet supported");
                    }
                    (false, false) => {
                        panic!("Invalid Extend node with no child or value");
                    }
                }
            }
            TAG_BRANCH => {
                // Read 16 child offsets
                let mut child_offsets = [0u64; 16];
                for child in child_offsets.iter_mut() {
                    let mut offset_bytes = [0u8; 8];
                    self.read_exact(&mut offset_bytes)?;
                    *child = u64::from_le_bytes(offset_bytes);
                }

                // Read value offset
                let mut value_offset_bytes = [0u8; 8];
                self.read_exact(&mut value_offset_bytes)?;
                let value_offset = u64::from_le_bytes(value_offset_bytes);

                // Build children NodeRefs
                let mut children: [NodeRef; 16] = Default::default();
                for (i, &offset) in child_offsets.iter().enumerate() {
                    if offset > 0 {
                        let saved_pos = self.position;
                        self.position = offset as usize;
                        let child = self.decode_node()?;
                        self.position = saved_pos;
                        children[i] = NodeRef::Node(Arc::new(child), OnceLock::new());
                    }
                }

                // Read value if present
                let value = if value_offset > 0 {
                    let saved_pos = self.position;
                    self.position = value_offset as usize;
                    let val = self.read_bytes_with_len()?;
                    self.position = saved_pos;
                    val
                } else {
                    vec![]
                };

                Ok(Node::Branch(Box::new(BranchNode::new_with_value(
                    children, value,
                ))))
            }
            _ => panic!("Invalid node tag: {}", tag),
        }
    }

    fn read_byte(&mut self) -> Result<u8, TrieError> {
        if self.position >= self.buffer.len() {
            panic!("Invalid buffer length when trying to read byte");
        }
        let byte = self.buffer[self.position];
        self.position += 1;
        Ok(byte)
    }

    /// Reads exactly `buf.len()` bytes into the provided buffer.
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), TrieError> {
        let end = self.position + buf.len();
        if end > self.buffer.len() {
            panic!("Invalid buffer length when trying to read exact bytes");
        }
        buf.copy_from_slice(&self.buffer[self.position..end]);
        self.position = end;
        Ok(())
    }

    /// Reads a byte array with length prefix.
    /// Format: `[length(4 bytes), data...]`
    fn read_bytes_with_len(&mut self) -> Result<Vec<u8>, TrieError> {
        // Read length prefix
        let mut len_bytes = [0u8; 4];
        self.read_exact(&mut len_bytes)?;
        let len = u32::from_le_bytes(len_bytes);

        // Read data
        let mut data = vec![0u8; len as usize];
        self.read_exact(&mut data)?;
        Ok(data)
    }
}

/// Helper function to serialize a Merkle Patricia Trie node to bytes.
pub fn serialize(node: &Node) -> Vec<u8> {
    Serializer::new().serialize_tree(node).unwrap()
}

#[cfg(test)]
mod test {
    use ethrex_trie::{InMemoryTrieDB, NodeHash, Trie};

    use super::*;

    fn new_temp() -> Trie {
        use std::collections::HashMap;
        use std::sync::Arc;
        use std::sync::Mutex;

        let hmap: HashMap<NodeHash, Vec<u8>> = HashMap::new();
        let map = Arc::new(Mutex::new(hmap));
        let db = InMemoryTrieDB::new(map);
        Trie::new(Box::new(db))
    }

    fn deserialize(buffer: &[u8]) -> Node {
        Deserializer::new(buffer).decode_tree().unwrap()
    }

    #[test]
    fn test_serialize_deserialize_empty_leaf() {
        let leaf = Node::Leaf(LeafNode {
            partial: Nibbles::from_hex(vec![]),
            value: vec![],
        });

        let bytes = serialize(&leaf);
        let recovered = deserialize(&bytes);

        assert_eq!(leaf, recovered);
    }

    #[test]
    fn test_serialize_deserialize_leaf_with_long_path() {
        let leaf = Node::Leaf(LeafNode {
            partial: Nibbles::from_hex(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5]),
            value: b"long_path_value".to_vec(),
        });

        let bytes = serialize(&leaf);
        let recovered = deserialize(&bytes);

        assert_eq!(leaf, recovered);
    }

    #[test]
    fn test_serialize_deserialize_branch_empty() {
        let branch = Node::Branch(Box::new(BranchNode {
            choices: Default::default(),
            value: vec![],
        }));

        let bytes = serialize(&branch);
        let recovered = deserialize(&bytes);

        assert_eq!(branch, recovered);
    }

    #[test]
    fn test_serialize_deserialize_tree_extension_to_leaf() {
        let leaf = Node::Leaf(LeafNode {
            partial: Nibbles::from_hex(vec![5, 6, 7]),
            value: b"nested_leaf".to_vec(),
        });

        let ext = Node::Extension(ExtensionNode {
            prefix: Nibbles::from_hex(vec![1, 2]),
            child: NodeRef::Node(Arc::new(leaf), OnceLock::new()),
        });

        let bytes = serialize(&ext);
        let recovered = deserialize(&bytes);
        assert_eq!(recovered, ext);
        match recovered {
            Node::Extension(ext_node) => {
                assert_eq!(ext_node.prefix, Nibbles::from_hex(vec![1, 2]));
                match &ext_node.child {
                    NodeRef::Node(arc_node, _) => match &**arc_node {
                        Node::Leaf(leaf_node) => {
                            assert_eq!(leaf_node.partial, Nibbles::from_hex(vec![5, 6, 7]));
                            assert_eq!(leaf_node.value, b"nested_leaf");
                        }
                        _ => panic!("Expected leaf node"),
                    },
                    _ => panic!("Expected embedded node"),
                }
            }
            _ => panic!("Expected extension node"),
        }
    }

    #[test]
    fn test_serialize_deserialize_deep_tree() {
        let leaf = Node::Leaf(LeafNode {
            partial: Nibbles::from_hex(vec![9, 8]),
            value: b"deep_leaf".to_vec(),
        });

        let inner_ext = Node::Extension(ExtensionNode {
            prefix: Nibbles::from_hex(vec![5, 6]),
            child: NodeRef::Node(Arc::new(leaf), OnceLock::new()),
        });

        let mut branch_choices: [NodeRef; 16] = Default::default();
        branch_choices[2] = NodeRef::Node(Arc::new(inner_ext), OnceLock::new());

        let branch = Node::Branch(Box::new(BranchNode {
            choices: branch_choices,
            value: vec![],
        }));

        let outer_ext = Node::Extension(ExtensionNode {
            prefix: Nibbles::from_hex(vec![1, 2, 3]),
            child: NodeRef::Node(Arc::new(branch), OnceLock::new()),
        });

        let bytes = serialize(&outer_ext);
        let recovered = deserialize(&bytes);

        assert_eq!(recovered, outer_ext);
    }

    #[test]
    fn test_trie_serialization_empty() {
        let trie = new_temp();
        let root = trie.root_node().unwrap();
        assert!(root.is_none());
    }

    #[test]
    fn test_trie_serialization_single_insert() {
        let mut trie = new_temp();
        trie.insert(b"key".to_vec(), b"value".to_vec()).unwrap();

        let root = trie.root_node().unwrap().unwrap();
        let bytes = serialize(&root);
        let recovered = deserialize(&bytes);

        assert_eq!(root, recovered);
    }

    #[test]
    fn test_trie_serialization_multiple_inserts() {
        let mut trie = new_temp();

        let test_data = vec![
            (b"do".to_vec(), b"verb".to_vec()),
            (b"dog".to_vec(), b"puppy".to_vec()),
            (b"doge".to_vec(), b"coin".to_vec()),
            (b"horse".to_vec(), b"stallion".to_vec()),
        ];

        for (key, value) in &test_data {
            trie.insert(key.clone(), value.clone()).unwrap();
        }

        let root = trie.root_node().unwrap().unwrap();
        let bytes = serialize(&root);
        let recovered = deserialize(&bytes);

        assert_eq!(recovered, root);
    }

    #[test]
    fn test_file_io() {
        use std::fs;

        // Create trie
        let mut trie = new_temp();
        trie.insert(b"file_key".to_vec(), b"file_value".to_vec())
            .unwrap();

        // Serialize to file
        let root = trie.root_node().unwrap().unwrap();
        let serialized = serialize(&root);

        let path = "/tmp/test_trie.mpt";
        fs::write(path, &serialized).unwrap();

        // Read from file and deserialize
        let read_data = fs::read(path).unwrap();
        let deserialized = deserialize(&read_data);

        assert_eq!(root, deserialized);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_get_by_path_serialized_simple() {
        let mut trie = new_temp();
        trie.insert(b"test".to_vec(), b"value".to_vec()).unwrap();

        let root = trie.root_node().unwrap().unwrap();
        let buffer = serialize(&root);

        let mut deserializer = Deserializer::new(&buffer);
        assert_eq!(
            deserializer.get_by_path(b"test").unwrap(),
            Some(b"value".to_vec())
        );

        // Reset position before decoding tree
        let mut deserializer = Deserializer::new(&buffer);
        let recovered = deserializer.decode_tree().unwrap();
        assert_eq!(root, recovered);
    }

    #[test]
    fn test_get_by_path_serialized() {
        let mut trie = new_temp();

        let test_data = vec![
            (b"do".to_vec(), b"verb".to_vec()),
            (b"dog".to_vec(), b"puppy".to_vec()),
            (b"doge".to_vec(), b"coin".to_vec()),
            (b"horse".to_vec(), b"stallion".to_vec()),
        ];

        for (key, value) in &test_data {
            trie.insert(key.clone(), value.clone()).unwrap();
        }

        let root = trie.root_node().unwrap().unwrap();
        let buffer = serialize(&root);

        let mut deserializer = Deserializer::new(&buffer);
        assert_eq!(
            deserializer.get_by_path(b"horse").unwrap(),
            Some(b"stallion".to_vec())
        );

        let mut deserializer = Deserializer::new(&buffer);
        assert_eq!(
            deserializer.get_by_path(b"dog").unwrap(),
            Some(b"puppy".to_vec())
        );

        let mut deserializer = Deserializer::new(&buffer);
        assert_eq!(
            deserializer.get_by_path(b"doge").unwrap(),
            Some(b"coin".to_vec())
        );

        let mut deserializer = Deserializer::new(&buffer);
        assert_eq!(
            deserializer.get_by_path(b"do").unwrap(),
            Some(b"verb".to_vec())
        );

        let mut deserializer = Deserializer::new(&buffer);
        assert_eq!(deserializer.get_by_path(b"cat").unwrap(), None);

        let mut deserializer = Deserializer::new(&buffer);
        assert_eq!(deserializer.get_by_path(b"").unwrap(), None);

        // Reset position before decoding tree
        let mut deserializer = Deserializer::new(&buffer);
        let recovered = deserializer.decode_tree().unwrap();
        assert_eq!(root, recovered);
    }

    #[test]
    fn test_complex_trie_serialization() {
        let mut trie = new_temp();

        let test_data = vec![
            (b"app".to_vec(), b"application".to_vec()),
            (b"apple".to_vec(), b"fruit".to_vec()),
            (b"application".to_vec(), b"software".to_vec()),
            (b"append".to_vec(), b"add_to_end".to_vec()),
            (b"applied".to_vec(), b"past_tense".to_vec()),
            (b"car".to_vec(), b"vehicle".to_vec()),
            (b"card".to_vec(), b"playing_card".to_vec()),
            (b"care".to_vec(), b"attention".to_vec()),
            (b"career".to_vec(), b"profession".to_vec()),
            (b"careful".to_vec(), b"cautious".to_vec()),
            (b"test".to_vec(), b"examination".to_vec()),
            (b"testing".to_vec(), b"verification".to_vec()),
            (b"tester".to_vec(), b"one_who_tests".to_vec()),
            (b"testament".to_vec(), b"will_document".to_vec()),
            (b"a".to_vec(), b"letter_a".to_vec()),
            (b"b".to_vec(), b"letter_b".to_vec()),
            (b"c".to_vec(), b"letter_c".to_vec()),
            (b"d".to_vec(), b"letter_d".to_vec()),
            (b"e".to_vec(), b"letter_e".to_vec()),
            (b"0x123456".to_vec(), b"hex_value_1".to_vec()),
            (b"0x123abc".to_vec(), b"hex_value_2".to_vec()),
            (b"0x124000".to_vec(), b"hex_value_3".to_vec()),
            (b"0xabcdef".to_vec(), b"hex_value_4".to_vec()),
            (
                b"very_long_key_that_creates_deep_structure_in_trie_1234567890".to_vec(),
                b"long_value_1".to_vec(),
            ),
            (
                b"very_long_key_that_creates_deep_structure_in_trie_abcdefghijk".to_vec(),
                b"long_value_2".to_vec(),
            ),
            (b"empty_value_key".to_vec(), vec![]),
            (b"similar_key_1".to_vec(), b"value_1".to_vec()),
            (b"similar_key_2".to_vec(), b"value_2".to_vec()),
            (b"similar_key_3".to_vec(), b"value_3".to_vec()),
            (b"123".to_vec(), b"number_123".to_vec()),
            (b"1234".to_vec(), b"number_1234".to_vec()),
            (b"12345".to_vec(), b"number_12345".to_vec()),
        ];

        for (key, value) in &test_data {
            trie.insert(key.clone(), value.clone()).unwrap();
        }

        let root = trie.root_node().unwrap().unwrap();

        let buffer = serialize(&root);

        for (key, expected_value) in &test_data {
            let mut deserializer = Deserializer::new(&buffer);
            let retrieved_value = deserializer.get_by_path(key).unwrap();
            assert_eq!(retrieved_value, Some(expected_value.clone()));
        }

        let non_existent_keys = vec![
            b"nonexistent".to_vec(),
            b"app_wrong".to_vec(),
            b"car_wrong".to_vec(),
            b"test_wrong".to_vec(),
            b"0x999999".to_vec(),
            b"similar_key_4".to_vec(),
            b"".to_vec(),
            b"very_long_nonexistent_key".to_vec(),
        ];

        for key in &non_existent_keys {
            let mut deserializer = Deserializer::new(&buffer);
            let result = deserializer.get_by_path(key).unwrap();
            assert_eq!(result, None);
        }

        let mut deserializer = Deserializer::new(&buffer);
        let recovered = deserializer.decode_tree().unwrap();

        assert_eq!(root, recovered);
    }
}
