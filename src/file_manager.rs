use ethrex_trie::TrieError;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Responsible for file management and offsets
pub struct FileManager {
    /// File where the data is stored
    file: File,
}

impl FileManager {
    /// Create a new database file
    pub fn create(file_path: PathBuf) -> Result<Self, TrieError> {
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .unwrap();

        // Write initial offset (8 bytes = 0)
        let initial_offset = 0u64;
        file.write_all(&initial_offset.to_le_bytes()).unwrap();

        file.flush().unwrap();

        Ok(Self { file })
    }

    /// Open an existing file
    pub fn open(file_path: PathBuf) -> Result<Self, TrieError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&file_path)
            .unwrap();

        Ok(Self { file })
    }

    /// Read the offset of the latest root (first 8 bytes of the file)
    pub fn read_latest_root_offset(&mut self) -> Result<u64, TrieError> {
        self.file.seek(SeekFrom::Start(0)).unwrap();

        let mut offset_bytes = [0u8; 8];
        self.file.read_exact(&mut offset_bytes).unwrap();

        Ok(u64::from_le_bytes(offset_bytes))
    }

    /// Update the offset of the latest root (first 8 bytes of the file)
    pub fn update_latest_root_offset(&mut self, new_offset: u64) -> Result<(), TrieError> {
        self.file.seek(SeekFrom::Start(0)).unwrap();

        self.file.write_all(&new_offset.to_le_bytes()).unwrap();

        self.file.flush().unwrap();

        Ok(())
    }

    /// Write data at the end of the file and return the offset where it was written
    pub fn write_at_end(&mut self, data: &[u8]) -> Result<u64, TrieError> {
        let offset = self.file.seek(SeekFrom::End(0)).unwrap();

        self.file.write_all(data).unwrap();

        self.file.flush().unwrap();

        Ok(offset)
    }

    /// Read data from a specific offset to the end of the file
    pub fn read_from_offset_to_end(&mut self, offset: u64) -> Result<Vec<u8>, TrieError> {
        self.file.seek(SeekFrom::Start(offset)).unwrap();

        let mut data = Vec::new();
        self.file.read_to_end(&mut data).unwrap();

        Ok(data)
    }

    /// Read exactly n bytes from the current position
    pub fn read_exact_bytes(&mut self, size: usize) -> Result<Vec<u8>, TrieError> {
        let mut data = vec![0u8; size];
        self.file.read_exact(&mut data).unwrap();
        Ok(data)
    }

    /// Get the current position in the file
    pub fn current_position(&mut self) -> Result<u64, TrieError> {
        Ok(self.file.stream_position().unwrap())
    }

    /// Move to a specific position
    pub fn seek_to(&mut self, offset: u64) -> Result<(), TrieError> {
        self.file.seek(SeekFrom::Start(offset)).unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_create_and_read_offset() {
        let temp_dir = TempDir::new("file_manager_test").unwrap();
        let file_path = temp_dir.path().join("test.db");

        let mut fm = FileManager::create(file_path.clone()).unwrap();

        // The initial offset should be 0
        assert_eq!(fm.read_latest_root_offset().unwrap(), 0);

        fm.update_latest_root_offset(123).unwrap();
        assert_eq!(fm.read_latest_root_offset().unwrap(), 123);
    }

    #[test]
    fn test_write_and_read_data() {
        let temp_dir = TempDir::new("file_manager_test").unwrap();
        let file_path = temp_dir.path().join("test.db");

        let mut fm = FileManager::create(file_path.clone()).unwrap();

        let test_data = b"hello, ethrexdb!";
        let offset = fm.write_at_end(test_data).unwrap();

        // The offset should be 8
        assert_eq!(offset, 8);

        let mut data = vec![0u8; test_data.len()];
        fm.file.seek(SeekFrom::Start(offset)).unwrap();
        fm.file.read_exact(&mut data).unwrap();

        assert_eq!(data, test_data);
    }

    #[test]
    fn test_open_existing() {
        let temp_dir = TempDir::new("file_manager_test").unwrap();
        let file_path = temp_dir.path().join("test.db");

        {
            let mut fm = FileManager::create(file_path.clone()).unwrap();
            fm.update_latest_root_offset(456).unwrap();
            fm.write_at_end(b"persistent data").unwrap();
        }

        let mut fm = FileManager::open(file_path).unwrap();
        assert_eq!(fm.read_latest_root_offset().unwrap(), 456);

        let data = fm.read_from_offset_to_end(8).unwrap();
        assert_eq!(data, b"persistent data");
    }
}
