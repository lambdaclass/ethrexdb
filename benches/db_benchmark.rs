use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ethrex_storage::trie_db::libmdbx::LibmdbxTrieDB;
use ethrex_storage::trie_db::test_utils::libmdbx::TestNodes;
use ethrex_trie::{InMemoryTrieDB, Trie};
use ethrexdb::EthrexDB;
use libmdbx::orm::{Database, Decodable, Encodable, Table, table_info};
use libmdbx::{DatabaseOptions, Mode, PageSize, ReadWriteOptions, table};
use rand::{seq::SliceRandom, thread_rng};
use sha3::{Digest, Keccak256};
use std::{sync::Arc, time::Duration};

use tempdir::TempDir;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathKey(Vec<u8>);

impl Encodable for PathKey {
    type Encoded = Vec<u8>;
    fn encode(self) -> Self::Encoded {
        self.0
    }
}

impl Decodable for PathKey {
    fn decode(b: &[u8]) -> anyhow::Result<Self> {
        Ok(PathKey(b.to_vec()))
    }
}

table!(
    /// Path-based table for storing key-value pairs directly by path
    (PathNodes) PathKey => Vec<u8>
);

// Generate test data (key = hash, value = account info)
fn generate_test_data(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    (1..=n)
        .map(|i| {
            // 32-byte key (hash)
            let key = Keccak256::new()
                .chain_update(i.to_be_bytes())
                .finalize()
                .to_vec();

            // 104-byte value (account info: 2 hashes + u256 + u64)
            let mut value = Vec::with_capacity(104);
            value.extend_from_slice(
                &Keccak256::new()
                    .chain_update((i * 2).to_be_bytes())
                    .finalize(),
            );
            value.extend_from_slice(
                &Keccak256::new()
                    .chain_update((i * 3).to_be_bytes())
                    .finalize(),
            );
            value.extend_from_slice(&[0u8; 24]); // u256 padding
            value.extend_from_slice(&(i as u64).to_be_bytes()); // u256 value
            value.extend_from_slice(&(i as u64).to_be_bytes()); // u64

            (key, value)
        })
        .collect()
}

fn create_libmdbx_db<T: Table>(path: std::path::PathBuf) -> Arc<Database> {
    let tables = [table_info!(T)].into_iter().collect();
    let options = DatabaseOptions {
        page_size: Some(PageSize::Set(4096)),
        mode: Mode::ReadWrite(ReadWriteOptions {
            max_size: Some(1024 * 1024 * 1024),
            ..Default::default()
        }),
        ..Default::default()
    };

    Arc::new(
        Database::create_with_options(Some(path), options, &tables)
            .expect("Failed to create LibMDBX database"),
    )
}

struct LibmdbxHashDB {
    trie: Trie,
}

impl LibmdbxHashDB {
    fn new(temp_dir: &std::path::Path) -> Self {
        let db = create_libmdbx_db::<TestNodes>(temp_dir.into());
        let trie = Trie::new(Box::new(LibmdbxTrieDB::<TestNodes>::new(db.clone())));
        Self { trie }
    }

    fn insert_batch(&mut self, data: &[(Vec<u8>, Vec<u8>)]) {
        for (key, value) in data {
            self.trie.insert(key.clone(), value.clone()).unwrap();
        }
        self.trie.commit().unwrap();
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.trie.get(&key.to_vec()).unwrap()
    }
}

struct LibmdbxPathDB {
    db: Arc<Database>,
}

impl LibmdbxPathDB {
    fn new(temp_dir: &std::path::Path) -> Self {
        let db = create_libmdbx_db::<PathNodes>(temp_dir.into());
        Self { db }
    }

    fn insert_batch(&self, data: &[(Vec<u8>, Vec<u8>)]) {
        let txn = self.db.begin_readwrite().unwrap();
        for (key, value) in data {
            txn.upsert::<PathNodes>(PathKey(key.clone()), value.clone())
                .unwrap();
        }
        txn.commit().unwrap();
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let txn = self.db.begin_read().unwrap();
        txn.get::<PathNodes>(PathKey(key.to_vec())).unwrap()
    }
}

fn create_ethrex_db(data: &[(Vec<u8>, Vec<u8>)], file_path: &std::path::Path) -> EthrexDB {
    let mut db = EthrexDB::new(file_path.into()).unwrap();
    let mut trie = Trie::new(Box::new(InMemoryTrieDB::new_empty()));
    for (key, value) in data {
        trie.insert(key.clone(), value.clone()).unwrap();
    }
    db.commit(&trie).unwrap();
    db
}

fn batch_insert_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");
    group.measurement_time(Duration::from_secs(15));

    for size in [1_000, 10_000, 100_000] {
        let data = generate_test_data(size);

        // Hash
        group.bench_with_input(BenchmarkId::new("libmdbx_hash", size), &data, |b, data| {
            b.iter_with_setup(
                || {
                    let temp_dir = TempDir::new("libmdbx_hash_bench").unwrap();
                    LibmdbxHashDB::new(temp_dir.path())
                },
                |mut db| {
                    db.insert_batch(black_box(data));
                    black_box(db)
                },
            );
        });

        // Path
        group.bench_with_input(BenchmarkId::new("libmdbx_path", size), &data, |b, data| {
            b.iter_with_setup(
                || {
                    let temp_dir = TempDir::new("libmdbx_path_bench").unwrap();
                    LibmdbxPathDB::new(temp_dir.path())
                },
                |db| {
                    db.insert_batch(black_box(data));
                    black_box(db)
                },
            );
        });

        // EthrexDB
        group.bench_with_input(BenchmarkId::new("ethrex_db", size), &data, |b, data| {
            b.iter_with_setup(
                || {
                    let temp_dir = TempDir::new("ethrex_bench").unwrap();
                    temp_dir.path().join("test.edb")
                },
                |file_path| {
                    let db = create_ethrex_db(black_box(data), black_box(&file_path));
                    black_box(db)
                },
            );
        });
    }

    group.finish();
}

fn random_read_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_read");
    group.measurement_time(Duration::from_secs(15));

    for size in [1_000, 10_000, 100_000] {
        let data = generate_test_data(size);

        let sample_size = std::cmp::max(1, size / 1000);
        let mut indices: Vec<usize> = (0..size).collect();
        indices.shuffle(&mut thread_rng());
        let sample_keys: Vec<_> = indices[..sample_size]
            .iter()
            .map(|&i| data[i].0.clone())
            .collect();

        let libmdbx_hash_temp = TempDir::new("libmdbx_hash_read").unwrap();
        let mut libmdbx_hash_db = LibmdbxHashDB::new(libmdbx_hash_temp.path());
        libmdbx_hash_db.insert_batch(&data);

        let libmdbx_path_temp = TempDir::new("libmdbx_path_read").unwrap();
        let libmdbx_path_db = LibmdbxPathDB::new(libmdbx_path_temp.path());
        libmdbx_path_db.insert_batch(&data);

        let ethrex_temp = TempDir::new("ethrex_read").unwrap();
        let ethrex_file = ethrex_temp.path().join("test.edb");
        let _ethrex_db = create_ethrex_db(&data, &ethrex_file);

        group.bench_with_input(
            BenchmarkId::new("libmdbx_hash", size),
            &sample_keys,
            |b, keys| {
                b.iter(|| {
                    let mut found = 0;
                    for key in keys {
                        if libmdbx_hash_db.get(black_box(key)).is_some() {
                            found += 1;
                        }
                    }
                    black_box(found)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("libmdbx_path", size),
            &sample_keys,
            |b, keys| {
                b.iter(|| {
                    let mut found = 0;
                    for key in keys {
                        if libmdbx_path_db.get(black_box(key)).is_some() {
                            found += 1;
                        }
                    }
                    black_box(found)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ethrex_db", size),
            &sample_keys,
            |b, keys| {
                b.iter(|| {
                    let mut found = 0;
                    for key in keys {
                        let mut db = EthrexDB::open(ethrex_file.clone()).unwrap();
                        if db.get(black_box(key)).unwrap().is_some() {
                            found += 1;
                        }
                    }
                    black_box(found)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, batch_insert_benchmark, random_read_benchmark);
criterion_main!(benches);
