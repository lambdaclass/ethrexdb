# EthrexDB

**EthrexDB** is a lightweight Merkle Patricia Trie (MPT) based database, designed to serve as a foundational storage layer for Ethereum execution environments.

Most blockchains separate the world state from the proof structure, typically using a general-purpose key-value store (like RocksDB) alongside a Merkle Patricia Trie. This leads to redundant data storage and inefficiencies—each state update can require O((log N)²) disk operations due to nested log-time insertions. EthrexDB solves this by tightly coupling the Merkle tree with the state storage itself. This unified approach removes duplication, reduces I/O overhead, and enables faster state transitions with native proof support.

It draws inspiration from [NOMT](https://github.com/thrumdev/nomt), [QMDB](https://github.com/LayerZero-Labs/qmdb), [Paprika](https://github.com/NethermindEth/Paprika) and [MonadDB](https://docs.monad.xyz/monad-arch/execution/monaddb), focusing on simplicity and modularity.

## Features


## Use Cases



## Getting Started

### Installation

