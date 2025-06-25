# ethrex_db

**EthrexDB** is a lightweight Merkle Patricia Trie (MPT) based database, designed to serve as a foundational storage layer for Ethereum execution environments.

Most blockchains separate the world state from the proof structure, typically using a general purpose key value store alongside a Merkle Patricia Trie. This leads to redundant data storage and inefficiencies each state update can require O((log N)²) disk operations due to nested log time insertions. EthrexDB solves this by tightly coupling the Merkle tree with the state storage itself. This unified approach removes duplication, reduces I/O overhead, and enables faster state transitions with native proof support.


## Getting Started

### Installation

## 📚 References and acknowledgements
The following links, repos, companies and projects have been important in the development of this library and we want to thank and acknowledge them.

- [NOMT](https://github.com/thrumdev/nomt)
- [QMDB](https://github.com/LayerZero-Labs/qmdb)
- [Paprika](https://github.com/NethermindEth/Paprika)
- [MonadDB](https://docs.monad.xyz/monad-arch/execution/monaddb)
- [Database Internals](https://www.databass.dev/)
- [PingCAP Talent Plan](https://github.com/pingcap/talent-plan)
- [Readings in Database System](http://www.redbook.io/)

We're thankful to the teams that created this databases since they were crucial for us to be able to create ethrex_db.
