//! Incrementally maintained predecessor index for [`super::Discovery`].
//!
//! The fixpoint used to rebuild the whole predecessor map from the block map before every
//! jump-table phase (about 30 times per `discover()` round on a 4M-block image). This index is
//! updated whenever a block's successors change instead, and must always equal
//! [`PredIndex::from_blocks`] of the current block map. That includes the order of the
//! predecessors: ascending source block, then the source's successor order. The matchers read the
//! first predecessor and the full list in that order.

use std::collections::BTreeMap;

#[cfg(test)]
use crate::cfg::BasicBlock;
use crate::cfg::Edge;

/// Incoming intra-function edges of every block: target -> `(source block, edge)`, sorted by
/// source block, edges of one source in its successor order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct PredIndex {
    map: BTreeMap<u64, Vec<(u64, Edge)>>,
}

/// The incoming edges of one block.
#[derive(Debug, Clone, Copy)]
pub(super) struct Pred<'p> {
    /// Number of incoming edges (an edge per successor slot, so a block whose taken and
    /// fallthrough edges both lead here counts twice).
    pub count: u32,
    /// Source of the first incoming edge (the lowest source block).
    pub from: u64,
    /// The first incoming edge.
    pub edge: Edge,
    entries: &'p [(u64, Edge)],
}

impl Pred<'_> {
    /// Source block of every incoming edge, `from` first.
    pub fn sources(&self) -> Vec<u64> {
        self.entries.iter().map(|&(from, _)| from).collect()
    }
}

impl PredIndex {
    /// The index of `blocks`, built from scratch (the reference the incremental updates match).
    #[cfg(test)]
    pub fn from_blocks(blocks: &BTreeMap<u64, BasicBlock>) -> Self {
        let mut map: BTreeMap<u64, Vec<(u64, Edge)>> = BTreeMap::new();
        for (&from, block) in blocks {
            for &edge in &block.successors {
                if let Some(target) = edge.intra_target() {
                    map.entry(target).or_default().push((from, edge));
                }
            }
        }
        Self { map }
    }

    /// Incoming edges of `target`, `None` when it has none.
    pub fn get(&self, target: &u64) -> Option<Pred<'_>> {
        let entries = self.map.get(target)?;
        let &(from, edge) = entries.first()?;
        Some(Pred {
            count: u32::try_from(entries.len()).unwrap_or(u32::MAX),
            from,
            edge,
            entries,
        })
    }

    /// Record the successors of block `from`. `from` must have no edges in the index yet (remove
    /// the old successors first when replacing them).
    pub fn add(&mut self, from: u64, successors: &[Edge]) {
        for &edge in successors {
            let Some(target) = edge.intra_target() else {
                continue;
            };
            let list = self.map.entry(target).or_default();
            // After every existing edge of `from` (an earlier successor of the same block) and
            // before every higher source, which keeps the successor order within a block.
            let at = list.partition_point(|&(f, _)| f <= from);
            list.insert(at, (from, edge));
        }
    }

    /// Forget the successors of block `from` (the edges previously passed to [`Self::add`]).
    pub fn remove(&mut self, from: u64, successors: &[Edge]) {
        for &edge in successors {
            let Some(target) = edge.intra_target() else {
                continue;
            };
            if let Some(list) = self.map.get_mut(&target) {
                list.retain(|&(f, _)| f != from);
                if list.is_empty() {
                    self.map.remove(&target);
                }
            }
        }
    }

    /// Number of blocks with at least one incoming edge.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(start: u64, successors: &[Edge]) -> BasicBlock {
        let mut b = BasicBlock::new(start);
        b.successors = successors.to_vec();
        b
    }

    fn rebuild(blocks: &BTreeMap<u64, BasicBlock>) -> PredIndex {
        PredIndex::from_blocks(blocks)
    }

    #[test]
    fn empty_index_has_no_preds() {
        let idx = PredIndex::default();
        assert!(idx.get(&0x1000).is_none());
        assert_eq!(idx, rebuild(&BTreeMap::new()));
    }

    #[test]
    fn adds_in_any_order_match_rebuild() {
        let mut blocks = BTreeMap::new();
        blocks.insert(0x30, block(0x30, &[Edge::Jump(0x10)]));
        blocks.insert(
            0x10,
            block(0x10, &[Edge::Taken(0x40), Edge::Fallthrough(0x20)]),
        );
        blocks.insert(0x20, block(0x20, &[Edge::Fallthrough(0x40), Edge::Return]));
        blocks.insert(0x40, block(0x40, &[Edge::IndirectJump { slot: None }]));
        blocks.insert(
            0x50,
            block(0x50, &[Edge::TailCall(0x10), Edge::JumpTable(0x40)]),
        );
        // Insert in descending order: the index must still sort sources ascending.
        let mut idx = PredIndex::default();
        for (&from, b) in blocks.iter().rev() {
            idx.add(from, &b.successors);
        }
        assert_eq!(idx, rebuild(&blocks));
        let p = idx.get(&0x40).unwrap();
        assert_eq!((p.count, p.from, p.edge), (3, 0x10, Edge::Taken(0x40)));
        assert_eq!(p.sources(), vec![0x10, 0x20, 0x50]);
        // Tail calls are not intra-function edges.
        assert_eq!(idx.get(&0x10).unwrap().sources(), vec![0x30]);
    }

    #[test]
    fn duplicate_edges_from_one_block_keep_successor_order() {
        let mut blocks = BTreeMap::new();
        blocks.insert(0x20, block(0x20, &[Edge::Jump(0x30)]));
        blocks.insert(
            0x10,
            block(0x10, &[Edge::Taken(0x30), Edge::Fallthrough(0x30)]),
        );
        let mut idx = PredIndex::default();
        idx.add(0x20, &blocks[&0x20].successors);
        idx.add(0x10, &blocks[&0x10].successors);
        assert_eq!(idx, rebuild(&blocks));
        let p = idx.get(&0x30).unwrap();
        assert_eq!((p.count, p.edge), (3, Edge::Taken(0x30)));
        assert_eq!(p.sources(), vec![0x10, 0x10, 0x20]);
    }

    #[test]
    fn remove_and_replace_successors_match_rebuild() {
        let mut blocks = BTreeMap::new();
        blocks.insert(0x10, block(0x10, &[Edge::Jump(0x30)]));
        blocks.insert(0x20, block(0x20, &[Edge::IndirectJump { slot: None }]));
        let mut idx = rebuild(&blocks);
        // Resolve the jump at 0x20 to a table.
        let table = [Edge::JumpTable(0x30), Edge::JumpTable(0x40)];
        idx.remove(0x20, &blocks[&0x20].successors);
        idx.add(0x20, &table);
        blocks.get_mut(&0x20).unwrap().successors = table.to_vec();
        assert_eq!(idx, rebuild(&blocks));
        assert_eq!(idx.get(&0x30).unwrap().sources(), vec![0x10, 0x20]);
        // Demote it again: 0x40 loses its only predecessor and leaves the index.
        let demoted = [Edge::IndirectJump { slot: None }];
        idx.remove(0x20, &blocks[&0x20].successors);
        idx.add(0x20, &demoted);
        blocks.get_mut(&0x20).unwrap().successors = demoted.to_vec();
        assert_eq!(idx, rebuild(&blocks));
        assert!(idx.get(&0x40).is_none());
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn split_style_update_matches_rebuild() {
        // 0x10 falls into 0x30 and also jumps to itself; split it at 0x18.
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x10,
            block(0x10, &[Edge::Taken(0x10), Edge::Fallthrough(0x30)]),
        );
        blocks.insert(0x30, block(0x30, &[Edge::Return]));
        let mut idx = rebuild(&blocks);
        let tail = blocks[&0x10].successors.clone();
        idx.remove(0x10, &tail);
        idx.add(0x10, &[Edge::Fallthrough(0x18)]);
        idx.add(0x18, &tail);
        blocks.get_mut(&0x10).unwrap().successors = vec![Edge::Fallthrough(0x18)];
        blocks.insert(0x18, block(0x18, &tail));
        assert_eq!(idx, rebuild(&blocks));
        assert_eq!(idx.get(&0x10).unwrap().sources(), vec![0x18]);
    }
}
