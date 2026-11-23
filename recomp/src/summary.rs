//! Deterministic JSON summary of a discovery run (addresses and counts only, never code bytes).

use std::fmt::Write as _;

use crate::cfg::{CallTarget, Edge};
use crate::code_image::CodeImage;
use crate::discover::Analysis;
use crate::reloc::RelocKind;

const EDGE_KINDS: [&str; 6] = [
    "fallthrough",
    "taken",
    "jump",
    "tail_call",
    "indirect_jump",
    "return",
];

const RELOC_KINDS: [RelocKind; 8] = [
    RelocKind::CallTarget,
    RelocKind::BranchTarget,
    RelocKind::RipData,
    RelocKind::RipAddress,
    RelocKind::CallSlot,
    RelocKind::JumpSlot,
    RelocKind::AbsoluteImmediate,
    RelocKind::AbsoluteDisplacement,
];

fn edge_index(edge: &Edge) -> usize {
    match edge {
        Edge::Fallthrough(_) => 0,
        Edge::Taken(_) => 1,
        Edge::Jump(_) => 2,
        Edge::TailCall(_) => 3,
        Edge::IndirectJump { .. } => 4,
        Edge::Return => 5,
    }
}

fn hex_list(values: impl IntoIterator<Item = u64>) -> String {
    let items: Vec<String> = values.into_iter().map(|v| format!("\"0x{v:x}\"")).collect();
    format!("[{}]", items.join(", "))
}

/// Render the summary as pretty-printed JSON with a trailing newline.
pub fn render_summary_json(image: &CodeImage<'_>, analysis: &Analysis) -> String {
    let mut edge_counts = [0usize; EDGE_KINDS.len()];
    let mut block_count = 0;
    let mut instruction_count = 0;
    let mut diagnostic_count = 0;
    for function in analysis.functions.values() {
        block_count += function.blocks.len();
        instruction_count += function.instruction_count();
        diagnostic_count += function.diagnostics().count();
        for block in function.blocks.values() {
            for edge in &block.successors {
                edge_counts[edge_index(edge)] += 1;
            }
        }
    }

    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(out, "  \"image_base\": \"0x{:x}\",", image.base());
    let _ = writeln!(out, "  \"image_end\": \"0x{:x}\",", image.end());
    let _ = writeln!(
        out,
        "  \"entry_points\": {},",
        hex_list(image.entry_points.iter().copied())
    );
    let _ = writeln!(out, "  \"function_count\": {},", analysis.functions.len());
    let _ = writeln!(out, "  \"block_count\": {block_count},");
    let _ = writeln!(out, "  \"instruction_count\": {instruction_count},");
    let _ = writeln!(out, "  \"diagnostic_count\": {diagnostic_count},");

    out.push_str("  \"edges\": {");
    for (i, (name, count)) in EDGE_KINDS.iter().zip(edge_counts).enumerate() {
        let sep = if i == 0 { "" } else { ", " };
        let _ = write!(out, "{sep}\"{name}\": {count}");
    }
    out.push_str("},\n");

    out.push_str("  \"relocations\": {");
    for (i, kind) in RELOC_KINDS.iter().enumerate() {
        let count = analysis
            .relocations
            .entries()
            .iter()
            .filter(|r| r.kind == *kind)
            .count();
        let sep = if i == 0 { "" } else { ", " };
        let _ = write!(out, "{sep}\"{}\": {count}", kind.as_str());
    }
    out.push_str("},\n");

    out.push_str("  \"functions\": [");
    for (i, function) in analysis.functions.values().enumerate() {
        out.push_str(if i == 0 { "\n" } else { ",\n" });
        let indirect_calls = function
            .calls()
            .filter(|c| matches!(c.target, CallTarget::Indirect { .. }))
            .count();
        let diagnostics: Vec<String> = function
            .diagnostics()
            .map(|d| {
                format!(
                    "{{\"site\": \"0x{:x}\", \"address\": \"0x{:x}\", \"kind\": \"{}\"}}",
                    d.site,
                    d.address,
                    d.kind.as_str()
                )
            })
            .collect();
        let _ = write!(
            out,
            "    {{\"entry\": \"0x{:x}\", \"blocks\": {}, \"instructions\": {}, \"callees\": {}, \
             \"tail_calls\": {}, \"indirect_calls\": {indirect_calls}, \"diagnostics\": [{}]}}",
            function.entry,
            function.blocks.len(),
            function.instruction_count(),
            hex_list(function.callees.iter().copied()),
            hex_list(function.tail_calls()),
            diagnostics.join(", ")
        );
    }
    if analysis.functions.is_empty() {
        out.push_str("]\n");
    } else {
        out.push_str("\n  ]\n");
    }
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_image::CodeSection;
    use crate::discover::discover;

    #[test]
    fn summary_is_deterministic_and_has_no_code_bytes() {
        let code = [
            0xe8, 0x06, 0x00, 0x00, 0x00, // 0: call 0xb
            0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // 5: jmp [rip+0] (slot at 0xb)
            0xc3, // b: ret
        ];
        let image = CodeImage::new(
            0x1000,
            vec![CodeSection::new(0x1000, &code, true)],
            vec![0x1000],
        );
        let first = render_summary_json(&image, &discover(&image));
        let second = render_summary_json(&image, &discover(&image));
        assert_eq!(first, second);
        assert_eq!(
            first,
            concat!(
                "{\n",
                "  \"image_base\": \"0x1000\",\n",
                "  \"image_end\": \"0x100c\",\n",
                "  \"entry_points\": [\"0x1000\"],\n",
                "  \"function_count\": 2,\n",
                "  \"block_count\": 2,\n",
                "  \"instruction_count\": 3,\n",
                "  \"diagnostic_count\": 0,\n",
                "  \"edges\": {\"fallthrough\": 0, \"taken\": 0, \"jump\": 0, \"tail_call\": 0, ",
                "\"indirect_jump\": 1, \"return\": 1},\n",
                "  \"relocations\": {\"call_target\": 1, \"branch_target\": 0, \"rip_data\": 0, ",
                "\"rip_address\": 0, \"call_slot\": 0, \"jump_slot\": 1, \"absolute_immediate\": 0, ",
                "\"absolute_displacement\": 0},\n",
                "  \"functions\": [\n",
                "    {\"entry\": \"0x1000\", \"blocks\": 1, \"instructions\": 2, \"callees\": [\"0x100b\"], ",
                "\"tail_calls\": [], \"indirect_calls\": 0, \"diagnostics\": []},\n",
                "    {\"entry\": \"0x100b\", \"blocks\": 1, \"instructions\": 1, \"callees\": [], ",
                "\"tail_calls\": [], \"indirect_calls\": 0, \"diagnostics\": []}\n",
                "  ]\n",
                "}\n",
            )
        );
    }
}
