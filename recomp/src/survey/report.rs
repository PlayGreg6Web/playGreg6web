//! Deterministic text and JSON rendering of a [`Survey`]. Only addresses, GUID constants, source
//! labels, import names and counts are printed.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use super::{ArgKey, KeyReport, Survey, evidence_str};

fn key_label(args: &[ArgKey]) -> String {
    if args.is_empty() {
        "-".to_string()
    } else {
        args.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" / ")
    }
}

fn hex_list(values: &[u64]) -> String {
    values
        .iter()
        .map(|v| format!("0x{v:x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn counts(map: &BTreeMap<&'static str, u32>) -> String {
    if map.is_empty() {
        return "none".to_string();
    }
    map.iter()
        .map(|(k, v)| format!("{k} {v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Fixed-width text table.
pub fn render_text(s: &Survey) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "interface survey: {} functions discovered, {} analysed in {} contexts, {} rounds, \
         {} aborted{}",
        s.stats.functions_discovered,
        s.stats.functions_analyzed,
        s.stats.contexts,
        s.stats.rounds,
        s.stats.aborted,
        if s.stats.context_limit_hit {
            " (context limit hit)"
        } else {
            ""
        }
    );
    let _ = writeln!(out, "guard slots: {}", hex_list(&s.guard_slots));
    for src in &s.sources {
        let _ = writeln!(out, "\nsource {} ({})", src.name, src.description);
        if src.keys.is_empty() {
            let _ = writeln!(out, "  no interface found");
        }
        for k in &src.keys {
            render_key_text(&mut out, k);
        }
    }
    let t = &s.totals;
    let _ = writeln!(
        out,
        "\ntotals: {} keys, {} source calls, {} slots, {} confirmed sites, {} unconfirmed sites, \
         {} functions",
        t.keys, t.source_calls, t.slots, t.sites, t.unconfirmed_sites, t.functions
    );
    let _ = writeln!(out, "unresolved uses: {}", counts(&s.unresolved));
    out
}

fn render_key_text(out: &mut String, k: &KeyReport) {
    let _ = writeln!(
        out,
        "  key {}: {} source calls in {} functions (out: stack {}, global {}, ret {}){}",
        key_label(&k.key_args),
        k.source_calls,
        k.source_functions,
        k.out_stack,
        k.out_global,
        k.out_ret,
        if k.globals.is_empty() {
            String::new()
        } else {
            format!("; globals {}", hex_list(&k.globals))
        }
    );
    if !k.slots.is_empty() {
        let _ = writeln!(
            out,
            "    {:>5} {:>6} {:>7} {:>6} {:>8} {:>7}  {:<14} {:<6}  examples",
            "slot", "sites", "unconf", "funcs", "callers", "args", "evidence", "conf"
        );
    }
    for sl in &k.slots {
        let args = if sl.min_args == sl.max_args {
            format!(">={}", sl.min_args)
        } else {
            format!(">={}..{}", sl.min_args, sl.max_args)
        };
        let mut ex = hex_list(&sl.examples);
        if !sl.unconfirmed_examples.is_empty() {
            let _ = write!(ex, " unconfirmed:{}", hex_list(&sl.unconfirmed_examples));
        }
        let _ = writeln!(
            out,
            "    {:>5} {:>6} {:>7} {:>6} {:>8} {:>7}  {:<14} {:<6}  {}",
            sl.slot,
            sl.sites,
            sl.unconfirmed_sites,
            sl.functions,
            sl.caller_sites,
            args,
            evidence_str(sl.evidence),
            sl.confidence.as_str(),
            ex
        );
    }
    if !k.unresolved.is_empty() {
        let _ = writeln!(out, "    unresolved: {}", counts(&k.unresolved));
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_hex_list(values: &[u64]) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("\"0x{v:x}\"")).collect();
    format!("[{}]", items.join(", "))
}

fn json_counts(map: &BTreeMap<&'static str, u32>) -> String {
    let items: Vec<String> = map
        .iter()
        .map(|(k, v)| format!("{}: {v}", json_str(k)))
        .collect();
    format!("{{{}}}", items.join(", "))
}

/// Pretty-printed JSON with a trailing newline.
pub fn render_json(s: &Survey) -> String {
    let mut o = String::new();
    o.push_str("{\n");
    let st = &s.stats;
    let _ = writeln!(
        o,
        "  \"stats\": {{\"functions_discovered\": {}, \"functions_analyzed\": {}, \"contexts\": {}, \
         \"rounds\": {}, \"aborted\": {}, \"context_limit_hit\": {}}},",
        st.functions_discovered,
        st.functions_analyzed,
        st.contexts,
        st.rounds,
        st.aborted,
        st.context_limit_hit
    );
    let _ = writeln!(o, "  \"guard_slots\": {},", json_hex_list(&s.guard_slots));
    let t = &s.totals;
    let _ = writeln!(
        o,
        "  \"totals\": {{\"keys\": {}, \"source_calls\": {}, \"slots\": {}, \"sites\": {}, \
         \"unconfirmed_sites\": {}, \"functions\": {}}},",
        t.keys, t.source_calls, t.slots, t.sites, t.unconfirmed_sites, t.functions
    );
    let _ = writeln!(o, "  \"unresolved\": {},", json_counts(&s.unresolved));
    o.push_str("  \"sources\": [");
    for (i, src) in s.sources.iter().enumerate() {
        o.push_str(if i == 0 { "\n" } else { ",\n" });
        let _ = write!(
            o,
            "    {{\"name\": {}, \"description\": {}, \"keys\": [",
            json_str(&src.name),
            json_str(&src.description)
        );
        for (j, k) in src.keys.iter().enumerate() {
            o.push_str(if j == 0 { "\n" } else { ",\n" });
            let key: Vec<String> = k
                .key_args
                .iter()
                .map(|a| json_str(&a.to_string()))
                .collect();
            let _ = write!(
                o,
                "      {{\"key\": [{}], \"source_calls\": {}, \"source_functions\": {}, \
                 \"out\": {{\"stack\": {}, \"global\": {}, \"ret\": {}}}, \"globals\": {}, \
                 \"unresolved\": {}, \"slots\": [",
                key.join(", "),
                k.source_calls,
                k.source_functions,
                k.out_stack,
                k.out_global,
                k.out_ret,
                json_hex_list(&k.globals),
                json_counts(&k.unresolved)
            );
            for (n, sl) in k.slots.iter().enumerate() {
                let _ = write!(
                    o,
                    "{}        {{\"slot\": {}, \"sites\": {}, \"unconfirmed_sites\": {}, \
                     \"functions\": {}, \"caller_sites\": {}, \"min_args\": {}, \"max_args\": {}, \
                     \"evidence\": {}, \"confidence\": {}, \"examples\": {}, \
                     \"unconfirmed_examples\": {}}}",
                    if n == 0 { "\n" } else { ",\n" },
                    sl.slot,
                    sl.sites,
                    sl.unconfirmed_sites,
                    sl.functions,
                    sl.caller_sites,
                    sl.min_args,
                    sl.max_args,
                    json_str(&evidence_str(sl.evidence)),
                    json_str(sl.confidence.as_str()),
                    json_hex_list(&sl.examples),
                    json_hex_list(&sl.unconfirmed_examples)
                );
            }
            o.push_str(if k.slots.is_empty() {
                "]}"
            } else {
                "\n      ]}"
            });
        }
        o.push_str(if src.keys.is_empty() {
            "]}"
        } else {
            "\n    ]}"
        });
    }
    o.push_str(if s.sources.is_empty() {
        "]\n"
    } else {
        "\n  ]\n"
    });
    o.push_str("}\n");
    o
}
