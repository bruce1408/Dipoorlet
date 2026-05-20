<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **Dipoorlet** (2436 symbols, 3387 relationships, 74 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/Dipoorlet/context` | Codebase overview, check index freshness |
| `gitnexus://repo/Dipoorlet/clusters` | All functional areas |
| `gitnexus://repo/Dipoorlet/processes` | All execution flows |
| `gitnexus://repo/Dipoorlet/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |
| Work in the Dipoorlet area (55 symbols) | `.claude/skills/generated/dipoorlet/SKILL.md` |
| Work in the Weight_transform area (42 symbols) | `.claude/skills/generated/weight-transform/SKILL.md` |
| Work in the 2_mobile_v2_dipoorlet_trt area (25 symbols) | `.claude/skills/generated/2-mobile-v2-dipoorlet-trt/SKILL.md` |
| Work in the Deploy area (24 symbols) | `.claude/skills/generated/deploy/SKILL.md` |
| Work in the 3_1_resnet18_dipoorlet_trt area (20 symbols) | `.claude/skills/generated/3-1-resnet18-dipoorlet-trt/SKILL.md` |
| Work in the Dipoorlet_utils area (16 symbols) | `.claude/skills/generated/dipoorlet-utils/SKILL.md` |
| Work in the 3_4_resnet18_dipoorlet_qnn area (14 symbols) | `.claude/skills/generated/3-4-resnet18-dipoorlet-qnn/SKILL.md` |
| Work in the 3_3_resnet18_native_int8_qnn area (10 symbols) | `.claude/skills/generated/3-3-resnet18-native-int8-qnn/SKILL.md` |
| Work in the 3_5_resnet18_aimet_qnn area (7 symbols) | `.claude/skills/generated/3-5-resnet18-aimet-qnn/SKILL.md` |
| Work in the 5_yolov8_dipoorlet area (7 symbols) | `.claude/skills/generated/5-yolov8-dipoorlet/SKILL.md` |
| Work in the Cluster_87 area (5 symbols) | `.claude/skills/generated/cluster-87/SKILL.md` |
| Work in the 6_yolov8_qnn area (4 symbols) | `.claude/skills/generated/6-yolov8-qnn/SKILL.md` |

<!-- gitnexus:end -->
