---
name: OpenSpec Change Proposal
about: Propose a spec-driven change using OpenSpec
title: "add-<short-feature-name>"
labels: enhancement
assignees: ''
---

## Why
Describe the problem and motivation.

## What Changes
- High-level bullets of what will change

## Impact
- Dependencies, scripts, data, risks

## Tasks
- [ ] Create OpenSpec change folder under `openspec/changes/<change-id>/`
- [ ] Draft `proposal.md` (Why/What/Impact/Out of Scope)
- [ ] Add spec deltas in `specs/<capability>/spec.md` with `## ADDED|MODIFIED|REMOVED` and `#### Scenario:` blocks
- [ ] Add `tasks.md` with checklist
- [ ] `openspec validate <change-id> --strict`

## Links
- Related specs/changes: 
- References:

