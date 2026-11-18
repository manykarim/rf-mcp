# official-skills-channel Specification

## Purpose
Lets users choose whether rf-mcp serves the official Robot Framework skills bundled in the `robotframework-plugins` package or a newer snapshot of a branch, tag or commit of MarketSquare/robotframework-plugins, and keeps such a snapshot current without network access at startup.

## Requirements

### Requirement: Official skills channel setting
The system SHALL take the official skills channel from `--official-skills` or `RF_MCP_OFFICIAL_SKILLS`, the command-line option winning. The value `bundled` SHALL serve the content of the installed package and is the default. Any other value SHALL be treated as a ref of MarketSquare/robotframework-plugins: a branch, a tag or a full commit SHA.

#### Scenario: Default
- **WHEN** neither `--official-skills` nor `RF_MCP_OFFICIAL_SKILLS` is set
- **THEN** the official skills come from the installed package, and the server makes no network request for them

#### Scenario: Ref from the environment
- **WHEN** `RF_MCP_OFFICIAL_SKILLS=stable` is set and `--official-skills` is not given
- **THEN** the channel is the ref `stable`

#### Scenario: Option overrides the environment
- **WHEN** `RF_MCP_OFFICIAL_SKILLS=stable` is set and the server is started with `--official-skills bundled`
- **THEN** the official skills come from the installed package

### Requirement: Startup serves a cached snapshot or the bundled content
With a ref channel, the server SHALL serve the snapshot it last fetched for that ref when that snapshot is still on disk and complete, and SHALL NOT use the network while starting. Otherwise it SHALL serve the installed package's content and SHALL say in the log and in `rf-mcp doctor` that the ref has no usable snapshot and how to fetch one.

#### Scenario: Snapshot fetched earlier
- **WHEN** the channel is `stable` and an earlier fetch of `stable` stored a complete snapshot of commit `56547f0`
- **THEN** the server serves the skills of that snapshot, with the same layer precedence as the bundled ones, without a network request

#### Scenario: Never fetched
- **WHEN** the channel is `main` and no snapshot of `main` was ever fetched
- **THEN** the server starts with the installed package's skills and logs that `main` has no snapshot yet and that `rf-mcp update-skills --ref main` fetches one

#### Scenario: Snapshot removed from the cache
- **WHEN** the recorded snapshot folder of the channel's ref was deleted
- **THEN** the server serves the installed package's skills and reports the missing snapshot

#### Scenario: Commit SHA as the channel
- **WHEN** the channel is a full commit SHA whose snapshot is in the cache
- **THEN** the server serves that snapshot and never refreshes it, because a commit does not change

### Requirement: Background refresh of a ref channel
With a branch or tag channel, the server SHALL fetch the ref in the background when it was last checked longer ago than the refresh interval, without delaying startup or tool calls. A fetched snapshot SHALL be served from the next start. A failed fetch SHALL be logged once and SHALL leave the served content and the stored snapshot unchanged.

#### Scenario: Refresh due
- **WHEN** the channel is `stable`, it was last checked 30 hours ago and the interval is 24 hours
- **THEN** the server starts with the stored snapshot, fetches `stable` in the background, and a later start serves the new commit

#### Scenario: Refresh not due
- **WHEN** the channel is `stable` and it was last checked 2 hours ago
- **THEN** the server makes no network request for the official skills

#### Scenario: Offline
- **WHEN** a refresh is due and GitHub cannot be reached
- **THEN** the server keeps serving the stored snapshot, or the installed package's content, and logs the failure once

### Requirement: Refresh interval setting
The refresh interval SHALL come from `RF_MCP_OFFICIAL_SKILLS_REFRESH` in hours, default 24. The value `0` SHALL turn the background refresh off, so that only `rf-mcp update-skills` fetches.

#### Scenario: Refresh off
- **WHEN** `RF_MCP_OFFICIAL_SKILLS_REFRESH=0` and the channel is `stable`
- **THEN** the server never fetches in the background and serves the stored snapshot, or the installed package's content

### Requirement: update-skills fetches a ref on demand
`rf-mcp update-skills` SHALL fetch the given ref, else the configured channel, store its snapshot and record it as the ref's current snapshot, and print the ref, commit and snapshot folder. It SHALL exit non-zero when the fetch fails or the channel is `bundled` and no `--ref` is given.

#### Scenario: Seeding an image or an offline machine
- **WHEN** `rf-mcp update-skills --ref stable` runs while building a Docker image
- **THEN** it prints the commit of `stable` and the snapshot folder, and a server in that image with channel `stable` serves the snapshot without network access

#### Scenario: Fetch fails
- **WHEN** `rf-mcp update-skills --ref no-such-branch` runs
- **THEN** it prints why the fetch failed and exits non-zero, and the stored snapshots are unchanged

### Requirement: init writes the official skills channel
`rf-mcp init --official-skills <channel>` SHALL add `RF_MCP_OFFICIAL_SKILLS=<channel>` to the environment of every agent entry it generates, for both project and user scope. Without the option, the entries SHALL not set it.

#### Scenario: User-level entry for the released skills
- **WHEN** a user runs `rf-mcp init --scope user --agent claude --official-skills stable --write`
- **THEN** the registered `rf-mcp` entry carries `RF_MCP_OFFICIAL_SKILLS=stable` in its environment

### Requirement: doctor reports the official skills source
`rf-mcp doctor` SHALL report the channel and what it serves: the installed package's version for `bundled`, or the ref, commit, fetch time and snapshot folder for a ref. When a ref channel falls back to the installed package, it SHALL say so and name `rf-mcp update-skills`. It SHALL NOT use the network.

#### Scenario: Snapshot served
- **WHEN** the channel is `stable` and a snapshot of commit `56547f0` fetched on 2026-11-16 is stored
- **THEN** the report names the channel `stable`, the commit, the fetch time and the number of skills, and the installed package's version as the fallback

#### Scenario: Fallback reported
- **WHEN** the channel is `main` and no snapshot of `main` is stored
- **THEN** the report says the installed package's skills are served instead and names `rf-mcp update-skills --ref main`
