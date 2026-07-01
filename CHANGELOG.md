---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Release Notes"
sidebar-title: "Release Notes"
description: "Changelog and feature history for PAIDF Augmentation releases."
description-agent: "Includes the PAIDF Augmentation release notes. Use when users ask about recent changes, the release cadence, or where to track versioned assets."
keywords: ["paidf augmentation release notes", "paidf augmentation changelog"]
content:
  type: "reference"
---

PAIDF Augmentation release notes.
Use this page to track the highlights of the latest release.
The latest release appears first.

## v1.0.1

PAIDF Augmentation v1.0.1 adds arm64 build support, makes image-edit generation more robust with a configurable request timeout and automatic retries, improves the PAS (person-attribute) workflow, and makes runs fail loudly on generation failures and undecodable input instead of falsely reporting success.

- Added `arm64`/`aarch64` build support so the images build and run on ARM hosts in addition to `x86_64`.
- Added a configurable per-request timeout (`pipeline.request_timeout`, default 120s) that bounds each generation endpoint call and serves as the base for exponential retry backoff.
- Added automatic retries with exponential backoff (`request_timeout`, then 2×, 4×, …) up to `pipeline.retry` on generation/endpoint failure; the OpenAI SDK's own retries are disabled so `pipeline.retry` is the single retry path.
- Added a dedicated PAS (person-attribute) workflow and expanded augmentation guidance, making person-attribute image editing easier to configure and run at scale.
- Failed samples now log the failed input and exit non-zero instead of falsely reporting success — the run no longer calls `tracker.complete()` when a sample exhausts its retries.
- Fail fast on undecodable input videos in the local `Cosmos` executor instead of failing later.

## v1.0.0

Initial release of PAIDF Augmentation — the media augmentation pipeline (captioning, generation, evaluation) with support for `cosmos-transfer2.5`, `cosmos-predict`, and `image-edit` models.
