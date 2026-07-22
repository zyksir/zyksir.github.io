# [Yikai Zhu's Blog](https://zyksir.github.io)

An al-folio **distill**-style Jekyll blog. See [`AUTHORING.md`](./AUTHORING.md) for the full
set of features (bilingual toggle, collapsible sections, margin sidenotes, hover footnotes,
numbered equations/figures/tables, tags filter, dark mode).

## Run a local server

The Mac's system Ruby (2.6) is too old, so use the Homebrew Ruby. Put it on your PATH for
the current shell:

```bash
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"

bundle install                 # first time only
bundle exec jekyll serve       # → http://127.0.0.1:4000
```

The site rebuilds automatically as you edit. Stop it with `Ctrl-C` (or `pkill -f jekyll`).

> Deploys are built in CI with Docker (`ruby:3.2.2`) and pushed to the `gh-pages` branch —
> see `Dockerfile` / `entrypoint.sh`. You don't need Ruby 3 locally as long as the Homebrew
> Ruby above works.

## Add a new post

Fastest way — the Rake task scaffolds the file (front matter + a bilingual 中文 / English body):

```bash
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
rake post title="My New Post" subtitle="An optional subtitle"
```

This creates `_posts/YYYY-MM-DD-my-new-post.md`. Then just edit it — fill in `description`
and `tags`, write inside the `.lang-zh` / `.lang-en` blocks (delete one block if the post is
single-language), and preview with `bundle exec jekyll serve`.

Or create the file by hand under `_posts/YYYY-MM-DD-title.md`:

```markdown
---
title: "My New Post"
date: 2026-01-01
description: "One-line summary shown in the byline and lists."
tags: ["LLM", "AI Infra"]
---

Your content here…
```

`layout: post` is applied automatically; the left-margin table of contents is generated from
your `##` / `###` headings. For toggles, sidenotes, footnotes, and numbered
equations/figures/tables, see [`AUTHORING.md`](./AUTHORING.md).
