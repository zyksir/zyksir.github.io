# Authoring guide — Yikai Zhu's Blog (al-folio distill style)

This site was migrated to an al-folio **distill**-style theme. Posts render with a
sticky left-margin table of contents, right-margin sidenotes, MathJax, numbered
equations/figures/tables, collapsible toggles, hover footnotes, syntax-highlighted
code, a bilingual (中文 / English) toggle, and light/dark mode.

## Local preview

The toolchain lives in Homebrew Ruby (production builds in Docker with Ruby 3.2.2):

```bash
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
bundle install          # first time only
bundle exec jekyll serve # http://127.0.0.1:4000
```

## Creating a post

Add `_posts/YYYY-MM-DD-title.md`. Front matter:

```yaml
---
title: "Your Title"
subtitle: "Optional subtitle"
date: 2026-01-01
description: "One-line summary shown on the blog list and in the byline."
tags: ["LLM", "AI Infra"]     # each tag links to /tags/#slug
# toc: false                  # optional: hide the left-margin table of contents
---
```

`layout: post` is applied automatically. The left TOC is generated from your `##`/`###`
headings. The byline shows Author / Published / Tags (tags are clickable).

## Bilingual posts (中文 / English)

Wrap each language in a `div` with `markdown="1"` so Markdown inside still renders.
The navbar toggle switches between them; the TOC follows the visible language.

```html
<div class="lang-zh" markdown="1">
## 中文标题
中文正文……
</div>

<div class="lang-en" markdown="1">
## English Heading
English body…
</div>
```

Untranslated posts need no wrapper — they simply show in whatever language they're written in.
See `_posts/2024-09-27-rope.md` for a complete bilingual example.

## Click-to-expand toggles

```html
<details markdown="1">
<summary>Click to expand</summary>

Hidden content (Markdown, math, and code all work here).
</details>
```

## Right-margin sidenotes

Put a short note right after the paragraph it annotates. On wide screens it floats into
the empty right margin; on narrow screens it becomes an inline note.

```html
<aside markdown="1">A short margin note.</aside>
```

## Hover footnotes

`<d-footnote>The note text — shown on hover, no click needed.</d-footnote>`

## Math + numbered equations

Inline `$…$`, display `$$…$$`. Every display equation is numbered. To reference one,
add a label inside the block and cite it with `\eqref`:

```
$$ E = mc^2 \label{eq:emc} $$

As shown in $\eqref{eq:emc}$, ...   (the "(1)" becomes a clickable link)
```

> Note: in a **bilingual** post the hidden language's equations also count toward the
> numbers, so the second language starts at a higher number. `\eqref` always links
> correctly regardless. Label equations only in the language you cite, or keep numbered
> equations in single-language posts.

## Numbered figures & tables + cross-references

Give a `<figure>` an id starting with `fig:` (figure) or `tbl:` (table). It's numbered
automatically and its `<figcaption>` gets a "Figure N" / "Table N" (图 N / 表 N) label.
Reference it with an **empty** link whose href is the id — the number fills in and clicking
it jumps there:

```html
<figure id="fig:arch">
  <img src="/img/arch.png" alt="Architecture">
  <figcaption>System architecture.</figcaption>
</figure>

<figure id="tbl:bench" markdown="1">

| Impl | Time (ms) |
|------|-----------|
| A    | 14.4      |

<figcaption>Benchmark results.</figcaption>
</figure>

See <a href="#fig:arch"></a> and <a href="#tbl:bench"></a>.
```

## Tags

Any tag in a post's front matter becomes a chip linking to `/tags/#<slug>`, where the
`/tags/` page lists every post under that tag.
