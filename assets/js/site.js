// Theme (light/dark) + language (zh/en) toggles.
(function () {
  var root = document.documentElement;

  // --- Theme -----------------------------------------------------------------
  function syncThemeIcon() {
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    var dark = root.getAttribute('data-theme') === 'dark';
    btn.innerHTML = dark
      ? '<i class="fa-solid fa-sun"></i>'
      : '<i class="fa-solid fa-moon"></i>';
  }
  var themeBtn = document.getElementById('theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      syncThemeIcon();
    });
  }
  syncThemeIcon();

  // --- Table of contents (client-side, language-aware) -----------------------
  // Built from the currently-visible headings inside <d-article>, so bilingual
  // posts only ever show the TOC for the language on screen.
  function slugify(text) {
    return text.trim().toLowerCase()
      .replace(/[\s]+/g, '-')
      .replace(/[^\w一-龥\-]/g, '');
  }
  function buildTOC() {
    var list = document.getElementById('toc-list');
    var article = document.querySelector('d-article');
    if (!list || !article) return;
    var headings = article.querySelectorAll('h2, h3');
    list.innerHTML = '';
    var count = 0;
    headings.forEach(function (h) {
      // Skip the TOC nav's own heading and anything inside d-contents.
      if (h.closest('d-contents')) return;
      // offsetParent is null when the heading (or a parent) is display:none,
      // i.e. it belongs to the hidden language — skip it.
      if (h.offsetParent === null) return;
      if (!h.id) h.id = slugify(h.textContent) || ('sec-' + count);
      var li = document.createElement('li');
      li.className = 'toc-entry toc-' + h.tagName.toLowerCase();
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      list.appendChild(li);
      count++;
    });
    var container = document.querySelector('d-contents');
    if (container) container.style.display = count ? '' : 'none';
  }

  // --- Numbered figures / tables + cross-references --------------------------
  // Author a figure or table as:
  //   <figure id="fig:demo"><img src="..."><figcaption>Caption.</figcaption></figure>
  //   <figure id="tbl:demo" markdown="1"> | a | b | ... <figcaption>Caption.</figcaption></figure>
  // Reference it with an EMPTY link whose href is the id; the number is filled in
  // and clicking it jumps to the figure/table:
  //   see <a href="#fig:demo"></a>
  // (Equations are numbered by MathJax via \label / \eqref.)
  var TYPE = {
    fig: { zh: '图', en: 'Figure' },
    tbl: { zh: '表', en: 'Table' }
  };
  function labelHTML(type, n) {
    return '<span class="lang-zh">' + TYPE[type].zh + ' ' + n + '</span>' +
           '<span class="lang-en">' + TYPE[type].en + ' ' + n + '</span>';
  }
  function numberRefs() {
    var article = document.querySelector('d-article');
    if (!article) return;
    var counters = { fig: 0, tbl: 0 };
    var labels = {};
    article.querySelectorAll('figure[id]').forEach(function (fig) {
      var type = fig.id.split(':')[0];
      if (!TYPE[type]) return;
      counters[type]++;
      labels[fig.id] = counters[type];
      var cap = fig.querySelector('figcaption');
      if (cap && !cap.dataset.numbered) {
        cap.dataset.numbered = '1';
        var span = document.createElement('span');
        span.className = 'fig-label';
        span.innerHTML = labelHTML(type, counters[type]) + '. ';
        cap.prepend(span);
      }
    });
    article.querySelectorAll('a[href^="#fig:"], a[href^="#tbl:"]').forEach(function (a) {
      var id = a.getAttribute('href').slice(1);
      if (labels[id] && !a.textContent.trim()) {
        var type = id.split(':')[0];
        a.innerHTML = labelHTML(type, labels[id]);
        a.classList.add('xref');
      }
    });
  }

  // --- Language --------------------------------------------------------------
  var langBtn = document.getElementById('lang-toggle');
  if (langBtn) {
    langBtn.addEventListener('click', function () {
      var next = root.getAttribute('data-lang') === 'zh' ? 'en' : 'zh';
      root.setAttribute('data-lang', next);
      localStorage.setItem('lang', next);
      buildTOC();
    });
  }

  // Build the TOC + number figures/tables once the distill article has rendered.
  if (document.querySelector('d-article')) {
    window.addEventListener('load', function () { buildTOC(); numberRefs(); });
    setTimeout(function () { buildTOC(); numberRefs(); }, 300); // fallback for late upgrades
  }
})();
