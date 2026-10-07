const http = require("http");

const PORT = Number(process.env.PORT || 9233);

function send(res, status, body, type = "text/html; charset=utf-8") {
  res.writeHead(status, { "Content-Type": type, "Cache-Control": "no-store" });
  res.end(body);
}

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EVAL003 React Forum</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; color: #20252c; }
    header { padding: 16px 22px; border-bottom: 1px solid #ccd5dd; }
    main { display: grid; grid-template-columns: 1fr 280px; gap: 18px; padding: 20px; }
    .post { border: 1px solid #cbd3dc; padding: 12px; margin-bottom: 10px; }
    .duplicate { border-color: #c33; }
    aside { border-left: 1px solid #d8dde5; padding-left: 16px; }
    input, button { padding: 8px; margin: 6px 0; }
    .toast { position: fixed; bottom: 14px; right: 14px; border: 1px solid #888; padding: 12px; background: white; }
  </style>
</head>
<body>
  <header><h1>Patch Notes Board</h1></header>
  <main>
    <section>
      <input id="query" placeholder="Search posts">
      <button onclick="search()">Search</button>
      <button onclick="loadMore()">Load more</button>
      <div id="posts"></div>
    </section>
    <aside>
      <h2>New post</h2>
      <input id="title" placeholder="Title">
      <button onclick="saveDraft()">Save draft</button>
      <button onclick="publish()">Publish</button>
      <p id="status">No draft saved</p>
    </aside>
  </main>
  <div class="toast" id="toast">Offline sync ready</div>
  <script>
    const base = ["Release notes", "Bug triage", "Search tips"];
    function render(items) {
      document.getElementById("posts").innerHTML = items.map((item, i) =>
        '<article class="post ' + (i === 1 ? 'duplicate' : '') + '"><h3>' +
        item +
        '</h3><p>Thread preview</p></article>'
      ).join("");
    }
    function search() {
      fetch("/api/search?q=" + encodeURIComponent(document.getElementById("query").value))
        .then(r => r.json()).then(data => render(data.items));
    }
    function loadMore() { render(base.concat(base)); }
    function saveDraft() {
      localStorage.setItem("draft-title", document.getElementById("title").value);
      document.getElementById("status").textContent = "Draft saved";
    }
    function publish() {
      fetch("/api/publish", { method: "POST" }).then(r => r.json()).then(data => { document.getElementById("status").textContent = data.message; });
    }
    render(base);
  </script>
</body>
</html>`;

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname === "/api/health") return send(res, 200, JSON.stringify({ ok: true, site: "EVAL003", port: PORT }), "application/json; charset=utf-8");
  if (url.pathname === "/api/search") return send(res, 200, JSON.stringify({ items: ["Release notes", "Release notes"], q: url.searchParams.get("q") }), "application/json; charset=utf-8");
  if (url.pathname === "/api/publish") return send(res, 500, JSON.stringify({ message: "Publish failed" }), "application/json; charset=utf-8");
  if (url.pathname === "/api/state") return send(res, 200, JSON.stringify({ saved: false }), "application/json; charset=utf-8");
  return send(res, 200, html);
});

server.listen(PORT, () => console.log(`EVAL003 listening on http://localhost:${PORT}/`));
