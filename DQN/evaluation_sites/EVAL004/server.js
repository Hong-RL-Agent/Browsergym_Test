const http = require("http");

const PORT = Number(process.env.PORT || 9234);

function send(res, status, body, type = "text/html; charset=utf-8") {
  res.writeHead(status, { "Content-Type": type, "Cache-Control": "no-store" });
  res.end(body);
}

function delay(ms, fn) {
  setTimeout(fn, ms);
}

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EVAL004 Next Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; color: #1e252c; }
    header { padding: 16px 24px; border-bottom: 1px solid #ccd5df; display: flex; justify-content: space-between; }
    main { padding: 20px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .panel { border: 1px solid #ccd5df; padding: 12px; min-height: 150px; }
    .wide { grid-column: span 2; }
    button, select { padding: 8px; margin: 6px 0; }
    .overflow { width: 160px; white-space: nowrap; overflow: hidden; border: 1px solid #c65; }
  </style>
</head>
<body>
  <header><strong>Ops Beacon</strong><button onclick="refresh()">Refresh</button></header>
  <main>
    <section class="panel"><h2>Revenue</h2><p id="revenue">$480</p><button onclick="calc()">Recalculate</button></section>
    <section class="panel"><h2>Network</h2><button onclick="network()">Check</button><p id="network">Idle</p></section>
    <section class="panel"><h2>Export</h2><button onclick="exportCsv()">Export CSV</button><p id="export">Ready</p></section>
    <section class="panel wide"><h2>Alerts</h2><p class="overflow">Critical alert text is clipped inside this panel at default width.</p></section>
    <section class="panel"><h2>Region</h2><select id="region"><option>APAC</option><option>EU</option></select><button onclick="saveRegion()">Save</button></section>
  </main>
  <script>
    function refresh() { fetch("/api/widgets").then(r => r.json()).then(data => { document.getElementById("network").textContent = data.status; }); }
    function calc() { fetch("/api/revenue").then(r => r.json()).then(data => { document.getElementById("revenue").textContent = "$" + data.total; }); }
    function network() { fetch("/api/network").then(r => r.json()).then(data => { document.getElementById("network").textContent = data.message; }).catch(() => { document.getElementById("network").textContent = "Network failed"; }); }
    function exportCsv() { fetch("/api/export").then(r => r.text()).then(text => { document.getElementById("export").textContent = text.slice(0, 20); }); }
    function saveRegion() { document.getElementById("export").textContent = "Region not saved"; }
  </script>
</body>
</html>`;

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname === "/api/health") return send(res, 200, JSON.stringify({ ok: true, site: "EVAL004", port: PORT }), "application/json; charset=utf-8");
  if (url.pathname === "/api/widgets") return send(res, 200, JSON.stringify({ status: "Loaded duplicate widget payload" }), "application/json; charset=utf-8");
  if (url.pathname === "/api/revenue") return send(res, 200, JSON.stringify({ total: 480, expected: 530 }), "application/json; charset=utf-8");
  if (url.pathname === "/api/network") return send(res, 503, JSON.stringify({ message: "Transport unavailable" }), "application/json; charset=utf-8");
  if (url.pathname === "/api/export") return delay(5000, () => send(res, 200, "date,value\n2026-09-16,10", "text/csv; charset=utf-8"));
  return send(res, 200, html);
});

server.listen(PORT, () => console.log(`EVAL004 listening on http://localhost:${PORT}/`));
