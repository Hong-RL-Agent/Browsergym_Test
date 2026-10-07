const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = Number(process.env.PORT || 9226);
const ROOT = __dirname;

function send(res, status, body, type = "application/json; charset=utf-8", extra = {}) {
  res.writeHead(status, { "Content-Type": type, "Cache-Control": "no-store", ...extra });
  res.end(body);
}

function file(res, name) {
  fs.readFile(path.join(ROOT, "public", name), (err, data) => {
    if (err) return send(res, 404, "Not found", "text/plain; charset=utf-8");
    const ext = path.extname(name);
    const type = ext === ".css" ? "text/css; charset=utf-8" : ext === ".js" ? "application/javascript; charset=utf-8" : "text/html; charset=utf-8";
    send(res, 200, data, type);
  });
}

function delayed(res, status, payload, ms, headers) {
  setTimeout(() => send(res, status, JSON.stringify(payload), "application/json; charset=utf-8", headers), ms);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname === "/api/health") return send(res, 200, JSON.stringify({ ok: true, site: "dev_next", port: PORT }));
  if (url.pathname === "/_next/static/chunks/app.js") return file(res, "framework-chunk.js");
  if (url.pathname === "/_next/data/prefetch.json") return delayed(res, 200, { route: "/reports", kind: "prefetch" }, 180, { "X-Framework-Request": "prefetch" });
  if (url.pathname === "/_next/rsc/reports") return delayed(res, 200, { segment: "reports", kind: "rsc" }, 260, { "Content-Type": "text/x-component; charset=utf-8", "X-Framework-Request": "rsc" });
  if (url.pathname === "/api/reports") return delayed(res, 200, { report: "Quarterly metrics ready" }, 720);
  if (url.pathname === "/api/progressive-a") return delayed(res, 200, { part: "summary" }, 280);
  if (url.pathname === "/api/progressive-b") return delayed(res, 200, { part: "details" }, 820);
  if (url.pathname === "/api/functional") return delayed(res, 200, { ok: true, source: "user-functional-request" }, 450);
  if (url.pathname === "/api/fail") return delayed(res, 500, { error: "Intentional DEV_NEXT API failure" }, 420);
  if (url.pathname === "/api/save") return delayed(res, 200, { saved: false, restored: "previous draft" }, 360);
  if (url.pathname === "/" || url.pathname === "/reports" || url.pathname === "/settings") return file(res, "index.html");
  return file(res, url.pathname.slice(1));
});

server.listen(PORT, () => {
  console.log(`DEV_NEXT listening on http://localhost:${PORT}/`);
});
