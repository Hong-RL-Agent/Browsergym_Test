const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = Number(process.env.PORT || 9225);
const ROOT = __dirname;
const VENDOR_ROOT = "C:/workspace/RL_Errorsite/RL_Errorsite/site002/node_modules";

const CONTENT_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8"
};

function send(res, status, body, type = "application/json; charset=utf-8") {
  res.writeHead(status, { "Content-Type": type, "Cache-Control": "no-store" });
  res.end(body);
}

function sendFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) return send(res, 404, "Not found", "text/plain; charset=utf-8");
    send(res, 200, data, CONTENT_TYPES[path.extname(filePath)] || "application/octet-stream");
  });
}

function delayedJson(res, status, payload, delayMs) {
  setTimeout(() => send(res, status, JSON.stringify(payload)), delayMs);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname === "/api/health") {
    return send(res, 200, JSON.stringify({ ok: true, site: "dev_react", framework: "react", port: PORT }));
  }
  if (url.pathname === "/api/delayed-state") {
    return delayedJson(res, 200, { status: "complete", message: "Delayed React state update completed" }, 650);
  }
  if (url.pathname === "/api/spinner-task") {
    return delayedJson(res, 200, { status: "complete", rows: ["Alpha", "Beta", "Gamma"] }, 850);
  }
  if (url.pathname === "/api/replace-content") {
    return delayedJson(res, 200, { status: "complete", content: "Final content restored after skeleton and empty container" }, 750);
  }
  if (url.pathname === "/api/validate") {
    return delayedJson(res, 200, { valid: true, message: "Async validation passed" }, 900);
  }
  if (url.pathname === "/api/render-count") {
    return delayedJson(res, 200, { status: "complete", renderCount: 3 }, 450);
  }
  if (url.pathname === "/api/defect-500") {
    return delayedJson(res, 500, { error: "Intentional DEV_REACT API failure" }, 500);
  }
  if (url.pathname === "/api/persist") {
    return delayedJson(res, 200, { saved: false, storedValue: "old-value" }, 400);
  }
  if (url.pathname === "/vendor/react.production.min.js") {
    return sendFile(res, path.join(VENDOR_ROOT, "react/umd/react.production.min.js"));
  }
  if (url.pathname === "/vendor/react-dom.production.min.js") {
    return sendFile(res, path.join(VENDOR_ROOT, "react-dom/umd/react-dom.production.min.js"));
  }
  const publicPath = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
  return sendFile(res, path.join(ROOT, "public", publicPath));
});

server.listen(PORT, () => {
  console.log(`DEV_REACT listening on http://localhost:${PORT}/`);
});
