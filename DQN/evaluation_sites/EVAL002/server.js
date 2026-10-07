const http = require("http");

const PORT = Number(process.env.PORT || 9232);

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
  <title>EVAL002 Next Booking</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; color: #1f2933; }
    nav { padding: 14px 22px; border-bottom: 1px solid #ccd5df; display: flex; gap: 14px; }
    main { padding: 22px; max-width: 900px; }
    label { display: block; margin: 12px 0; }
    input, select, button { padding: 8px; margin-left: 8px; }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .summary { border: 1px solid #c9d3df; padding: 12px; min-height: 120px; }
    .clip { width: 180px; height: 34px; overflow: hidden; border: 1px solid #d45; }
  </style>
</head>
<body>
  <nav>
    <a href="/">Booking</a>
    <a href="/account" id="account-link">Account</a>
    <a href="/checkout" id="checkout-link">Checkout</a>
  </nav>
  <main>
    <h1>Harbor Stay Booking</h1>
    <div class="two">
      <form onsubmit="submitBooking(event)">
        <label>Name <input id="name"></label>
        <label>Date <input id="date" type="date"></label>
        <label>Guests <select id="guests"><option>1</option><option>2</option><option>3</option></select></label>
        <button id="reserve">Reserve</button>
        <div id="form-msg"></div>
      </form>
      <section class="summary">
        <h2>Summary</h2>
        <p id="price">$120.00</p>
        <button onclick="calculate()">Recalculate</button>
        <p class="clip">Long cancellation policy text is clipped on small card widths.</p>
      </section>
    </div>
  </main>
  <script>
    document.getElementById("account-link").onclick = event => { event.preventDefault(); location.href = "/checkout"; };
    function submitBooking(event) {
      event.preventDefault();
      const name = document.getElementById("name").value;
      fetch("/api/reserve", { method: "POST", body: JSON.stringify({ name }) })
        .then(r => r.json()).then(data => { document.getElementById("form-msg").textContent = data.message; });
    }
    function calculate() {
      fetch("/api/price").then(r => r.json()).then(data => { document.getElementById("price").textContent = "$" + data.total.toFixed(2); });
    }
  </script>
</body>
</html>`;

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname === "/api/health") return send(res, 200, JSON.stringify({ ok: true, site: "EVAL002", port: PORT }), "application/json; charset=utf-8");
  if (url.pathname === "/api/reserve") return send(res, 200, JSON.stringify({ ok: false, message: "Saved draft only" }), "application/json; charset=utf-8");
  if (url.pathname === "/api/price") return send(res, 200, JSON.stringify({ subtotal: 120, fees: 30, total: 120 }), "application/json; charset=utf-8");
  if (url.pathname === "/api/slow-availability") return delay(4500, () => send(res, 200, JSON.stringify({ available: true }), "application/json; charset=utf-8"));
  return send(res, 200, html);
});

server.listen(PORT, () => console.log(`EVAL002 listening on http://localhost:${PORT}/`));
