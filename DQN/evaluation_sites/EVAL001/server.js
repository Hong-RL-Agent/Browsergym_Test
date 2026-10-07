const http = require("http");

const PORT = Number(process.env.PORT || 9231);

function send(res, status, body, type = "text/html; charset=utf-8") {
  res.writeHead(status, { "Content-Type": type, "Cache-Control": "no-store" });
  res.end(body);
}

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EVAL001 React Commerce</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; color: #20242b; }
    header { padding: 16px 24px; border-bottom: 1px solid #d8dde6; display: flex; justify-content: space-between; }
    main { display: grid; grid-template-columns: 260px 1fr; gap: 18px; padding: 20px; }
    aside { border: 1px solid #d8dde6; padding: 14px; height: 420px; overflow: hidden; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 14px; }
    .card { border: 1px solid #cfd6e0; padding: 12px; position: relative; min-height: 180px; }
    .badge { position: absolute; right: 8px; top: 8px; background: #f4d35e; padding: 4px 8px; }
    .occluder { position: absolute; right: 0; bottom: 42px; width: 130px; height: 34px; background: rgba(220, 60, 60, .92); color: white; }
    button, input, select { margin-top: 8px; padding: 8px; }
    .cart { min-height: 38px; border: 1px dashed #778; padding: 8px; }
  </style>
</head>
<body>
  <header><strong>Meridian Market</strong><span id="cart-count">Cart: 0</span></header>
  <main>
    <aside>
      <h2>Filters</h2>
      <input id="search" placeholder="Search products">
      <button id="search-btn" onclick="searchProducts()">Search</button>
      <select id="sort"><option>Newest</option><option>Price low</option></select>
      <button id="apply-sort" onclick="applySort()">Apply sort</button>
      <div id="filter-status">Ready</div>
    </aside>
    <section>
      <div class="cart" id="cart-msg">Cart is empty</div>
      <div class="grid" id="products"></div>
    </section>
  </main>
  <script>
    const products = ["Desk lamp", "Travel mug", "Notebook", "Cable kit", "Plant stand", "Backpack"];
    function render(list = products) {
      document.getElementById("products").innerHTML = list.map((name, index) =>
        '<article class="card" data-product="' + name + '">' +
          '<span class="badge">Sale</span>' +
          '<h3>' + name + '</h3>' +
          '<p>Ships today from aisle ' + (index + 1) + '.</p>' +
          (index === 2 ? '<div class="occluder">Promo covers action</div>' : '') +
          '<button onclick="addToCart(' + index + ')">Add to cart</button>' +
          '<button onclick="location.href=\\'/product/' + index + '\\'">Details</button>' +
        '</article>').join("");
    }
    function addToCart(index) {
      if (index === 1) return;
      document.getElementById("cart-msg").textContent = "Added " + products[index];
      document.getElementById("cart-count").textContent = "Cart: 1";
    }
    function searchProducts() {
      const q = document.getElementById("search").value.toLowerCase();
      fetch("/api/search?q=" + encodeURIComponent(q)).then(r => r.json()).then(data => {
        render(data.results);
        document.getElementById("filter-status").textContent = data.results.length + " results";
      });
    }
    function applySort() {
      fetch("/api/sort").then(r => r.json()).then(data => {
        render(data.items);
        document.getElementById("filter-status").textContent = data.label;
      });
    }
    render();
  </script>
</body>
</html>`;

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname === "/api/health") return send(res, 200, JSON.stringify({ ok: true, site: "EVAL001", port: PORT }), "application/json; charset=utf-8");
  if (url.pathname === "/api/search") return send(res, 200, JSON.stringify({ results: ["Travel mug", "Cable kit"], query: url.searchParams.get("q") }), "application/json; charset=utf-8");
  if (url.pathname === "/api/sort") return send(res, 200, JSON.stringify({ items: ["Backpack", "Plant stand", "Cable kit"], label: "Newest applied" }), "application/json; charset=utf-8");
  if (url.pathname.startsWith("/product/")) return send(res, 200, html);
  return send(res, 200, html);
});

server.listen(PORT, () => console.log(`EVAL001 listening on http://localhost:${PORT}/`));
