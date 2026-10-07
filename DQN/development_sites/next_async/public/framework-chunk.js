(function () {
  const app = document.getElementById("app");
  window.__scenarioState = window.__scenarioState || {};
  window.__devNetwork = window.__devNetwork || [];
  const originalFetch = window.fetch.bind(window);
  window.fetch = function (url, options) {
    window.__devNetwork.push({ url: String(url), at: Date.now(), options: options || null });
    return originalFetch(url, options);
  };

  function card(id, kind, title, body) {
    return `<section class="scenario" data-scenario-id="${id}" data-scenario-type="${kind}"><h2>${id} ${title}</h2>${body}</section>`;
  }

  function shell(route) {
    app.innerHTML = `
      <h1>DEV_NEXT Async Testbed</h1>
      <nav>
        <a href="/" data-testid="home-link">Home</a>
        <a href="/reports" data-testid="nn02-link" data-scenario-id="N-N02">Reports</a>
        <a href="/settings" data-testid="settings-link">Settings</a>
      </nav>
      <div class="status" data-testid="route-status">Route ${route}</div>
      <div class="grid">
        ${card("N-N01", "NORMAL", "Prefetch", `<button data-testid="nn01-trigger" data-scenario-id="N-N01">Prefetch reports</button><div class="status" data-testid="nn01-status">Prefetch idle</div>`)}
        ${card("N-N02", "NORMAL", "Route navigation", `<button data-testid="nn02-trigger" data-scenario-id="N-N02">Client navigate</button><div class="status" data-testid="nn02-status">Navigation idle</div>`)}
        ${card("N-N03", "NORMAL", "Suspense loading UI", `<button data-testid="nn03-trigger" data-scenario-id="N-N03">Load suspense panel</button><div class="status" data-testid="nn03-status">Panel idle</div>`)}
        ${card("N-N04", "NORMAL", "Progressive rendering", `<button data-testid="nn04-trigger" data-scenario-id="N-N04">Progressively render</button><div class="status" data-testid="nn04-status">Waiting</div>`)}
        ${card("N-N05", "NORMAL", "Background framework network", `<button data-testid="nn05-trigger" data-scenario-id="N-N05">Functional request</button><div class="status" data-testid="nn05-status">Network idle</div>`)}
        ${card("N-D01", "DEFECT", "Navigation failure", `<button data-testid="nd01-trigger" data-bug-id="dev-next-nd01" data-scenario-id="N-D01">Broken navigate</button><div class="status">Route will not change</div>`)}
        ${card("N-D02", "DEFECT", "API 500 + visible failure", `<button data-testid="nd02-trigger" data-bug-id="dev-next-nd02" data-scenario-id="N-D02">Call failing API</button><div class="status bad" data-testid="nd02-status">Waiting</div>`)}
        ${card("N-D03", "DEFECT", "Layout clipping overlap", `<div class="clip-box" data-bug-id="dev-next-nd03" data-scenario-id="N-D03"><span>Clipped text extends beyond the intended layout and cannot be read completely</span></div><div class="overlap"><span>Layer A</span><strong>Layer B</strong></div>`)}
        ${card("N-D04", "DEFECT", "Action no response", `<button data-testid="nd04-trigger" data-bug-id="dev-next-nd04" data-scenario-id="N-D04">No response action</button><div class="status">No state change will occur</div>`)}
        ${card("N-D05", "DEFECT", "Save persistence failure", `<button data-testid="nd05-trigger" data-bug-id="dev-next-nd05" data-scenario-id="N-D05">Save draft</button><div class="status bad" data-testid="nd05-status">Not saved</div>`)}
      </div>`;
    bind();
  }

  function bind() {
    document.querySelector("[data-testid='nn01-trigger']").onclick = async function () {
      const status = document.querySelector("[data-testid='nn01-status']");
      status.textContent = "Prefetching framework data";
      await fetch("/_next/data/prefetch.json", { headers: { "Purpose": "prefetch" } });
      status.textContent = "Prefetch complete without user-visible change";
      window.__scenarioState["N-N01"] = "SUCCESS";
    };
    document.querySelector("[data-testid='nn02-trigger']").onclick = async function () {
      const status = document.querySelector("[data-testid='nn02-status']");
      status.textContent = "Route transition loading";
      history.pushState({}, "", "/reports");
      await fetch("/_next/rsc/reports");
      await fetch("/api/reports");
      status.textContent = "Reports route ready";
      document.querySelector("[data-testid='route-status']").textContent = "Route /reports";
      window.__scenarioState["N-N02"] = "SUCCESS";
    };
    document.querySelector("[data-testid='nn03-trigger']").onclick = async function () {
      const status = document.querySelector("[data-testid='nn03-status']");
      status.innerHTML = '<div class="fallback">Loading fallback</div>';
      await fetch("/api/reports");
      status.textContent = "Suspense content ready";
      window.__scenarioState["N-N03"] = "SUCCESS";
    };
    document.querySelector("[data-testid='nn04-trigger']").onclick = async function () {
      const status = document.querySelector("[data-testid='nn04-status']");
      status.textContent = "First shell visible";
      const a = await fetch("/api/progressive-a").then((r) => r.json());
      status.textContent = `Partial ${a.part}`;
      const b = await fetch("/api/progressive-b").then((r) => r.json());
      status.textContent = `Final progressive content with ${a.part} and ${b.part}`;
      window.__scenarioState["N-N04"] = "SUCCESS";
    };
    document.querySelector("[data-testid='nn05-trigger']").onclick = async function () {
      const status = document.querySelector("[data-testid='nn05-status']");
      status.textContent = "User request and framework background request running";
      fetch("/_next/data/prefetch.json", { headers: { "Purpose": "prefetch" } });
      await fetch("/api/functional");
      status.textContent = "Functional request complete";
      window.__scenarioState["N-N05"] = "SUCCESS";
    };
    document.querySelector("[data-testid='nd01-trigger']").onclick = function () {
      window.__scenarioState["N-D01"] = "DEFECT_TRIGGERED";
    };
    document.querySelector("[data-testid='nd02-trigger']").onclick = async function () {
      const res = await fetch("/api/fail");
      document.querySelector("[data-testid='nd02-status']").textContent = res.ok ? "Unexpected success" : "Visible API failure";
      window.__scenarioState["N-D02"] = "DEFECT_TRIGGERED";
    };
    document.querySelector("[data-testid='nd04-trigger']").onclick = function () {
      window.__scenarioState["N-D04"] = "DEFECT_TRIGGERED";
    };
    document.querySelector("[data-testid='nd05-trigger']").onclick = async function () {
      const data = await fetch("/api/save").then((r) => r.json());
      document.querySelector("[data-testid='nd05-status']").textContent = data.saved ? "Saved" : `Save failed, restored ${data.restored}`;
      window.__scenarioState["N-D05"] = "DEFECT_TRIGGERED";
    };
  }

  window.addEventListener("popstate", function () { shell(location.pathname); });
  shell(location.pathname);
})();
