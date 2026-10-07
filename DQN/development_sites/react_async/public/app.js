(function () {
  const e = React.createElement;

  function useAsyncStatus(initial) {
    const [status, setStatus] = React.useState(initial || "Idle");
    const [busy, setBusy] = React.useState(false);
    return { status, setStatus, busy, setBusy };
  }

  function ScenarioCard(props) {
    return e("section", { className: "scenario", "data-scenario-id": props.id, "data-scenario-type": props.kind },
      e("h2", null, `${props.id} ${props.title}`),
      props.children
    );
  }

  function DelayedState() {
    const s = useAsyncStatus("Ready");
    async function run() {
      window.__devEvents.push({ scenario: "R-N01", type: "action", at: Date.now() });
      s.setStatus("Loading after click");
      s.setBusy(true);
      const res = await fetch("/api/delayed-state");
      const data = await res.json();
      s.setStatus(data.message);
      s.setBusy(false);
      window.__scenarioState["R-N01"] = "SUCCESS";
    }
    return e(ScenarioCard, { id: "R-N01", title: "Delayed state update", kind: "NORMAL" },
      e("button", { "data-testid": "rn01-trigger", "data-scenario-id": "R-N01", onClick: run, disabled: s.busy }, "Run delayed update"),
      e("div", { className: "status", "data-testid": "rn01-status" }, s.status)
    );
  }

  function LoadingSpinner() {
    const s = useAsyncStatus("Rows not loaded");
    async function run() {
      window.__devEvents.push({ scenario: "R-N02", type: "action", at: Date.now() });
      s.setBusy(true);
      s.setStatus("Spinner active");
      const res = await fetch("/api/spinner-task");
      const data = await res.json();
      s.setStatus(`Loaded ${data.rows.length} rows`);
      s.setBusy(false);
      window.__scenarioState["R-N02"] = "SUCCESS";
    }
    return e(ScenarioCard, { id: "R-N02", title: "Loading spinner", kind: "NORMAL" },
      e("button", { "data-testid": "rn02-trigger", "data-scenario-id": "R-N02", onClick: run, disabled: s.busy }, s.busy ? e("span", null, e("span", { className: "spinner" }), "Loading") : "Load rows"),
      e("div", { className: "status", "data-testid": "rn02-status" }, s.status)
    );
  }

  function TemporaryReplacement() {
    const [phase, setPhase] = React.useState("content");
    async function run() {
      window.__devEvents.push({ scenario: "R-N03", type: "action", at: Date.now() });
      setPhase("skeleton");
      setTimeout(() => setPhase("empty"), 220);
      const res = await fetch("/api/replace-content");
      const data = await res.json();
      setPhase(data.content);
      window.__scenarioState["R-N03"] = "SUCCESS";
    }
    let body = e("p", { "data-testid": "rn03-final" }, "Stable content is present");
    if (phase === "skeleton") body = e("div", null, e("div", { className: "skeleton" }), e("div", { className: "skeleton" }));
    else if (phase === "empty") body = e("div", { className: "empty-stage", "data-testid": "rn03-empty" });
    else if (phase !== "content") body = e("p", { className: "ok", "data-testid": "rn03-final" }, phase);
    return e(ScenarioCard, { id: "R-N03", title: "Temporary DOM replacement", kind: "NORMAL" },
      e("button", { "data-testid": "rn03-trigger", "data-scenario-id": "R-N03", onClick: run }, "Refresh content"),
      e("div", { className: "status" }, body)
    );
  }

  function AsyncValidation() {
    const s = useAsyncStatus("Submit waits for async validation");
    async function run() {
      window.__devEvents.push({ scenario: "R-N04", type: "action", at: Date.now() });
      s.setBusy(true);
      s.setStatus("Validating");
      const res = await fetch("/api/validate");
      const data = await res.json();
      s.setStatus(data.message);
      s.setBusy(false);
      window.__scenarioState["R-N04"] = "SUCCESS";
    }
    return e(ScenarioCard, { id: "R-N04", title: "Async form validation", kind: "NORMAL" },
      e("input", { "data-testid": "rn04-input", "aria-label": "Async validation email", defaultValue: "valid@example.com" }),
      e("button", { "data-testid": "rn04-trigger", "data-scenario-id": "R-N04", onClick: run, disabled: s.busy }, "Validate"),
      e("div", { className: "status" }, s.status)
    );
  }

  function MultiRender() {
    const [count, setCount] = React.useState(0);
    async function run() {
      window.__devEvents.push({ scenario: "R-N05", type: "action", at: Date.now() });
      setCount((x) => x + 1);
      setTimeout(() => setCount((x) => x + 1), 120);
      await fetch("/api/render-count");
      setCount((x) => x + 1);
      window.__scenarioState["R-N05"] = "SUCCESS";
    }
    return e(ScenarioCard, { id: "R-N05", title: "Re-render without failure", kind: "NORMAL" },
      e("button", { "data-testid": "rn05-trigger", "data-scenario-id": "R-N05", onClick: run }, "Update state"),
      e("div", { className: "status" }, `Render-safe count ${count}`)
    );
  }

  function Defects() {
    async function apiFailure() {
      window.__scenarioState["R-D04"] = "DEFECT_TRIGGERED";
      const res = await fetch("/api/defect-500");
      document.querySelector("[data-testid='rd04-status']").textContent = res.ok ? "Unexpected success" : "Visible API failure";
    }
    async function persistence() {
      const res = await fetch("/api/persist");
      const data = await res.json();
      document.querySelector("[data-testid='rd05-status']").textContent = data.saved ? "Saved" : "Persistence mismatch: old value restored";
      window.__scenarioState["R-D05"] = "DEFECT_TRIGGERED";
    }
    return [
      e(ScenarioCard, { key: "rd01", id: "R-D01", title: "Button no response", kind: "DEFECT" },
        e("button", { "data-testid": "rd01-trigger", "data-bug-id": "dev-react-rd01", "data-scenario-id": "R-D01", onClick: function () {} }, "Broken action"),
        e("div", { className: "status" }, "No response will be rendered")
      ),
      e(ScenarioCard, { key: "rd02", id: "R-D02", title: "Layout overlap", kind: "DEFECT" },
        e("div", { className: "defect-overlap", "data-bug-id": "dev-react-rd02", "data-scenario-id": "R-D02" }, e("span", { className: "badge" }, "Badge"), e("span", { className: "label" }, "Overlapping label"))
      ),
      e(ScenarioCard, { key: "rd03", id: "R-D03", title: "Validation failure", kind: "DEFECT" },
        e("input", { "data-testid": "rd03-input", "aria-label": "Defect validation email", "data-bug-id": "dev-react-rd03", defaultValue: "bad-email" }),
        e("button", { "data-testid": "rd03-trigger", "data-scenario-id": "R-D03", onClick: function () { window.__scenarioState["R-D03"] = "DEFECT_TRIGGERED"; } }, "Submit invalid"),
        e("div", { className: "status" }, "Required validation message never appears")
      ),
      e(ScenarioCard, { key: "rd04", id: "R-D04", title: "API 500 + UI failure", kind: "DEFECT" },
        e("button", { "data-testid": "rd04-trigger", "data-bug-id": "dev-react-rd04", "data-scenario-id": "R-D04", onClick: apiFailure }, "Call failing API"),
        e("div", { className: "status bad", "data-testid": "rd04-status" }, "Waiting")
      ),
      e(ScenarioCard, { key: "rd05", id: "R-D05", title: "Persistence mismatch", kind: "DEFECT" },
        e("button", { "data-testid": "rd05-trigger", "data-bug-id": "dev-react-rd05", "data-scenario-id": "R-D05", onClick: persistence }, "Save value"),
        e("div", { className: "status bad", "data-testid": "rd05-status" }, "Not saved")
      )
    ];
  }

  function App() {
    window.__scenarioState = window.__scenarioState || {};
    window.__devEvents = window.__devEvents || [];
    return e("main", { className: "app", "data-testid": "dev-react-root" },
      e("header", null, e("h1", null, "DEV_REACT Async Testbed"), e("span", null, "React 18.2.0 production-like")),
      e("div", { className: "grid" }, e(DelayedState), e(LoadingSpinner), e(TemporaryReplacement), e(AsyncValidation), e(MultiRender), e(Defects))
    );
  }

  ReactDOM.createRoot(document.getElementById("root")).render(e(App));
})();
