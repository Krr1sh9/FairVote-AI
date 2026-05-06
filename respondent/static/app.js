"use strict";

    // ── State ──
    let config = null;
    let selectedOption = null;

    // Client-side duplicate guard. This is casual duplicate prevention only:
    // users can clear localStorage, use another browser/device, or edit storage.
    // It is NOT secure voter authentication and must not be described as
    // one-person-one-vote for a real election. It stores no true answer.
    const SUBMISSION_STORAGE_PREFIX = "fairvote.submitted.";

    function safeLocalStorageGet(key) {
      try {
        return localStorage.getItem(key);
      } catch (_err) {
        return null;
      }
    }

    // Production client intentionally has no raw-answer debug mode.  Earlier
    // prototypes supported ?audit=1, but that exposed selected/perturbed values
    // to screenshots and shared devices.  Keep mechanism explanations only.

    // ── DOM refs ──
    const loadingEl     = document.getElementById("loading-state");
    const formEl        = document.getElementById("poll-form");
    const questionEl    = document.getElementById("question-text");
    const gridEl        = document.getElementById("options-grid");
    const demoCard      = document.getElementById("demographics-card");
    const demoFieldsEl  = document.getElementById("demographics-fields");
    const submitBtn     = document.getElementById("submit-btn");
    const statusEl      = document.getElementById("status-msg");
    const resultEl      = document.getElementById("result-state");
    const sentValueEl   = document.getElementById("sent-value");
    const privacyExplEl = document.getElementById("privacy-explain");

    // Audit
    const auditToggle   = document.getElementById("audit-toggle");
    const auditContent  = document.getElementById("audit-content");
    const auditEpsEl    = document.getElementById("audit-eps");
    const auditKEl      = document.getElementById("audit-k");
    const auditPkeepEl  = document.getElementById("audit-pkeep");
    const auditPkeep2El = document.getElementById("audit-pkeep2");
    const auditPflipEl  = document.getElementById("audit-pflip");

    const debugCard      = document.getElementById("debug-card");

    function showElement(el) {
      if (el) el.classList.remove("hidden");
    }

    function hideElement(el) {
      if (el) el.classList.add("hidden");
    }

    function setLoadingError(message) {
      loadingEl.textContent = `Failed to load poll: ${message}`;
      loadingEl.classList.add("status", "error");
    }

    // ── Load config ──
    async function loadConfig() {
      try {
        const res = await fetch("/api/config");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        config = await res.json();
        renderPoll();
      } catch (err) {
        setLoadingError(err.message);
      }
    }

    // ── Render ──
    function renderPoll() {
      questionEl.textContent = config.question;

      // Options
      config.options.forEach((label, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "option-btn";
        const marker = document.createElement("span");
        marker.className = "radio-circle";
        const labelNode = document.createTextNode(String(label));
        btn.appendChild(marker);
        btn.appendChild(labelNode);
        btn.dataset.idx = idx;
        btn.addEventListener("click", () => selectOption(idx, btn));
        gridEl.appendChild(btn);
      });

      // Demographics
      if (config.demographic_fields && config.demographic_fields.length > 0) {
        showElement(demoCard);
        config.demographic_fields.forEach(field => {
          const wrapper = document.createElement("div");
          wrapper.className = "demo-field";

          const lbl = document.createElement("label");
          lbl.htmlFor = `demo-${field.name}`;
          lbl.textContent = field.label;
          wrapper.appendChild(lbl);

          const sel = document.createElement("select");
          sel.id = `demo-${field.name}`;
          sel.dataset.name = field.name;
          const emptyOpt = document.createElement("option");
          emptyOpt.value = "";
          emptyOpt.textContent = "— Select —";
          sel.appendChild(emptyOpt);
          field.options.forEach(opt => {
            const o = document.createElement("option");
            o.value = opt;
            o.textContent = opt;
            sel.appendChild(o);
          });
          wrapper.appendChild(sel);
          demoFieldsEl.appendChild(wrapper);
        });
      }

      // Audit panel
      const { pKeep, pFlip } = rrParams(config.epsilon, config.options.length);
      auditEpsEl.textContent  = config.epsilon.toFixed(2);
      auditKEl.textContent    = config.options.length;
      auditPkeepEl.textContent  = (pKeep * 100).toFixed(1) + "%";
      auditPkeep2El.textContent = (pKeep * 100).toFixed(1) + "%";
      auditPflipEl.textContent  = (pFlip * 100).toFixed(2) + "% each";

      // Show form unless this browser has already submitted this poll.
      hideElement(loadingEl);
      const storedSubmission = getStoredSubmission();
      if (storedSubmission) {
        showAlreadySubmitted(storedSubmission);
      } else {
        showElement(formEl);
      }
    }

    function stableHash(str) {
      let h = 2166136261;
      for (let i = 0; i < str.length; i += 1) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
      return (h >>> 0).toString(16);
    }

    function submissionStorageKey() {
      if (!config) return null;
      const pollId = `${config.question}|${JSON.stringify(config.options)}`;
      return SUBMISSION_STORAGE_PREFIX + stableHash(pollId);
    }

    function getStoredSubmission() {
      const key = submissionStorageKey();
      if (!key) return null;
      try {
        const raw = localStorage.getItem(key);
        return raw ? JSON.parse(raw) : null;
      } catch (_err) {
        return null;
      }
    }

    function markSubmitted() {
      const key = submissionStorageKey();
      if (!key) return;
      try {
        localStorage.setItem(key, JSON.stringify({
          submitted_at: new Date().toISOString(),
          poll_hash: key.replace(SUBMISSION_STORAGE_PREFIX, ""),
          guard: "client-localStorage-casual-duplicate-prevention"
        }));
      } catch (_err) {
        // If localStorage is unavailable, submission still succeeds. The guard is
        // a convenience feature, not a security boundary.
      }
    }

    function showAlreadySubmitted(_storedSubmission) {
      hideElement(formEl);
      showElement(resultEl);
      sentValueEl.textContent = "Already submitted from this browser";
      privacyExplEl.textContent =
        "This browser has already submitted this poll, so the form is locked " +
        "to prevent accidental duplicate responses. This is only a localStorage " +
        "convenience guard, not secure voter authentication.";
      hideElement(debugCard);
    }

    function showDebugAudit(_selectedIdx, _perturbedIdx, _eps, _k) {
      // No-op in the production privacy client. The app never displays raw or
      // perturbed per-response values outside the post-submit reported value.
      hideElement(debugCard);
    }

    function selectOption(idx, btnEl) {
      selectedOption = idx;
      gridEl.querySelectorAll(".option-btn").forEach(b => b.classList.remove("selected"));
      btnEl.classList.add("selected");
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit response (with privacy)";
    }

    // ── Audit toggle ──
    auditToggle.addEventListener("click", () => {
      auditToggle.classList.toggle("open");
      auditContent.classList.toggle("visible");
    });

    // ── Submit ──
    submitBtn.addEventListener("click", async () => {
      if (selectedOption === null || !config) return;

      const existingSubmission = getStoredSubmission();
      if (existingSubmission) {
        showAlreadySubmitted(existingSubmission);
        return;
      }

      submitBtn.disabled = true;
      submitBtn.textContent = "Submitting…";
      statusEl.className = "status";
      hideElement(statusEl);

      const k = config.options.length;
      const eps = config.epsilon;

      // Apply RR in the browser before constructing the network payload.
      const perturbedAnswer = karyRR(selectedOption, eps, k);
      showDebugAudit(selectedOption, perturbedAnswer, eps, k);

      // Collect demographics (sent as-is)
      const demographics = {};
      demoFieldsEl.querySelectorAll("select").forEach(sel => {
        if (sel.value) {
          demographics[sel.dataset.name] = sel.value;
        }
      });

      // Build payload with only the perturbed answer and demographics.
      const payload = {
        perturbed_answer: perturbedAnswer,
        demographics: demographics
      };

      try {
        const res = await fetch("/api/respond", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const err = await res.text();
          throw new Error(err || `HTTP ${res.status}`);
        }

        // Mark local browser as submitted only after the server accepts the
        // privatized response. This stores metadata only, not the true answer.
        markSubmitted();

        // Show result
        hideElement(formEl);
        showElement(resultEl);
        sentValueEl.textContent = config.options[perturbedAnswer];

        const wasFlipped = perturbedAnswer !== selectedOption;
        if (wasFlipped) {
          privacyExplEl.textContent =
            "The mechanism flipped your answer to a different option. " +
            "Because the mechanism could have flipped your answer, you have plausible deniability: " +
            "the server cannot determine with certainty whether this was your original choice.";
        } else {
          privacyExplEl.textContent =
            "The mechanism kept your original answer — but the server " +
            "cannot distinguish this from a flip, so you still have plausible deniability.";
        }
      } catch (err) {
        submitBtn.disabled = false;
        submitBtn.textContent = "Submit response (with privacy)";
        statusEl.className = "status error";
        showElement(statusEl);
        statusEl.textContent = `Error: ${err.message}. Please try again.`;
      }
    });

    // ── Init ──
    loadConfig();

    // ── Register Service Worker for PWA ──
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
          .then(registration => {
            console.log('SW registered');
            // Ask the browser to check for the latest service worker so stale
            // cached rr.js/index.html does not hide privacy-mechanism changes.
            registration.update();
          })
          .catch(err => console.log('SW registration failed:', err));
      });
    }
