"""Node-based checks for browser-side k-ary Randomized Response.

The browser app uses JavaScript for the local privacy mechanism. These tests run
that implementation directly with Node when Node is available.
"""
from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RR_JS = ROOT / "respondent" / "static" / "rr.js"


# Node's real WebCrypto can be slow in tight statistical loops on small CI
# runners. These tests are about the k-ary RR logic, so the statistical tests
# install a deterministic, fast getRandomValues implementation. The production
# browser code still uses crypto.getRandomValues().
DETERMINISTIC_CRYPTO_JS = """
let rngState = 0x12345678;
function nextUint32() {
  rngState = (1664525 * rngState + 1013904223) >>> 0;
  return rngState;
}
Object.defineProperty(globalThis, 'crypto', {
  value: {
    getRandomValues(arr) {
      for (let i = 0; i < arr.length; i++) arr[i] = nextUint32();
      return arr;
    }
  },
  configurable: true
});
"""

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _run_node(script: str) -> str:
    # Run the browser RR implementation directly under Node so the JavaScript
    # privacy mechanism is checked independently of Python wrappers.
    # Force Node to exit after the assertions. In some CI/container builds the
    # WebCrypto runtime can otherwise keep the event loop alive after stdout.
    result = subprocess.run(
        ["node", "-e", script + "\nprocess.exit(0);"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    return result.stdout.strip()


def test_rr_params_match_kary_rr_formula() -> None:
    # Verify that the JavaScript pKeep and pFlip match the analytic k-ary RR
    # formula: pKeep = e^eps / (e^eps + k - 1).  Any discrepancy would mean
    # the browser and server use different privacy channels.
    script = textwrap.dedent(
        f"""
        global.crypto = require('crypto').webcrypto;
        const {{ rrParams }} = require({str(RR_JS)!r});
        const eps = 1.0;
        const k = 6;
        const params = rrParams(eps, k);
        const expectedKeep = Math.exp(eps) / (Math.exp(eps) + k - 1);
        const expectedFlip = 1 / (Math.exp(eps) + k - 1);
        if (Math.abs(params.pKeep - expectedKeep) > 1e-12) throw new Error('bad pKeep');
        if (Math.abs(params.pFlip - expectedFlip) > 1e-12) throw new Error('bad pFlip');
        console.log('ok');
        """
    )
    assert _run_node(script) == "ok"


def test_kary_rr_flips_at_approximately_expected_rate() -> None:
    # Statistical test: over many trials, the observed flip rate should be
    # close to (1 - pKeep).  The tolerance of 3.5% allows for sampling noise.
    script = textwrap.dedent(
        f"""
        global.crypto = require('crypto').webcrypto;
        const {{ rrParams, karyRR }} = require({str(RR_JS)!r});
        const eps = 1.0;
        const k = 6;
        const n = 20000;
        const trueAnswer = 2;
        let flips = 0;
        for (let i = 0; i < n; i++) {{
          if (karyRR(trueAnswer, eps, k) !== trueAnswer) flips++;
        }}
        const observed = flips / n;
        const expected = 1 - rrParams(eps, k).pKeep;
        if (Math.abs(observed - expected) > 0.035) {{
          throw new Error(`observed flip rate ${{observed}} too far from expected ${{expected}}`);
        }}
        console.log(JSON.stringify({{observed, expected}}));
        """
    )
    out = _run_node(script)
    assert "observed" in out


def test_low_epsilon_produces_frequent_flips() -> None:
    # With near-zero epsilon, nearly every report should be a random flip.
    # This guards against an implementation that accidentally passes through
    # the true answer when epsilon is very small.
    script = textwrap.dedent(
        f"""
        global.crypto = require('crypto').webcrypto;
        const {{ karyRR }} = require({str(RR_JS)!r});
        const eps = 0.01;
        const k = 6;
        const n = 12000;
        const trueAnswer = 0;
        let flips = 0;
        for (let i = 0; i < n; i++) {{
          if (karyRR(trueAnswer, eps, k) !== trueAnswer) flips++;
        }}
        const observed = flips / n;
        if (observed < 0.75) {{
          throw new Error(`low epsilon should flip frequently, observed ${{observed}}`);
        }}
        console.log(observed.toFixed(3));
        """
    )
    observed = float(_run_node(script))
    assert observed > 0.75
