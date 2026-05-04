/**
 * rr.js — Client-side k-ary Randomised Response (LDP mechanism)
 *
 * This module implements the k-ary Randomised Response mechanism entirely
 * in the browser. The selected answer is used locally to construct the
 * perturbed report; only the perturbed answer is sent to the server.
 *
 * Uses crypto.getRandomValues() for cryptographically secure randomness.
 */

"use strict";

/**
 * Compute the RR parameters for a given epsilon and k.
 *
 * @param {number} epsilon  Privacy parameter (> 0). Smaller = more privacy.
 * @param {number} k        Number of answer categories (>= 2).
 * @returns {{pKeep: number, pFlip: number}}
 *   pKeep = P(report true answer) = e^eps / (e^eps + k - 1)
 *   pFlip = P(report any other answer) = 1 / (e^eps + k - 1)
 */
function rrParams(epsilon, k) {
  // Match the Python k-ary RR channel so that browser submissions and
  // server-side analysis use the same probability model.
  if (k < 2) throw new Error("k must be >= 2");
  if (epsilon <= 0) throw new Error("epsilon must be > 0");

  const expEps = Math.exp(epsilon);
  const denom = expEps + k - 1;
  const pKeep = expEps / denom;
  const pFlip = 1.0 / denom;

  return { pKeep, pFlip };
}

/**
 * Generate a cryptographically secure random float in [0, 1).
 * Uses crypto.getRandomValues() instead of Math.random().
 *
 * @returns {number} A random float in [0, 1).
 */
function secureRandom() {
  const arr = new Uint32Array(1);
  crypto.getRandomValues(arr);
  return arr[0] / (0xFFFFFFFF + 1);
}

/**
 * Apply k-ary Randomised Response to a true answer.
 *
 * With probability pKeep, the true answer is returned.
 * With probability (1 - pKeep), a uniformly random different category is returned.
 *
 * @param {number} trueAnswer  The respondent's true answer (integer in [0, k-1]).
 * @param {number} epsilon     Privacy parameter (> 0).
 * @param {number} k           Number of answer categories (>= 2).
 * @returns {number} The perturbed answer (integer in [0, k-1]).
 */
function karyRR(trueAnswer, epsilon, k) {
  // Client-side privacy boundary: the returned value is the perturbed answer
  // that should be submitted, not a value to be reversed on the server.
  if (trueAnswer < 0 || trueAnswer >= k) {
    throw new Error(`trueAnswer must be in [0, ${k - 1}], got ${trueAnswer}`);
  }

  const { pKeep } = rrParams(epsilon, k);
  const u = secureRandom();

  if (u < pKeep) {
    // Keep the selected category.
    return trueAnswer;
  } else {
    // Report a uniformly random different category.
    // Here "other" means any category except the selected one; it is not a
    // separate poll option. Pick from [0, k-2], then shift to skip trueAnswer.
    const raw = Math.floor(secureRandom() * (k - 1));
    return raw >= trueAnswer ? raw + 1 : raw;
  }
}

// Export for testing (Node.js) or use as globals (browser). The export is
// only for verification; it does not change the browser privacy flow.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { rrParams, secureRandom, karyRR };
}
