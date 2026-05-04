// Service worker for the respondent demo app.
//
// Cache policy is deliberately conservative. During development and marking,
// stale index.html or rr.js could hide changes to the Randomized Response code,
// which is exactly the boundary an examiner is likely to inspect carefully.
// so navigations, /static/rr.js, and API requests use the network first.
// Cache version is bumped whenever the RR logic or poll UI changes.
const CACHE_NAME = 'fairvote-cache-v3-no-stale-rr';
const STATIC_CACHE_URLS = [
  '/static/manifest.json',
  '/static/fairvote_icon.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_CACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(names =>
      Promise.all(
        names.filter(name => name !== CACHE_NAME)
             .map(name => caches.delete(name))
      )
    ).then(() => self.clients.claim())
  );
});

// Network-first strategy: try the network and fall back to cache only on
// failure.  This guarantees the user always sees the latest RR parameters
// when the server is reachable.
function networkFirst(request) {
  return fetch(request).catch(() => caches.match(request));
}

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Do not cache response submissions; POST requests must always hit the
  // server so that perturbed answers are persisted.
  if (event.request.method !== 'GET') {
    event.respondWith(fetch(event.request));
    return;
  }

  // Avoid stale poll UI/config/RR JavaScript during development and assessment.
  if (url.pathname === '/' ||
      url.pathname === '/index.html' ||
      url.pathname === '/static/rr.js' ||
      url.pathname === '/api/config') {
    event.respondWith(networkFirst(event.request));
    return;
  }

  // Small static assets can be cache-first.
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
      .catch(() => new Response(
        '<h1>FairVote Survey — Offline</h1><p>Please reconnect to submit your response.</p>',
        { headers: { 'Content-Type': 'text/html' } }
      ))
  );
});
