const CACHE_NAME = 'media-converter-shell-v1';
const APP_SHELL = [
    '/',
    '/manifest.webmanifest',
    '/static/icon-192.png',
    '/static/icon-512.png',
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(APP_SHELL))
            .then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys()
            .then((keys) => Promise.all(
                keys.filter((key) => key !== CACHE_NAME)
                    .map((key) => caches.delete(key))
            ))
            .then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
    if (event.request.method !== 'GET'
            || new URL(event.request.url).origin !== self.location.origin) {
        return;
    }

    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request)
                .then((response) => {
                    const copy = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put('/', copy));
                    return response;
                })
                .catch(() => caches.match('/'))
        );
        return;
    }

    if (new URL(event.request.url).pathname.startsWith('/static/')) {
        event.respondWith(
            caches.match(event.request)
                .then((cached) => cached || fetch(event.request))
        );
    }
});
