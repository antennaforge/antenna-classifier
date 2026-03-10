/* ============================================================
   platform-nav.js — AntennaForge Top Navigation Module
   Phase A Foundation · 2026-03-07

   Two modes:
   1. Hamfeeds (base.html): Nav HTML is server-rendered by Jinja.
      This script wires up hamburger toggle + user dropdown.
   2. Classifier (standalone): Call renderPlatformNav() to
      dynamically build the nav from /api/me data.
   ============================================================ */

(function () {
    'use strict';

    /* ---------- 1. Wire up server-rendered nav (hamfeeds) ---------- */

    function initNav() {
        var hamburger = document.querySelector('.platform-nav__hamburger');
        var overlay = document.querySelector('.platform-nav__overlay');
        var userBtn = document.querySelector('.platform-nav__user-btn');
        var dropdown = document.querySelector('.platform-nav__dropdown');

        // Hamburger toggle
        if (hamburger && overlay) {
            hamburger.addEventListener('click', function (e) {
                e.stopPropagation();
                var isOpen = overlay.classList.toggle('open');
                hamburger.setAttribute('aria-expanded', isOpen);
            });
        }

        // User dropdown toggle
        if (userBtn && dropdown) {
            userBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                var isOpen = dropdown.classList.toggle('open');
                userBtn.setAttribute('aria-expanded', isOpen);
            });
        }

        // Close dropdown + overlay on outside click
        document.addEventListener('click', function (e) {
            if (dropdown && dropdown.classList.contains('open')) {
                if (!dropdown.contains(e.target) && !userBtn.contains(e.target)) {
                    dropdown.classList.remove('open');
                    if (userBtn) userBtn.setAttribute('aria-expanded', 'false');
                }
            }
            if (overlay && overlay.classList.contains('open')) {
                if (!overlay.contains(e.target) && hamburger && !hamburger.contains(e.target)) {
                    overlay.classList.remove('open');
                    if (hamburger) hamburger.setAttribute('aria-expanded', 'false');
                }
            }
        });

        // Close on Escape
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                if (dropdown) {
                    dropdown.classList.remove('open');
                    if (userBtn) userBtn.setAttribute('aria-expanded', 'false');
                }
                if (overlay) {
                    overlay.classList.remove('open');
                    if (hamburger) hamburger.setAttribute('aria-expanded', 'false');
                }
            }
        });
    }

    /* ---------- 2. Dynamic nav renderer (for classifier) ---------- */

    /**
     * Render the platform nav bar dynamically.
     * Call from the classifier: renderPlatformNav({ mountTo: document.body })
     *
     * Options:
     *   mountTo:  Element to prepend the nav into (default: document.body)
     *   apiBase:  Base URL for /api/me (default: '' = same origin)
     *   cssHref:  URL for platform.css (default: '/static/css/platform.css')
     *   currentPath: Override for window.location.pathname
     */
    window.renderPlatformNav = function (options) {
        options = options || {};
        var mountTo = options.mountTo || document.body;
        var apiBase = options.apiBase || '';
        var currentPath = options.currentPath || window.location.pathname;

        // Load platform.css if not already present
        if (options.cssHref !== false) {
            var cssHref = options.cssHref || '/static/css/platform.css';
            if (!document.querySelector('link[href="' + cssHref + '"]')) {
                var link = document.createElement('link');
                link.rel = 'stylesheet';
                link.href = cssHref;
                document.head.appendChild(link);
            }
        }

        // Fetch auth state, then build nav
        fetch(apiBase + '/api/me', { credentials: 'include' })
            .then(function (r) {
                if (!r.ok) return null;
                return r.json();
            })
            .catch(function () { return null; })
            .then(function (user) {
                var nav = buildNavElement(user, currentPath);
                mountTo.insertBefore(nav, mountTo.firstChild);

                // Add body padding
                document.body.style.paddingTop = 'var(--nav-height, 56px)';

                // Wire up interactions
                initNav();
            });
    };

    /* ---------- 3. Build nav DOM ---------- */

    var NAV_ITEMS_AUTH = [
        { label: 'Propagation Explorer', icon: '📡', href: '/' },
        { label: 'Operator View',        icon: '🧭', href: '/operator' },
        { label: 'Experiments',           icon: '📊', href: '/analytics/antenna-performance' },
        { label: 'Antennas',              icon: '📐', href: '/antenna/' },
        { label: 'Blog',                  icon: '📝', href: '/blog' }
    ];

    var NAV_ITEMS_PUBLIC = [
        { label: 'Propagation Explorer', icon: '📡', href: '/' }
    ];

    function isActive(href, currentPath) {
        if (href === '/') return currentPath === '/';
        return currentPath.indexOf(href) === 0;
    }

    function buildNavElement(user, currentPath) {
        var nav = document.createElement('nav');
        nav.className = 'platform-nav';
        nav.setAttribute('role', 'navigation');
        nav.setAttribute('aria-label', 'Main navigation');

        var items = user ? NAV_ITEMS_AUTH : NAV_ITEMS_PUBLIC;

        // Brand
        var brand = '<a href="/" class="platform-nav__brand">' +
            '<span class="platform-nav__brand-icon">◆</span>' +
            '<span>AntennaForge</span></a>';

        // Links
        var links = items.map(function (item) {
            var cls = 'platform-nav__link' + (isActive(item.href, currentPath) ? ' active' : '');
            return '<a href="' + item.href + '" class="' + cls + '">' +
                '<span class="platform-nav__link-icon">' + item.icon + '</span>' +
                item.label + '</a>';
        }).join('');

        // Right section
        var right = '';
        if (user) {
            var callsign = (user.callsign || user.username || 'User')
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            right = '<div class="platform-nav__right">' +
                '<button class="platform-nav__user-btn" aria-expanded="false" aria-haspopup="true">' +
                '<span>👤</span> ' + callsign + ' <span class="caret">▾</span></button>' +
                '<div class="platform-nav__dropdown">' +
                '<a href="/profile" class="platform-nav__dropdown-item">Profile</a>' +
                (user.is_admin ? '<a href="/admin" class="platform-nav__dropdown-item">Admin</a>' : '') +
                '<div class="platform-nav__dropdown-sep"></div>' +
                '<a href="/logout" class="platform-nav__dropdown-item platform-nav__dropdown-item--danger">Logout</a>' +
                '</div></div>';
        } else {
            right = '<div class="platform-nav__right">' +
                '<a href="/login" class="platform-nav__auth-link platform-nav__auth-link--login">Login</a>' +
                '<a href="/register" class="platform-nav__auth-link platform-nav__auth-link--signup">Sign Up</a>' +
                '</div>';
        }

        // Hamburger
        var hamburger = '<button class="platform-nav__hamburger" aria-expanded="false" aria-label="Menu">' +
            '<span></span><span></span><span></span></button>';

        nav.innerHTML = brand +
            '<div class="platform-nav__links">' + links + '</div>' +
            right + hamburger;

        // Build mobile overlay
        var overlay = document.createElement('div');
        overlay.className = 'platform-nav__overlay';

        var overlayLinks = items.map(function (item) {
            var cls = 'platform-nav__link' + (isActive(item.href, currentPath) ? ' active' : '');
            return '<a href="' + item.href + '" class="' + cls + '">' +
                '<span class="platform-nav__link-icon">' + item.icon + '</span>' +
                item.label + '</a>';
        }).join('');

        if (user) {
            var escapedCallsign = (user.callsign || user.username || 'User')
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            overlay.innerHTML = overlayLinks +
                '<div class="platform-nav__overlay-sep"></div>' +
                '<div class="platform-nav__overlay-user">' + escapedCallsign + '</div>' +
                '<a href="/profile" class="platform-nav__link">👤 Profile</a>' +
                (user.is_admin ? '<a href="/admin" class="platform-nav__link">🔐 Admin</a>' : '') +
                '<a href="/logout" class="platform-nav__link" style="color:var(--accent-danger)">🚪 Logout</a>';
        } else {
            overlay.innerHTML = overlayLinks +
                '<div class="platform-nav__overlay-sep"></div>' +
                '<a href="/login" class="platform-nav__link">Login</a>' +
                '<a href="/register" class="platform-nav__link">Sign Up</a>';
        }

        // Append overlay after nav
        nav.appendChild(overlay);

        return nav;
    }

    /* ---------- 4. Auto-init on DOMContentLoaded ---------- */

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initNav);
    } else {
        initNav();
    }

})();
