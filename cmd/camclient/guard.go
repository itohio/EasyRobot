package main

import (
	"net"
	"net/http"
	"net/url"
)

// sameOrigin guards both servers against other web pages (verify S1):
//   - a Host that is not a loopback name is refused, so a DNS-rebound
//     hostname cannot read the image or the settings;
//   - a request other than GET/HEAD whose Origin is not this server's own
//     origin is refused, so another site's page cannot drive the board,
//     overwrite the relayed image or write snapshots. Requests without an
//     Origin (curl, peers' scripts) pass.
func sameOrigin(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !loopbackHost(r.Host) {
			http.Error(w, "camclient: Host must be a loopback address", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			if o := r.Header.Get("Origin"); o != "" {
				u, err := url.Parse(o)
				if err != nil || u.Scheme != "http" || u.Host != r.Host {
					http.Error(w, "camclient: cross-origin request refused", http.StatusForbidden)
					return
				}
			}
		}
		h.ServeHTTP(w, r)
	})
}

// loopbackHost reports whether host (Host header, with or without a port)
// names this machine: localhost or a loopback IP.
func loopbackHost(host string) bool {
	if h, _, err := net.SplitHostPort(host); err == nil {
		host = h
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
