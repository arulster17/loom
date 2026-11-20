// Command caddy is Caddy with the per-client rate-limit module the production Caddyfile
// uses (docs/phase1-m3-spec.md §6.7, §15 D2). go.mod and go.sum pin every version.
package main

import (
	caddycmd "github.com/caddyserver/caddy/v2/cmd"

	// The standard modules, as the official image has them.
	_ "github.com/caddyserver/caddy/v2/modules/standard"
	// The rate-limit module (the `rate_limit` directive).
	_ "github.com/mholt/caddy-ratelimit"
)

func main() {
	caddycmd.Main()
}
