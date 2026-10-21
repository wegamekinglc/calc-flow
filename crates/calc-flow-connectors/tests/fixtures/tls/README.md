These certificates and the unencrypted server key are public, local test fixtures.
They trust only a generated test CA and a `localhost` leaf (no IP SAN), allowing
offline checks for trusted, untrusted and wrong-host TLS connections. They are
not runtime credentials. The CA signing key was discarded after generation.
