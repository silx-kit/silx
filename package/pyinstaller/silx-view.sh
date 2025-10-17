#!/bin/bash
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$APP_DIR/silx" view "$@"