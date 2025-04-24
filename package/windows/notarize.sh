#!/usr/bin/env bash

# The script submits the application for notarization to Apple.

# Exit immediately if a command exits with a non-zero status
set -e
# Print commands and their arguments as they are executed (debug mode)
set -x

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') INFO [notarize] $1"
}

log "Setting the required environment variables."
APP_NAME="silx-view"
ROOT="${PWD}"
APP_DMG="${ROOT}/artifacts/${APP_NAME}.dmg"

log "Submiting the application for notarization."
xcrun notarytool submit \
  --apple-id "${APPLE_ID}" \
  --team-id "${APPLE_TEAM_ID}" \
  --password "${APPLICATION_SPECIFIC_PASSWORD}" \
  --wait "${APP_DMG}"

log "Stapling the notarization ticket to the application bundle."
xcrun stapler staple "${APP_DMG}"

log "Notarization completed successfully."