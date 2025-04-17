#!/usr/bin/env bash

# The script submits the application for notarization to Apple.

set -e

echo "Setting the required environment variables."
APP_NAME="silx-view"
ROOT="${PWD}"
APP_DMG="${ROOT}"/artifacts/${APP_NAME}.dmg

echo "Submiting the application for notarization."
xcrun notarytool submit --apple-id $APPLE_ID --team-id $APPLE_TEAM_ID --password $APPLICATION_SPECIFIC_PASSWORD  --wait "$APP_DMG"

echo "Stapling the notarization ticket to the application bundle."
xcrun stapler staple "$APP_DMG"

echo "Notarization completed successfully."