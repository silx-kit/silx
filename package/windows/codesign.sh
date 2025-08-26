#!/usr/bin/env bash

# The script codesigns the application bundle.

# Exit immediately if a command exits with a non-zero status
set -e
# Print commands and their arguments as they are executed (debug mode)
set -x

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') INFO [codesign] $1"
}

log "Setting the required environment variables."
APP_NAME="silx-view"
ROOT="${PWD}"
APP_PATH="${ROOT}/dist/${APP_NAME}.app"
KEYCHAIN_PATH="${ROOT}/notarize.keychain-db"
CERTIFICATE_PATH="${ROOT}/certificate.p12"

log "Creating a temporary keychain."
security create-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"
security set-keychain-settings -lut 21600 "${KEYCHAIN_PATH}"
security unlock-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"

log "Calculate the SHA-256 hash of the keychain password."
log "Keychain password SHA-256: $(echo -n "${KEYCHAIN_PASSWORD}" | shasum -a 256 | awk '{print $1}')"

log "Importing the certificate from the base64 string."
echo "${CERTIFICATE_BASE64}" | base64 --decode -o "${CERTIFICATE_PATH}"

log "Calculate the SHA-256 hash of the certificate."
log "Certificate SHA-256: $(shasum -a 256 "${CERTIFICATE_PATH}" | awk '{print $1}')"

log "Importing the certificate to the keychain."
security import "${CERTIFICATE_PATH}" \
  -P "${CERTIFICATE_PASSWORD}" \
  -A -t cert -f pkcs12 \
  -k "${KEYCHAIN_PATH}"

log "Configuring keychain access control for codesigning without UI prompts."
security set-key-partition-list \
  -S apple-tool:,apple: \
  -k "${KEYCHAIN_PASSWORD}" \
  "${KEYCHAIN_PATH}"

security find-certificate ${ROOT}/notarize.keychain-db

log "Codesigning the application bundle."
codesign --verbose --force --deep --options=runtime \
  --entitlements ./entitlements.plist \
  --keychain "${KEYCHAIN_PATH}" \
  --timestamp "${APP_PATH}" \
  --sign "Developer ID Application: MARIUS SEPTIMIU RETEGAN (${APPLE_TEAM_ID})"

log "Removing the certificate file and keychain."
rm "${CERTIFICATE_PATH}"
security delete-keychain "${KEYCHAIN_PATH}"

log "Codesigning completed successfully."