#!/usr/bin/env bash

# The script codesigns the application bundle.

# Exit immediately if a command exits with a non-zero status.
set -e
# Print commands and their arguments as they are executed (debug mode).
set -x

log() {
  echo "$(date "+%Y-%m-%d %H:%M:%S") INFO [codesign] $1"
}

# Check if required environment variables are set,
if [[ -z "${APPLE_TEAM_ID}" || -z "${KEYCHAIN_PASSWORD}" || -z "${CERTIFICATE_BASE64}" || -z "${CERTIFICATE_PASSWORD}" ]]; then
  log "Skipping code signing; the required environment variables (APPLE_TEAM_ID, KEYCHAIN_PASSWORD, CERTIFICATE_BASE64, CERTIFICATE_PASSWORD) are not set."
  exit 0
fi

log "Setting the required environment variables."
APP_NAME="silx-view"
ROOT="${PWD}"
APP_PATH="${ROOT}/dist/${APP_NAME}.app"
KEYCHAIN_PATH="${ROOT}/notarize.keychain-db"
CERTIFICATE_PATH="${ROOT}/certificate.p12"
APPLE_SIGNING_ID="Developer ID Application: MARIUS SEPTIMIU RETEGAN (${APPLE_TEAM_ID})"

log "Creating a temporary keychain."
security create-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"
security set-keychain-settings -lut 21600 "${KEYCHAIN_PATH}"
security unlock-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"

log "Importing the certificate from the base64 string."
echo "${CERTIFICATE_BASE64}" | base64 --decode -o "${CERTIFICATE_PATH}"

log "Importing the certificate to the keychain."
security import "${CERTIFICATE_PATH}" \
  -P "${CERTIFICATE_PASSWORD}" \
  -A -t cert -f pkcs12 \
  -k "${KEYCHAIN_PATH}"

log "Configuring keychain access control for codesigning without UI prompts."
security set-key-partition-list \
  -S "apple-tool:,apple:,codesign:" \
  -s -k "${KEYCHAIN_PASSWORD}" \
  -D "${APPLE_SIGNING_ID}" \
  -t private \
  "${KEYCHAIN_PATH}"
  
log "Setting the keychain as the default so codesign can find the certificate."
security list-keychains -s "${KEYCHAIN_PATH}"

log "Codesigning the application bundle."
codesign -vvv --force --deep --strict --sign "${APPLE_SIGNING_ID}" \
  --timestamp \
  --options runtime \
  --entitlements "./entitlements.plist" \
  --keychain "${KEYCHAIN_PATH}" \
  "${APP_PATH}"

log "Cleanup the keychain and certificate file."
rm -f "${CERTIFICATE_PATH}"
security delete-keychain "${KEYCHAIN_PATH}"

log "Codesigning completed successfully."