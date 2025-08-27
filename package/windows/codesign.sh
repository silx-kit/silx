#!/usr/bin/env bash

# The script codesigns the application bundle.

# Exit immediately if a command exits with a non-zero status
set -e
# Print commands and their arguments as they are executed (debug mode)
set -x

log() {
  echo "$(date "+%Y-%m-%d %H:%M:%S") INFO [codesign] $1"
}

log "Setting the required environment variables."
APP_NAME="silx-view"
ROOT="${PWD}"
APP_PATH="${ROOT}/dist/${APP_NAME}.app"
KEYCHAIN_PATH="${ROOT}/notarize.keychain-db"
CERTIFICATE_PATH="${ROOT}/certificate.p12"

# Ensure cleanup always runs: remove certificate file and delete temporary keychain
cleanup() {
  # Don't fail the cleanup if a command fails
  set +e
  log "Cleanup: removing certificate and deleting temporary keychain if present."

  if [ -n "${CERTIFICATE_PATH}" ] && [ -f "${CERTIFICATE_PATH}" ]; then
    rm -f "${CERTIFICATE_PATH}"
  fi

  # Try deleting the temporary keychain if it exists
  if [ -n "${KEYCHAIN_PATH}" ]; then
    # security delete-keychain returns non-zero when the keychain does not exist; ignore errors
    security delete-keychain "${KEYCHAIN_PATH}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

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
  
  # -A -t cert -f pkcs12 \
  # -T /usr/bin/codesign \
  # -T /usr/bin/security \

# security find-identity -v -p codesigning "$KEYCHAIN_PATH"

APPLE_SIGNING_ID="CCE0C936E0F330BF8058D5C5636025F3095E0A82"

# log "Configuring keychain access control for codesigning without UI prompts."
# security set-key-partition-list \
#   -S "apple-tool:,apple:,codesign:" \
#   -s -k "${KEYCHAIN_PASSWORD}" \
#   -D "${APPLE_SIGNING_ID}" \
#   -t private \
#   "${KEYCHAIN_PATH}"
  
# security find-certificate "${ROOT}/notarize.keychain-db"
security list-keychains -s "${KEYCHAIN_PATH}"

log "Codesigning the application bundle."
codesign -vvv --force --deep --strict --sign "${APPLE_SIGNING_ID}" \
  --timestamp \
  --options runtime \
  --entitlements "./entitlements.plist" \
  --keychain "${KEYCHAIN_PATH}" \
  "${APP_PATH}"

log "Codesigning completed successfully."

# Cleanup is handled by the EXIT trap
log "Cleanup handled by trap."