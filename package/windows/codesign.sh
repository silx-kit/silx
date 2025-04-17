#!/usr/bin/env bash

# The script codesigns the application bundle.

set -e
set -x

echo "Setting the required environment variables."
APP_NAME="silx-view"
ROOT="${PWD}"
APP_PATH="${ROOT}/dist/${APP_NAME}.app"
KEYCHAIN_PATH="${ROOT}/notarize.keychain-db"
CERTIFICATE_PATH="${ROOT}/certificate.p12"

echo "Creating a temporary keychain."
security create-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"
security set-keychain-settings -lut 21600 "${KEYCHAIN_PATH}"
security unlock-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"

echo "Importing the certificate from the base64 string."
echo "${CERTIFICATE_BASE64}" | base64 --decode -o "${CERTIFICATE_PATH}"

cat "${CERTIFICATE_PATH}" 

echo "Importing the certificate to the keychain."
security import "${CERTIFICATE_PATH}" -P "${CERTIFICATE_PASSWORD}" -A -t cert -f pkcs12 -k "${KEYCHAIN_PATH}"
security set-key-partition-list -S apple-tool:,apple: -k "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"

echo "Codesigning the application bundle."
codesign --force --deep --options=runtime --entitlements ./entitlements.plist --verbose --sign "Developer ID Application: MARIUS SEPTIMIU RETEGAN (${APPLE_TEAM_ID})" --keychain "${KEYCHAIN_PATH}" --timestamp "${APP_PATH}"

echo "Removing the certificate and keychain."
rm "${CERTIFICATE_PATH}"
security delete-keychain "${KEYCHAIN_PATH}"

echo "Codesigning completed successfully."