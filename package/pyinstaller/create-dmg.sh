#!/usr/bin/env bash

# The script creates a DMG file for the application bundle.

# Exit immediately if a command exits with a non-zero status
set -e
# Print commands and their arguments as they are executed (debug mode)
# set -x

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') INFO [create-dmg] $1"
}

log "Setting environment variables."
APP_NAME="silx"
ROOT="${PWD}"
APP="${ROOT}/dist/${APP_NAME}.app"
RESOURCES="${ROOT}" # The path to resources (volume icon, background, ...)."

log "Setting derived environment variables."
ARTIFACTS="${ROOT}/artifacts"
TEMPLATE="${ARTIFACTS}/template"
TEMPLATE_DMG="${ARTIFACTS}/template.dmg"
APP_DMG="${ARTIFACTS}/${APP_NAME}.dmg"

SLEEP_INTERVAL=5

log "Checking if artifacts folder exists."
[ -d "${ARTIFACTS}" ] && rm -rf "${ARTIFACTS}"

log "Creating the artifacts folder."
mkdir -p "${ARTIFACTS}"

log "Removing any previous images."
if [[ -e "${TEMPLATE}" ]]; then rm -rf "${TEMPLATE}"; fi
if [[ -e "${APP_DMG}" ]]; then rm -rf "${APP_DMG}"; fi

log "Copying required files."
mkdir -p "${TEMPLATE}/.background"
cp -a "${RESOURCES}/background.pdf" "${TEMPLATE}/.background/background.pdf"
cp -a "${RESOURCES}/silx.icns" "${TEMPLATE}/.VolumeIcon.icns"
cp -a "${RESOURCES}/DS_Store" "${TEMPLATE}/.DS_Store"
cp -a "${APP}" "${TEMPLATE}/${APP_NAME}.app"
ln -s "/Applications/" "${TEMPLATE}/Applications"

log "Creating a regular .fseventsd/no_log file."
mkdir "${TEMPLATE}/.fseventsd"
touch "${TEMPLATE}/.fseventsd/no_log"

log "Sleeping for a few seconds."
sleep "${SLEEP_INTERVAL}"

log "Creating the temporary disk image."
hdiutil create -verbose -format UDRW -volname "${APP_NAME}" -fs APFS \
       -srcfolder "${TEMPLATE}" \
       "${TEMPLATE_DMG}"

log "Sleeping for a few seconds."
sleep "${SLEEP_INTERVAL}"

log "Detaching the temporary disk image if still attached."
hdiutil detach -verbose "/Volumes/${APP_NAME}" -force || true

log "Attaching the temporary disk image in read/write mode."
MOUNT_OUTPUT=$(hdiutil attach -readwrite -noverify -noautoopen "${TEMPLATE_DMG}" | grep '^/dev/')
DEV_NAME=$(echo -n "${MOUNT_OUTPUT}" | head -n 1 | awk '{print $1}')
MOUNT_POINT=$(echo -n "${MOUNT_OUTPUT}" | tail -n 1 | awk '{print $3}')

log "Fixing permissions."
chmod -Rf go-w "${TEMPLATE}" || true

log "Hiding the background directory even more."
SetFile -a V "${MOUNT_POINT}/.background"

log "Seting the custom icon volume flag so that volume has nice icon."
SetFile -a C "${MOUNT_POINT}"

log "Moving the icons to the appropriate position relative to the background."
WINDOW_SIZE="{0, 0, 500, 300}"
APP_ICON_POSITION="{114, 124}"
APPLICATIONS_ICON_POSITION="{386, 124}"

osascript <<EOF
tell application "Finder"
    tell disk "${APP_NAME}"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set bounds of container window to ${WINDOW_SIZE}

        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 96
        set background picture of viewOptions to file ".background:background.pdf"

        set position of item "${APP_NAME}" to ${APP_ICON_POSITION}
        set position of item "Applications" to ${APPLICATIONS_ICON_POSITION}

        close
        open
        delay 1
        update without registering applications
    end tell
end tell
EOF

log "Sleeping for a few seconds."
sleep "${SLEEP_INTERVAL}"

log "Detaching the temporary disk image"
hdiutil detach "${DEV_NAME}" -force || true

log "Sleeping for a few seconds."
sleep "${SLEEP_INTERVAL}"

log "Converting the temporary image to a compressed image."
hdiutil convert "${TEMPLATE_DMG}" \
       -format UDZO \
       -imagekey zlib-level=9 \
       -o "${APP_DMG}"

log "Cleaning up."
rm -rf "${TEMPLATE}"
rm -rf "${TEMPLATE_DMG}"