wget -O simple_example.zip https://www.dropbox.com/scl/fi/gme6vcocnniv7csez0muy/simple_example.zip?rlkey=v0c57s094gtezqqrfds80dje2
EXPECTED_HASH="fecd9ac5e5466367eaa5826246449a94e7a6e54313b4bfce9633f7f817da5100"

CALCULATED_HASH=$(sha256sum simple_example.zip | awk '{ print $1 }')
# Compare the calculated hash to the expected hash
if [ "$CALCULATED_HASH" != "$EXPECTED_HASH" ]; then
  echo "Error: SHA-256 hash does not match"
else
  echo "SHA-256 hash matches"
fi