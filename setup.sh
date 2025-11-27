VAR_NAME="POPCORN_API_URL"
VAR_VALUE="http://44.228.22.179:8000"

if [ -n "$ZSH_VERSION" ] || [[ "$SHELL" == *"zsh"* ]]; then
    RC_FILE="$HOME/.zshrc"
else
    RC_FILE="$HOME/.bashrc"
fi

echo "export $VAR_NAME=\"$VAR_VALUE\"" >> "$RC_FILE"
echo "  Variable: $VAR_NAME"
echo "  Value.  : $VAR_VALUE"
echo "Saved to $RC_FILE."
echo "To activate, please run:"
echo "  source $RC_FILE"