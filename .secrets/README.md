# Local Secrets (Do Not Commit)

This directory stores host-local secret files used by Docker Compose `secrets:`.

## OpenAI API Key

Create the OpenAI secret file:

```bash
printf '%s' 'sk-YOUR-OPENAI-API-KEY' > secrets/openai_api_key
chmod 600 secrets/openai_api_key
```

The `ac-dashboard` service reads it from `/run/secrets/openai_api_key`.

## Notes

- Keep this directory on the deployment host only.
- Files under `secrets/*` are git-ignored (except this README).
- Rotate any secret if exposed.
