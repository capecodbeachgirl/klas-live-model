# Put the KLAS dashboard online

This version is designed to run itself from GitHub Actions and publish `docs/index.html` through GitHub Pages.

## Before publishing

Your repository must contain the trained live bundles:

- `data/model/h08.joblib` through `data/model/h18.joblib`
- `data/model/manifest.json`

Run locally:

```powershell
python scripts\check_deploy_ready.py
python scripts\live_update.py
```

## First GitHub setup

1. Create/publish a GitHub repository for this project (recommended name: `klas-live-model`).
2. Make sure the entire `data/model` folder is committed and pushed.
3. On GitHub open **Settings → Pages**.
4. Under **Build and deployment**, choose **GitHub Actions** as the source.
5. Open the **Actions** tab → **KLAS hourly live update + Pages** → **Run workflow** once.
6. After the run succeeds, GitHub Pages will show the public site URL in the deployment job / Pages settings.

The normal GitHub.com Pages address is usually:

`https://YOUR-USERNAME.github.io/klas-live-model/`

## What happens automatically

At approximately `:05` after every hour, Las Vegas local time, GitHub Actions:

1. pulls the latest KLAS observations (with live fallback source),
2. reads the fixed NWS morning high,
3. updates NWS hourly rain/thunder/cloud intelligence,
4. reads the latest Las Vegas NWS Area Forecast Discussion,
5. checks MRMS radar proximity/trend,
6. pulls current Kalshi temperature-market quotes,
7. runs the current KLAS checkpoint model,
8. appends/preserves the live hourly history,
9. regenerates `docs/index.html`, and
10. deploys the refreshed page through GitHub Pages.

The workflow can also be run manually from the Actions tab at any time.

## Important operational notes

- GitHub scheduled workflows can occasionally start late. The dashboard therefore treats `:05` as a target, not an exact guarantee.
- Scheduled workflows only run from the repository's default branch.
- Public-repository schedules may be disabled by GitHub after 60 days without repository activity.
- If radar, AFD, or Kalshi temporarily fails, the live program is designed to continue and mark that source unavailable where possible.
- If both live KLAS observation sources fail, the refresh cannot make a safe current prediction and should fail rather than invent data.
