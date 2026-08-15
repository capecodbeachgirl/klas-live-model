from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path


def _v(value, suffix=""):
    if value is None:
        return "—"
    return f"{value}{suffix}"


def _friendly_time(value: object) -> str:
    if value is None:
        return "—"
    try:
        dt = datetime.fromisoformat(str(value))
        # Windows-safe rendering is not relevant here; this runs in Python but outputs HTML.
        return dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return str(value)


def _risk_class(value: object) -> str:
    v = str(value or "UNKNOWN").upper()
    return {"LOW": "low", "MEDIUM": "med", "HIGH": "high"}.get(v, "neutral")


def _fmt_pct(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.0f}%"
    except Exception:
        return "—"


def _progression_html(state: dict) -> str:
    rows = state.get("progression") or []
    if not rows:
        return '<div class="muted">Hourly progression will appear after multiple updates today.</div>'
    body = []
    for r in rows:
        current = r.get("latest_precise_temp_f")
        if current is None:
            current = r.get("latest_temp_f")
        body.append(
            "<tr>"
            f"<td>{_v(r.get('checkpoint_hour'), ':00')}</td>"
            f"<td>{_v(None if current is None else round(float(current),1), '°')}</td>"
            f"<td>{_v(r.get('six_hour_max_f'), '°')}</td>"
            f"<td><strong>{_v(None if r.get('model_predicted_high_f') is None else round(float(r.get('model_predicted_high_f')),1), '°')}</strong></td>"
            f"<td>{_v(r.get('nws_am_forecast_high_f'), '°')}</td>"
            f"<td>{escape(str(r.get('weather_risk') or '—'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Time</th><th>KLAS</th><th>6h max</th><th>Model</th><th>NWS</th><th>Risk</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def render_dashboard(state: dict) -> str:
    model = state.get("model_available")
    markets = state.get("markets", [])[:7]
    bucket_rows = "".join(
        f"<tr><td><strong>{escape(str(m.get('subtitle') or m.get('ticker') or ''))}</strong></td>"
        f"<td>{_v(None if m.get('model_probability') is None else round(100*m['model_probability'],1),'%')}</td>"
        f"<td>{_v(None if m.get('yes_bid') is None else round(100*m['yes_bid'],1),'¢')}</td>"
        f"<td>{_v(None if m.get('yes_ask') is None else round(100*m['yes_ask'],1),'¢')}</td>"
        f"<td class='{('pos' if (m.get('edge_vs_ask') or 0) > 0 else 'neg')}'>{_v(None if m.get('edge_vs_ask') is None else round(100*m['edge_vs_ask'],1),' pts')}</td></tr>"
        for m in markets
    ) or '<tr><td colspan="5">Kalshi market data not available yet.</td></tr>'

    reasons = "".join(f"<li>{escape(str(x))}</li>" for x in state.get("weather_reasons", []))
    model_high = round(state.get("model_predicted_high_f", 0), 1) if model else None
    likely = f"{state.get('likely_low_f')}–{state.get('likely_high_f')}°F" if model else "—"
    correction = state.get("model_correction_f")
    correction_text = "—" if correction is None else f"{correction:+.1f}°F vs NWS"
    current_precise = state.get("latest_precise_temp_f")
    current_display = round(current_precise, 1) if current_precise is not None else state.get("latest_temp_f")
    metar_peak = state.get("precise_metar_peak_f")
    metar_peak_display = round(metar_peak, 1) if metar_peak is not None else state.get("raw_metar_peak_f")
    six = state.get("six_hour_max_f")
    six_display = "Not reported yet" if six is None else f"{six}°F"
    total = state.get("bucket_probability_total")
    total_text = "" if total is None else f"Model bucket total: {100*total:.1f}%"
    top_gap = state.get("largest_model_ask_gap")
    top_gap_text = "—"
    if top_gap:
        top_gap_text = f"{escape(str(top_gap.get('subtitle')))} · {100*top_gap.get('edge_vs_ask',0):+.1f} pts"

    nws_live = state.get("nws_live_forecast") or {}
    afd = state.get("afd") or {}
    radar = state.get("radar") or {}
    satellite = state.get("satellite") or {}
    components = state.get("weather_risk_components") or {}
    pop = nws_live.get("max_pop_pct")
    sky = nws_live.get("max_sky_cover_pct")
    thunder = "YES" if nws_live.get("thunder_possible") else "No"
    radar_summary = escape(str(radar.get("summary") or "Radar scan unavailable"))
    nearest_echo = ((radar.get("current") or {}).get("nearest_echo_miles"))
    if nearest_echo is None:
        radar_distance_text = "Nearest meaningful echo: none detected within the 50-mile scan rings"
    else:
        radar_distance_text = f"Nearest meaningful echo: about {float(nearest_echo):.0f} miles from KLAS"
    if radar.get("approaching"):
        radar_distance_text += " · trend: moving closer"
    afd_snippet = escape(str(afd.get("snippet") or "AFD unavailable"))[:520]
    radar_url = escape(str(radar.get("image_url") or ""), quote=True)
    radar_panel = (
        f'<div class="radar-box"><img src="{radar_url}" alt="NWS MRMS radar around KLAS">'
        '<div class="crosshair">✚</div><div class="radar-label">KLAS</div></div>'
        if radar_url else '<div class="muted">Radar image unavailable.</div>'
    )

    satellite_url = escape(
        str(satellite.get("geocolor_image_url") or ""),
        quote=True,
    )
    satellite_summary = escape(
        str(satellite.get("summary") or "Satellite cloud watch unavailable")
    )
    satellite_risk = escape(
        str(satellite.get("risk") or "UNKNOWN")
    )

    satellite_panel = (
        f'<div class="radar-box"><img src="{satellite_url}" '
        f'alt="GOES satellite cloud cover around Las Vegas" '
        f'style="max-height:340px; object-fit:contain;"></div>'
        if satellite_url
        else '<div class="muted">Satellite image unavailable.</div>'
    )

    if correction is None:
        why = "Model not available at this checkpoint yet."
    elif correction <= -0.5:
        why = "KLAS is tracking cool enough versus its historical intraday pattern to pull the model below the NWS morning high."
    elif correction >= 0.5:
        why = "KLAS is tracking warm enough versus its historical intraday pattern to push the model above the NWS morning high."
    else:
        why = "Current KLAS behavior does not justify moving far from the NWS morning forecast."
    if state.get("weather_risk") in {"MEDIUM", "HIGH"}:
        why += " Forward-looking weather risk lowers confidence; it does not automatically change the validated temperature correction."

    status = escape(str(state.get("research_status") or "MODEL RUNNING"))
    next_update = _friendly_time(state.get("next_update_local"))

    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KLAS Live Model</title><style>
:root{{--bg:#f4f6f8;--card:#fff;--border:#dde2e7;--text:#18212b;--muted:#667085;--good:#16794c;--warn:#a15c00;--bad:#b42318;--accent:#1f4e79}}
*{{box-sizing:border-box}} body{{font-family:Arial,sans-serif;background:var(--bg);color:var(--text);margin:0}} .wrap{{max-width:1180px;margin:auto;padding:22px}}
h1{{margin:0 0 4px;font-size:34px}} .sub{{color:var(--muted);margin-bottom:14px}} .status{{background:#142b44;color:#fff;padding:13px 16px;border-radius:12px;margin-bottom:14px;display:flex;justify-content:space-between;gap:12px;align-items:center}} .status strong{{font-size:19px}} .status span{{font-size:13px;opacity:.9}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}} .card,section{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}} .big{{font-size:30px;font-weight:700;margin-top:7px}} .label{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
section{{margin-top:14px}} .two{{display:grid;grid-template-columns:1.15fr .85fr;gap:14px}} .intel-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}} .pill{{border:1px solid var(--border);border-radius:10px;padding:10px}} .pill .k{{font-size:11px;color:var(--muted);text-transform:uppercase}} .pill .v{{font-size:17px;font-weight:700;margin-top:4px}} .low{{color:var(--good)}} .med{{color:var(--warn)}} .high{{color:var(--bad)}} .neutral{{color:var(--muted)}}
table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:9px;border-bottom:1px solid #edf0f2;font-size:14px}} th{{color:var(--muted)}} .risk{{font-weight:700}} ul{{margin:8px 0 0 18px}} .foot,.muted,.mini{{font-size:12px;color:var(--muted)}} .pos{{font-weight:700}} .neg{{color:var(--muted)}} .why{{border-left:4px solid var(--accent);padding:8px 10px;background:#f8fafc;margin-top:10px;font-size:14px}}
.radar-box{{position:relative;background:#eef1f4;border-radius:10px;overflow:hidden;min-height:260px}} .radar-box img{{width:100%;height:100%;min-height:260px;object-fit:cover;display:block}} .crosshair{{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);font-size:30px;color:#111;text-shadow:0 0 3px #fff}} .radar-label{{position:absolute;left:calc(50% + 13px);top:calc(50% - 22px);font-size:11px;font-weight:700;background:#fff;padding:2px 4px;border-radius:3px}}
@media(max-width:850px){{.two{{grid-template-columns:1fr}} .intel-grid{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:650px){{.wrap{{padding:12px}} h1{{font-size:28px}} th,td{{padding:7px 5px;font-size:12px}} .big{{font-size:26px}} .status{{align-items:flex-start;flex-direction:column}}}}
</style></head><body><div class="wrap"><h1>KLAS Live High Model</h1><div class="sub">Updated {_friendly_time(state.get('updated_at_local'))} · Latest METAR {_friendly_time(state.get('latest_metar_time'))}</div>
<div class="status"><strong>{status}</strong><span>Next hourly refresh: ~{next_update} Las Vegas time</span></div>
<div class="grid">
<div class="card"><div class="label">Current KLAS</div><div class="big">{_v(current_display,'°F')}</div><div class="mini">Precise METAR T-group when available</div></div>
<div class="card"><div class="label">NWS Morning High</div><div class="big">{_v(state.get('nws_am_forecast_high_f'),'°F')}</div></div>
<div class="card"><div class="label">Our Model High</div><div class="big">{_v(model_high,'°F')}</div><div>{correction_text}</div></div>
<div class="card"><div class="label">80% Model Range</div><div class="big">{likely}</div><div>{escape(str(state.get('confidence','—')))} confidence</div></div>
<div class="card"><div class="label">6-Hour ASOS Max</div><div class="big">{six_display}</div></div>
<div class="card"><div class="label">Precise METAR Peak</div><div class="big">{_v(metar_peak_display,'°F')}</div></div>
</div>
<section><h3>Live weather intelligence</h3><div class="intel-grid">
<div class="pill"><div class="k">Observed KLAS</div><div class="v {_risk_class(components.get('observed_risk'))}">{escape(str(components.get('observed_risk','—')))}</div></div>
<div class="pill"><div class="k">NWS forecast</div><div class="v {_risk_class(components.get('forecast_risk'))}">{escape(str(components.get('forecast_risk','—')))}</div></div>
<div class="pill"><div class="k">NWS discussion</div><div class="v {_risk_class(components.get('afd_risk'))}">{escape(str(components.get('afd_risk','—')))}</div></div>
<div class="pill"><div class="k">Radar</div><div class="v {_risk_class(components.get('radar_risk'))}">{escape(str(components.get('radar_risk','—')))}</div></div>
</div>
<div class="risk">Overall weather risk: <span class="{_risk_class(state.get('weather_risk'))}">{escape(str(state.get('weather_risk','—')))}</span></div><ul>{reasons}</ul>
<div class="why"><strong>Why model is here:</strong> {escape(why)}</div></section>
<div class="two"><section><h3>Forecast rain / convection</h3>
<p><strong>Thunder forecast:</strong> {thunder} &nbsp; · &nbsp; <strong>Max rain chance:</strong> {_fmt_pct(pop)} &nbsp; · &nbsp; <strong>Max sky cover:</strong> {_fmt_pct(sky)}</p>
<div class="mini">{escape(str(nws_live.get('summary') or 'NWS hourly forecast unavailable'))}</div><h4>NWS Las Vegas discussion</h4><div class="mini">{afd_snippet}</div></section>
<section>
<h3>Radar around KLAS</h3>
{radar_panel}
<div class="mini" style="margin-top:7px">
<strong>{escape(radar_distance_text)}</strong><br>
{radar_summary}. Center marker = KLAS. Automated ring scan is intentionally coarse.
</div>

<h4>Satellite / cloud shading</h4>
{satellite_panel}

<div class="risk" style="margin-top:8px">
Cloud shading risk:
<span class="{_risk_class(satellite.get('risk'))}">
{satellite_risk}
</span>
</div>

<div class="mini" style="margin-top:6px">
{satellite_summary}. GOES GeoColor is currently a visual cross-check; the shading risk is based on observed KLAS cloud cover.
</div>
</section>
</div>
<section><h3>Today's progression</h3>{_progression_html(state)}</section>
<section><h3>Kalshi buckets</h3><table><thead><tr><th>Bucket</th><th>Model</th><th>Bid</th><th>Ask</th><th>Model − ask</th></tr></thead><tbody>{bucket_rows}</tbody></table><div class="mini">{escape(total_text)} · Largest model-vs-ask gap: {top_gap_text}</div></section>
<section><h3>Model status</h3><p>Checkpoint: {_v(state.get('checkpoint_hour'),':00 local')} · Held-out MAE: {_v(None if state.get('model_mae_f') is None else round(state.get('model_mae_f'),2),'°F')} · Hourly refresh target: ~:05 after the routine METAR.</p><div class="mini">Forecast/radar/AFD signals currently affect risk and confidence, not the validated temperature correction. We will only let them alter the predicted high after separate historical validation.</div></section>
<div class="foot">Research dashboard only. Model probabilities are estimates, not guarantees. Final settlement target remains the official NWS Daily Climate Report high.</div></div></body></html>'''


def save_dashboard(state: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(state), encoding="utf-8")
    return path
