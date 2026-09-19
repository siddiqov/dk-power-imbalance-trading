import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

from src.ledger_css import LEDGER_CSS
from src.ledger_js import LEDGER_JS

COST_EUR_MWH = 0.51

def render_unified_ledger(df_day):
    st.subheader("96-Quarter Intraday Trading Ledger: V3.2 vs V4.0 vs V4.1 Side-by-Side")
    st.markdown("""
    **Authentic Multi-Generation Audit:** Side-by-side comparison across **V3.2 (Flow-Aware)**, **V4.0 (Full-Grid Champion)**, and **V4.1 (Institutional High-Alpha)**.
    """)

    if df_day is None or df_day.empty:
        st.warning("No live trading data available for selected date.")
        return

    st.markdown("##### ⚙️ Model Selection & Display Mode")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_v32 = st.checkbox("Show V3.2 Flow-Aware", value=False, key="ledger_show_v32")
    with col2:
        show_v40 = st.checkbox("Show V4.0 Full-Grid", value=False, key="ledger_show_v40")
    with col3:
        show_v41 = st.checkbox("Show V4.1 High-Alpha", value=True, key="ledger_show_v41")

    selected_count = sum([bool(show_v32), bool(show_v40), bool(show_v41)])
    if selected_count == 0:
        st.info("Please select at least one version to display.")
        return

    only_v41 = (show_v41 and not show_v32 and not show_v40)

    if only_v41:
        st.info("💡 **Single Model Mode (V4.1 Only):** Displaying flat single table with all V4.1 trading decisions, grid signals, and commercial cash flows. Accordion folding drawers are disabled.")
    else:
        st.info("💡 **Multi-Model Comparison Mode:** Displaying comparative master table with interactive folding drawers for granular audit.")

    rows_html = []
    for i, row in df_day.iterrows():
        row_idx = int(i)
        time_val = str(row['time_dk']).split(' ')[1][:5] if pd.notna(row['time_dk']) and ' ' in str(row['time_dk']) else str(row['time_dk'])
        qid = row.get('quarter', f"Q{row_idx+1}")
        q_label = f"{qid} ({time_val})"

        spot_val = float(row.get('spot_price_eur', 0.0))
        de_spot_val = float(row.get('de_spot_eur', spot_val))
        f_de = float(row.get('flow_de', row.get('scheduled_flow_mw', 0.0)))
        f_nord = float(row.get('flow_nordic', 0.0))
        f_no = f_nord * 0.7
        f_se = f_nord * 0.3
        f_gb = float(row.get('flow_gb', 0.0))
        f_nl = float(row.get('flow_nl', 0.0))
        f_sb = float(row.get('flow_great_belt', 0.0))

        # V3.2
        v32_dec = str(row.get('V3_2_Decision', '⚪ HOLD'))
        v32_spread = float(row.get('V3_2_Meta_Score', 0.0))
        v32_imb = spot_val + v32_spread
        v32_pnl = row.get('PnL_V3_2', np.nan)

        # V4.0
        v4_dec = str(row.get('V4_Decision', '⚪ HOLD'))
        v4_spread = float(row.get('V4_Predicted_Spread_EUR', 0.0))
        v4_imb = spot_val + v4_spread
        v4_pnl = row.get('PnL_V4_0', np.nan)

        # V4.1
        v41_dec = str(row.get('V4_1_Decision', '⚪ HOLD'))
        v41_spread = float(row.get('V4_1_Predicted_Spread_EUR', 0.0))
        v41_imb = spot_val + v41_spread
        v41_vol = float(row.get('V4_1_Volume_MW', 0.0))
        v41_mwh = float(row.get('V4_1_MWh', v41_vol / 4.0))
        v41_pnl = row.get('PnL_V4_1', np.nan)

        # Settled
        is_settled = bool(row.get('is_settled', False))
        settled_disp = str(row.get('actual_settled_display', '-- (Pending Delivery)'))
        settled_val = float(row.get('actual_settled_val', np.nan)) if pd.notna(row.get('actual_settled_val')) else np.nan

        # Badges
        dec32_class = "badge-buy" if "BUY" in v32_dec else ("badge-sell" if "SELL" in v32_dec else "badge-hold")
        dec4_class = "badge-buy" if "BUY" in v4_dec else ("badge-sell" if "SELL" in v4_dec else "badge-hold")
        dec41_class = "badge-buy" if "BUY" in v41_dec else ("badge-sell" if "SELL" in v41_dec else "badge-hold")
        status_badge = '<span class="badge badge-settled">🟢 Settled</span>' if is_settled else '<span class="badge badge-pending">🟡 Pending</span>'

        pnl32_text = f"€{v32_pnl:+,.2f}" if pd.notna(v32_pnl) else "--"
        pnl32_class = "pnl-pos" if (pd.notna(v32_pnl) and v32_pnl > 0) else ("pnl-neg" if (pd.notna(v32_pnl) and v32_pnl < 0) else "pnl-zero")

        pnl4_text = f"€{v4_pnl:+,.2f}" if pd.notna(v4_pnl) else "--"
        pnl4_class = "pnl-pos" if (pd.notna(v4_pnl) and v4_pnl > 0) else ("pnl-neg" if (pd.notna(v4_pnl) and v4_pnl < 0) else "pnl-zero")

        pnl41_text = f"€{v41_pnl:+,.2f}" if pd.notna(v41_pnl) else "--"
        pnl41_class = "pnl-pos" if (pd.notna(v41_pnl) and v41_pnl > 0) else ("pnl-neg" if (pd.notna(v41_pnl) and v41_pnl < 0) else "pnl-zero")

        # Cash flows
        if "BUY" in v41_dec:
            da_outlay = f"-€{spot_val * v41_mwh:,.2f}"
            settle_cf = f"+€{settled_val * v41_mwh:,.2f}" if is_settled and pd.notna(settled_val) else "Pending Gate Closure"
            gross = f"€{(settled_val - spot_val) * v41_mwh:+,.2f}" if is_settled and pd.notna(settled_val) else "Pending"
            fees = f"-€{v41_mwh * COST_EUR_MWH:,.2f}"
            net_cf = f"€{v41_pnl:+,.2f}" if is_settled and pd.notna(v41_pnl) else "Pending"
        elif "SELL" in v41_dec:
            da_outlay = f"+€{spot_val * v41_mwh:,.2f}"
            settle_cf = f"-€{settled_val * v41_mwh:,.2f}" if is_settled and pd.notna(settled_val) else "Pending Gate Closure"
            gross = f"€{(spot_val - settled_val) * v41_mwh:+,.2f}" if is_settled and pd.notna(settled_val) else "Pending"
            fees = f"-€{v41_mwh * COST_EUR_MWH:,.2f}"
            net_cf = f"€{v41_pnl:+,.2f}" if is_settled and pd.notna(v41_pnl) else "Pending"
        else:
            da_outlay, settle_cf, gross, fees, net_cf = "€0.00", "€0.00", "€0.00", "€0.00", "€0.00"

        zebra_class = "even-row" if row_idx % 2 == 0 else "odd-row"
        mfrr_up = float(row.get('mfrr_up_mw', 0.0))
        mfrr_dn = float(row.get('mfrr_down_mw', 0.0))
        smard_res = float(row.get('smard_residual_load_mw', 0.0))
        skew = float(row.get('order_flow_skew', 0.0))

        if only_v41:
            # Single Flat Table Row
            rows_html.append(f"""
            <tr class="master-row {zebra_class}" id="row-{row_idx}">
                <td style="font-weight:700; color:#0F172A;">{q_label}</td>
                <td>€{spot_val:.2f}</td>
                <td>€{de_spot_val:.2f}</td>
                <td style="font-weight:600;">{f_de:+.0f} MW</td>
                <td style="font-weight:600;">{f_no:+.0f} MW</td>
                <td style="font-weight:600;">{f_se:+.0f} MW</td>
                <td style="font-weight:600;">{f_gb:+.0f} MW</td>
                <td style="font-weight:600;">{f_nl:+.0f} MW</td>
                <td style="font-weight:600;">{f_sb:+.0f} MW</td>
                <td style="font-weight:700; color:#0D9488; background-color:#F0FDFA;">€{v41_imb:.2f}</td>
                <td style="background-color:#F0FDFA;">{v41_spread:+.2f} €</td>
                <td style="background-color:#F0FDFA;"><span class="badge {dec41_class}">{v41_dec}</span></td>
                <td style="background-color:#F0FDFA; font-weight:600;">{v41_vol:.0f} MW</td>
                <td style="background-color:#F0FDFA;">{v41_mwh:.2f} MWh</td>
                <td style="background-color:#F0FDFA;" class="{pnl41_class}">{pnl41_text}</td>
                <td>{mfrr_up:.0f} / {mfrr_dn:.0f}</td>
                <td>{smard_res:+,.0f} MW</td>
                <td>{skew:+.2f}</td>
                <td style="font-weight:500;">{da_outlay}</td>
                <td style="font-weight:500;">{settle_cf}</td>
                <td style="font-weight:500;">{gross}</td>
                <td style="color:#DC2626;">{fees}</td>
                <td style="font-weight:700; color:{'#16A34A' if pd.notna(v41_pnl) and v41_pnl > 0 else ('#DC2626' if pd.notna(v41_pnl) and v41_pnl < 0 else '#64748B')};">{net_cf}</td>
                <td style="font-weight:600;">{settled_disp}</td>
                <td>{status_badge}</td>
            </tr>
            """)
        else:
            # Multi-Model Comparative Master Row with Drawer
            v32_tds = f"""
                <td style="font-weight:600; color:#1E293B;">€{v32_imb:.2f}</td>
                <td><span class="badge {dec32_class}">{v32_dec}</span></td>
                <td class="{pnl32_class}">{pnl32_text}</td>
            """ if show_v32 else ""

            v40_tds = f"""
                <td style="font-weight:600; color:#0284C7;">€{v4_imb:.2f}</td>
                <td><span class="badge {dec4_class}">{v4_dec}</span></td>
                <td class="{pnl4_class}">{pnl4_text}</td>
            """ if show_v40 else ""

            v41_tds = f"""
                <td style="font-weight:700; color:#0D9488; background-color:#F0FDFA;">€{v41_imb:.2f}</td>
                <td style="background-color:#F0FDFA;"><span class="badge {dec41_class}">{v41_dec}</span></td>
                <td style="background-color:#F0FDFA; font-weight:600;">{v41_vol:.0f} MW</td>
                <td style="background-color:#F0FDFA;" class="{pnl41_class}">{pnl41_text}</td>
            """ if show_v41 else ""

            audit_rows = ""
            if show_v32:
                audit_rows += f'<tr><td><b>V3.2 Flow-Aware</b></td><td style="text-align:right;">€{v32_imb:.2f}</td><td style="text-align:center;">{v32_dec}</td><td style="text-align:right;">{pnl32_text}</td></tr>'
            if show_v40:
                audit_rows += f'<tr><td><b>V4.0 Full-Grid</b></td><td style="text-align:right;">€{v4_imb:.2f}</td><td style="text-align:center;">{v4_dec}</td><td style="text-align:right;">{pnl4_text}</td></tr>'
            if show_v41:
                audit_rows += f'<tr class="highlight-champ"><td><b>V4.1 High-Alpha</b></td><td style="text-align:right;">€{v41_imb:.2f}</td><td style="text-align:center;">{v41_dec}</td><td style="text-align:right;">{pnl41_text}</td></tr>'

            alpha_parts = []
            if show_v41 and show_v40:
                alpha_parts.append("vs V4.0: <b>" + (f"€{v41_pnl - v4_pnl:+,.2f}" if pd.notna(v41_pnl) and pd.notna(v4_pnl) else "--") + "</b>")
            if show_v41 and show_v32:
                alpha_parts.append("vs V3.2: <b>" + (f"€{v41_pnl - v32_pnl:+,.2f}" if pd.notna(v41_pnl) and pd.notna(v32_pnl) else "--") + "</b>")
            alpha_html = f'<div class="audit-notice"><b>V4.1 Alpha Outperformance:</b> {" | ".join(alpha_parts)}</div>' if alpha_parts else ""

            total_cols = 5 + (3 if show_v32 else 0) + (3 if show_v40 else 0) + (4 if show_v41 else 0) + 2

            rows_html.append(f"""
            <tr class="master-row {zebra_class}" onclick="toggleRow({row_idx})" id="row-{row_idx}">
                <td class="chevron-cell" onclick="event.stopPropagation(); toggleRow({row_idx});"><span class="chevron-icon" id="icon-{row_idx}">&#9654;</span></td>
                <td style="font-weight:700; color:#0F172A;">{q_label}</td>
                <td>€{spot_val:.2f}</td>
                <td>€{de_spot_val:.2f}</td>
                <td style="font-weight:600;">{f_de:+.0f} MW</td>
                {v32_tds}
                {v40_tds}
                {v41_tds}
                <td style="font-weight:600;">{settled_disp}</td>
                <td>{status_badge}</td>
            </tr>
            <tr class="drawer-row" id="drawer-{row_idx}" style="display: none;">
                <td colspan="{total_cols}" class="drawer-cell">
                    <div class="drawer-banner">
                        <span>⚡ <b>AUDIT BREAKDOWN:</b> {q_label} &mdash; Multi-Generation Comparative Audit</span>
                        <span><b>Delivery:</b> {time_val} CEST &bull; <b>MARI / PICASSO / SMARD / XBID Grounded</b></span>
                    </div>
                    <div class="drawer-cards-grid">
                        <div class="drawer-card">
                            <h5>🌐 Layer 4 Balancing & Cross-Border Telemetry</h5>
                            <table class="sub-table">
                                <thead>
                                    <tr><th>Metric / Border</th><th style="text-align:right;">Live Value</th><th style="text-align:right;">State</th></tr>
                                </thead>
                                <tbody>
                                    <tr><td><b>MARI mFRR Up</b></td><td style="text-align:right;">{mfrr_up:.0f} MW</td><td style="text-align:right;">Merit Order</td></tr>
                                    <tr><td><b>MARI mFRR Down</b></td><td style="text-align:right;">{mfrr_dn:.0f} MW</td><td style="text-align:right;">Merit Order</td></tr>
                                    <tr><td><b>🇩🇪 SMARD Residual Load</b></td><td style="text-align:right;">{smard_res:+,.0f} MW</td><td style="text-align:right;">DE System</td></tr>
                                    <tr><td><b>XBID Order Flow Skew</b></td><td style="text-align:right;">{skew:+.2f}</td><td style="text-align:right;">{'Buy Skew' if skew > 0 else 'Sell Skew'}</td></tr>
                                    <tr><td><b>DE ➔ DK Physical Flow</b></td><td style="text-align:right;">{f_de:+.0f} MW</td><td style="text-align:right;">Kassø Lines</td></tr>
                                    <tr><td><b>NO ➔ DK Physical Flow</b></td><td style="text-align:right;">{f_no:+.0f} MW</td><td style="text-align:right;">Skagerrak 1-4</td></tr>
                                    <tr><td><b>SE ➔ DK Physical Flow</b></td><td style="text-align:right;">{f_se:+.0f} MW</td><td style="text-align:right;">Konti-Skan / Øresund</td></tr>
                                    <tr><td><b>GB ➔ DK Physical Flow</b></td><td style="text-align:right;">{f_gb:+.0f} MW</td><td style="text-align:right;">Viking Link</td></tr>
                                    <tr><td><b>NL ➔ DK Physical Flow</b></td><td style="text-align:right;">{f_nl:+.0f} MW</td><td style="text-align:right;">COBRAcable</td></tr>
                                    <tr><td><b>DK1 &harr; DK2 Great Belt</b></td><td style="text-align:right;">{f_sb:+.0f} MW</td><td style="text-align:right;">Storebælt HVDC</td></tr>
                                </tbody>
                            </table>
                        </div>
                        <div class="drawer-card">
                            <h5>🧠 Comparative Model Audit</h5>
                            <table class="sub-table">
                                <thead>
                                    <tr><th>Model</th><th style="text-align:right;">Pred Imb</th><th style="text-align:center;">Decision</th><th style="text-align:right;">PnL (€)</th></tr>
                                </thead>
                                <tbody>
                                    {audit_rows}
                                </tbody>
                            </table>
                            {alpha_html}
                        </div>
                        <div class="drawer-card">
                            <h5>💰 Commercial Cash Flow & PnL Ledger</h5>
                            <div class="cf-row"><span>Day-Ahead Outlay / Cash Flow:</span><b>{da_outlay}</b></div>
                            <div class="cf-row"><span>Real-Time Settlement Cash Flow:</span><b>{settle_cf}</b></div>
                            <div class="cf-row"><span>Gross Realized Trading PnL:</span><b>{gross}</b></div>
                            <div class="cf-row"><span>Exchange Fees (€{COST_EUR_MWH:.2f}/MWh):</span><b style="color:#DC2626;">{fees}</b></div>
                            <div class="cf-row total"><span>Net Realized PnL:</span><b style="font-size:13px; color:{'#16A34A' if pd.notna(v41_pnl) and v41_pnl > 0 else ('#DC2626' if pd.notna(v41_pnl) and v41_pnl < 0 else '#64748B')};">{net_cf}</b></div>
                        </div>
                    </div>
                </td>
            </tr>
            """)

    table_body = "\n".join(rows_html)

    if only_v41:
        headers_html = """
          <tr>
            <th draggable="true" title="Trading Quarter & Delivery Time"><div class="col-header-wrap"><span class="col-title">Quarter</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="DK Day-Ahead Spot Price"><div class="col-header-wrap"><span class="col-title">DK Spot</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="German Day-Ahead Spot Price"><div class="col-header-wrap"><span class="col-title">DE Spot</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="DE to DK Scheduled Exchange Flow"><div class="col-header-wrap"><span class="col-title">DE➔DK</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Norway (NO2) to DK Scheduled Flow — Skagerrak 1-4"><div class="col-header-wrap"><span class="col-title">NO➔DK</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Sweden (SE3/SE4) to DK Scheduled Flow — Konti-Skan / Øresund"><div class="col-header-wrap"><span class="col-title">SE➔DK</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Great Britain to DK Scheduled Flow — Viking Link"><div class="col-header-wrap"><span class="col-title">GB➔DK</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Netherlands to DK Scheduled Flow — COBRAcable"><div class="col-header-wrap"><span class="col-title">NL➔DK</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="DK1 <-> DK2 Great Belt HVDC Flow"><div class="col-header-wrap"><span class="col-title">Storebælt</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            
            <th draggable="true" title="V4.1 Predicted Imbalance Price (€)" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Pred</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.1 Predicted Spread vs DK Spot (€/MWh)" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Spread</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.1 Institutional Trading Decision" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Pos</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.1 Position Size (MW)" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Vol</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.1 Traded Energy per Quarter (MWh)" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 MWh</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.1 Realized Trading PnL (€)" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 PnL</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            
            <th draggable="true" title="MARI mFRR Net Up / Down Activations (MW)"><div class="col-header-wrap"><span class="col-title">mFRR (Up/Dn)</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="SMARD German Residual Load (MW)"><div class="col-header-wrap"><span class="col-title">DE Res Load</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Continuous XBID Order Flow Skew"><div class="col-header-wrap"><span class="col-title">XBID Skew</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Day-Ahead Outlay Cash Flow"><div class="col-header-wrap"><span class="col-title">DA Outlay</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Real-Time Settlement Cash Flow"><div class="col-header-wrap"><span class="col-title">RT Settle</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Gross Realized Trading PnL (before fees)"><div class="col-header-wrap"><span class="col-title">Gross PnL</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Exchange Fees"><div class="col-header-wrap"><span class="col-title">Fees</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Net Realized Trading PnL"><div class="col-header-wrap"><span class="col-title">Net PnL</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            
            <th draggable="true" title="Authentic Energinet Settled Price (€)"><div class="col-header-wrap"><span class="col-title">Settled Imb</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Settlement Status"><div class="col-header-wrap"><span class="col-title">Status</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
          </tr>
        """
        toolbar_buttons = ""
    else:
        v32_th = """
            <th draggable="true" title="V3.2 Predicted Imbalance Price (€) [Spot + Spread]" style="background-color:#334155;"><div class="col-header-wrap"><span class="col-title">V3.2 Pred</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V3.2 Meta Model Trading Decision" style="background-color:#334155;"><div class="col-header-wrap"><span class="col-title">V3.2 Pos</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V3.2 Realized Trading PnL (€)" style="background-color:#334155;"><div class="col-header-wrap"><span class="col-title">V3.2 PnL</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
        """ if show_v32 else ""

        v40_th = """
            <th draggable="true" title="V4.0 Predicted Imbalance Price (€) [Spot + Spread]" style="background-color:#0284C7;"><div class="col-header-wrap"><span class="col-title">V4.0 Pred</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.0 Champion Trading Decision" style="background-color:#0284C7;"><div class="col-header-wrap"><span class="col-title">V4.0 Pos</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.0 Realized Trading PnL (€)" style="background-color:#0284C7;"><div class="col-header-wrap"><span class="col-title">V4.0 PnL</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
        """ if show_v40 else ""

        v41_th = """
            <th draggable="true" title="V4.1 Predicted Imbalance Price (€) [Spot + Spread]" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Pred</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.1 Institutional Trading Decision" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Pos</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.1 Position Size with Conviction Scaling" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Vol</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="V4.1 Realized Trading PnL (€)" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 PnL</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
        """ if show_v41 else ""

        headers_html = f"""
          <tr>
            <th class="no-drag" style="width:20px; padding:7px 0;"></th>
            <th draggable="true" title="Trading Quarter & Delivery Time"><div class="col-header-wrap"><span class="col-title">Quarter</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="DK Day-Ahead Spot Price"><div class="col-header-wrap"><span class="col-title">DK Spot</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="German Day-Ahead Spot Price"><div class="col-header-wrap"><span class="col-title">DE Spot</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="DE to DK Scheduled Exchange Flow"><div class="col-header-wrap"><span class="col-title">DE➔DK</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            {v32_th}
            {v40_th}
            {v41_th}
            <th draggable="true" title="Authentic Energinet Settled Price (€)"><div class="col-header-wrap"><span class="col-title">Settled Imb</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
            <th draggable="true" title="Settlement Status"><div class="col-header-wrap"><span class="col-title">Status</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
          </tr>
        """
        toolbar_buttons = """
            <div style="display:flex; gap:8px;">
              <button type="button" class="btn-action" onclick="expandAll()">&#9660; Expand All 96 Quarters</button>
              <button type="button" class="btn-action" style="background-color:#475569;" onclick="collapseAll()">&#9654; Collapse All</button>
            </div>
        """

    accordion_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    {LEDGER_CSS}
    </head>
    <body>
      <div class="toolbar-container">
        {toolbar_buttons}
        <div style="display:flex; gap:6px; align-items:center; margin-left:auto;">
          <button type="button" class="btn-action" id="btnCopySelected" style="background-color:#0D9488;" onclick="copySelectedColumns()">📋 Copy Selected</button>
          <input type="text" id="filterInput" class="search-input" placeholder="🔍 Search Quarter / Action / Time..." onkeyup="filterQuarters()">
        </div>
      </div>
      <div id="copyToast">✅ Copied to clipboard!</div>
      <div class="table-wrapper">
        <table class="master-table" id="masterTable">
          <thead>
            {headers_html}
          </thead>
          <tbody>
            {table_body}
          </tbody>
        </table>
      </div>
      {LEDGER_JS}
    </body>
    </html>
    """

    components.html(accordion_html, height=800, scrolling=True)
