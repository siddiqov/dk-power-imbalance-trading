# ==============================================================================
# generate_decision_feature_whitepaper.py
# Generates a publication-grade Technical Reference PDF on:
# 1. Two-Tier Decision Hierarchy & Spread Priority (BUY vs SELL vs HOLD)
# 2. Tri-State Softmax Direction Probabilities & Step-by-Step Math Derivation
# 3. Complete 14-Dimensional Feature Vector (x in R^14) & Data Provenance (Energi Data Service)
# ==============================================================================

import os
import sys
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers and running headers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "V3 Optimeering Energy Trader — Decision Hierarchy, Softmax & 14-D Telemetry")
            self.drawRightString(612 - 54, 750, "Technical Reference Guide")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)

        # Running Footer (All Pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        
        self.drawString(54, 32, f"Confidential & Proprietary — Nurex Trading & Energinet DK1/DK2 Power Desk — {datetime.now().strftime('%Y-%m-%d')}")
        self.drawRightString(612 - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf(filename="results/V3_Decision_Hierarchy_and_14D_Features_Guide.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=64,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    c_primary = colors.HexColor("#0f172a")     # Slate 900
    c_secondary = colors.HexColor("#1e293b")   # Slate 800
    c_accent = colors.HexColor("#0284c7")      # Sky 600
    c_green = colors.HexColor("#059669")       # Emerald 600
    c_red = colors.HexColor("#dc2626")         # Red 600
    c_border = colors.HexColor("#e2e8f0")      # Slate 200
    c_bg_light = colors.HexColor("#f8fafc")    # Slate 50

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=c_secondary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13.5,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )
    formula_style = ParagraphStyle(
        'Formula_Custom',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=5,
        spaceAfter=6
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#334155")
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a")
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # TITLE & HEADER
    # =========================================================================
    story.append(Paragraph("V3 Optimeering Energy Trading System", title_style))
    story.append(Paragraph("<b>Technical Reference: Decision Hierarchy, Softmax Probabilities & 14-Dimensional Telemetry</b>", ParagraphStyle('Sub', parent=title_style, fontSize=12, leading=15, textColor=c_accent)))
    story.append(Paragraph(f"<b>Classification:</b> Proprietary & Confidential | <b>Target Markets:</b> Energinet DK1 (Jutland) & DK2 (Zealand) | <b>Published:</b> {datetime.now().strftime('%d %B %Y')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceAfter=10))

    # =========================================================================
    # SECTION 1: TWO-TIER DECISION HIERARCHY
    # =========================================================================
    story.append(Paragraph("1. Two-Tier Trading Decision Hierarchy & Spread Priority", h1_style))
    story.append(Paragraph(
        "In the V3 algorithmic trading engine, trading actions (<b>BUY Spot / Long</b>, <b>SELL Spot / Short</b>, and <b>HOLD</b>) "
        "are governed by a strict, deterministic quantitative hierarchy. The <b>Predicted Spread</b> has 100% primary decision priority, "
        "while the <b>Direction Probabilities P(Up) / P(Dn)</b> serve as secondary risk shields and dynamic volume multipliers.",
        body_style
    ))

    story.append(Paragraph("Core Spread Mathematical Formulation", h2_style))
    story.append(Paragraph(
        "Spread Equation:  S_pred(t) = P_pred_imbalance(t) - P_spot(t)<br/>"
        "• If S_pred(t) > +1.20 EUR/MWh  --> Trigger BUY Spot (Long) [System expects deficit]<br/>"
        "• If S_pred(t) < -1.20 EUR/MWh  --> Trigger SELL Spot (Short) [System expects surplus]<br/>"
        "• If |S_pred(t)| <= 1.20 EUR/MWh --> Trigger HOLD [Balanced / Neutral band]",
        formula_style
    ))

    # Decision Priority Table
    dec_data = [
        [Paragraph("Tier Level", table_header), Paragraph("Quantitative Component", table_header), Paragraph("Role in Trade Execution", table_header), Paragraph("Priority Level", table_header)],
        [Paragraph("<b>Tier 1</b>", table_cell_bold), Paragraph("<b>Predicted Spread (S)</b><br/>P_imb - P_spot", table_cell), Paragraph("<b>100% Primary Decision Trigger.</b> Directly dictates BUY, SELL, or HOLD based on expected arbitrage profitability.", table_cell), Paragraph("<font color='#059669'><b>First Priority (100%)</b></font>", table_cell)],
        [Paragraph("<b>Tier 2</b>", table_cell_bold), Paragraph("<b>Direction Probabilities</b><br/>P(Up) / P(Dn) Softmax", table_cell), Paragraph("<b>Secondary Risk & Sizing Modifier.</b> Scales volume (1.0 to 5.0 MW) and triggers Spike Shield vetoes against short squeezes.", table_cell), Paragraph("<font color='#0284c7'><b>Secondary Modifier</b></font>", table_cell)]
    ]
    t_dec = Table(dec_data, colWidths=[60, 120, 220, 104])
    t_dec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_dec)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 2: SOFTMAX PROBABILITIES & NUMERICAL DERIVATION
    # =========================================================================
    story.append(Paragraph("2. Tri-State Direction Probabilities & Softmax Derivation", h1_style))
    story.append(Paragraph(
        "When the dashboard displays a ratio such as <b>14.1% Up / 50.6% Dn</b> (or <b>0.2% Up / 0.4% Dn</b>), it represents the two active directional "
        "tails of a <b>Tri-State Softmax Probability Distribution</b>. The remaining probability mass resides in the <b>Balanced / Neutral</b> state:",
        body_style
    ))

    story.append(Paragraph(
        "Total Probability Identity:  P(Down) + P(Balanced) + P(Up) = 100.0%<br/>"
        "Example (14.1% / 50.6%):     50.6% Down + 35.3% Balanced + 14.1% Up = 100.0%<br/>"
        "Example (0.2% / 0.4%):       0.4% Down + 99.4% Balanced + 0.2% Up = 100.0%",
        formula_style
    ))

    story.append(Paragraph("Step-by-Step Softmax Mathematical Derivation", h2_style))
    story.append(Paragraph(
        "The model computes unnormalized linear/tree logits z = [z_down, z_balanced, z_up]^T = W * x + b from the 14-D feature vector. "
        "The normalized probabilities are derived via the Softmax operator:",
        body_style
    ))
    story.append(Paragraph(
        "Softmax Formula:  P(k) = exp(z_k) / sum(exp(z_j)) for j in {Down, Balanced, Up}<br/><br/>"
        "Given Raw Logits:  z_down = +0.85,  z_balanced = +0.49,  z_up = -0.43<br/>"
        "1. Exponentials:   exp(+0.85) = 2.3396 | exp(+0.49) = 1.6323 | exp(-0.43) = 0.6505<br/>"
        "2. Denominator:    Sum = 2.3396 + 1.6323 + 0.6505 = 4.6224<br/>"
        "3. Probabilities:  P(Down)     = 2.3396 / 4.6224 = 50.61% --> 50.6%<br/>"
        "                   P(Balanced) = 1.6323 / 4.6224 = 35.31% --> 35.3%<br/>"
        "                   P(Up)       = 0.6505 / 4.6224 = 14.07% --> 14.1%",
        formula_style
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 3: THE 14-DIMENSIONAL FEATURE VECTOR
    # =========================================================================
    story.append(Paragraph("3. The 14-Dimensional Telemetry Feature Vector (x in R^14)", h1_style))
    story.append(Paragraph(
        "The complete input feature vector <b>x = [x1, x2, ..., x14]^T</b> encapsulates real-time physical grid telemetry, "
        "autoregressive price settlement lags, momentum derivatives, and market structure anchors. Every value originates strictly from "
        "authentic Danish Energinet and Nord Pool data streams.",
        body_style
    ))

    feat_data = [
        [Paragraph("#", table_header), Paragraph("Feature Name (Code)", table_header), Paragraph("Formula / Symbol", table_header), Paragraph("Sample Value", table_header), Paragraph("Data Source & Physical Meaning", table_header)],
        [Paragraph("x1", table_cell_bold), Paragraph("wind_forecast_error_mw", table_cell_bold), Paragraph("Wind_act - Wind_fcst", table_cell), Paragraph("+185.4 MW", table_cell), Paragraph("<b>Energi Data Service:</b> Live wind generation vs D-1 forecast. +185 MW excess pushes prices down.", table_cell)],
        [Paragraph("x2", table_cell_bold), Paragraph("solar_forecast_error_mw", table_cell_bold), Paragraph("Solar_act - Solar_fcst", table_cell), Paragraph("+22.1 MW", table_cell), Paragraph("<b>Energi Data Service:</b> Live solar actuals vs forecast.", table_cell)],
        [Paragraph("x3", table_cell_bold), Paragraph("imb_spread_lag_1", table_cell_bold), Paragraph("S(t-1) = P_imb - P_spot", table_cell), Paragraph("-4.50 EUR", table_cell), Paragraph("<b>ImbalancePrice API:</b> Settlement spread from t-15m.", table_cell)],
        [Paragraph("x4", table_cell_bold), Paragraph("imb_spread_lag_2", table_cell_bold), Paragraph("S(t-2)", table_cell), Paragraph("-2.80 EUR", table_cell), Paragraph("<b>ImbalancePrice API:</b> Settlement spread from t-30m.", table_cell)],
        [Paragraph("x5", table_cell_bold), Paragraph("imb_spread_lag_4", table_cell_bold), Paragraph("S(t-4)", table_cell), Paragraph("-1.20 EUR", table_cell), Paragraph("<b>ImbalancePrice API:</b> Settlement spread from t-60m (1h ago).", table_cell)],
        [Paragraph("x6", table_cell_bold), Paragraph("imb_velocity_1q", table_cell_bold), Paragraph("dS / dt = S(t-1) - S(t-2)", table_cell), Paragraph("-1.70 EUR/15m", table_cell), Paragraph("<b>First Derivative:</b> Rate of spread change per 15 minutes.", table_cell)],
        [Paragraph("x7", table_cell_bold), Paragraph("imb_acceleration_1q", table_cell_bold), Paragraph("dVel / dt = Vel(t-1) - Vel(t-2)", table_cell), Paragraph("-0.60 EUR/15m", table_cell), Paragraph("<b>Second Derivative:</b> Momentum shift in grid imbalance.", table_cell)],
        [Paragraph("x8", table_cell_bold), Paragraph("imb_roll_std_4q", table_cell_bold), Paragraph("StdDev(S_{t-1:t-4})", table_cell), Paragraph("3.85 EUR/MWh", table_cell), Paragraph("<b>4-Quarter Volatility:</b> Rolling 1-hour spread standard deviation.", table_cell)],
        [Paragraph("x9", table_cell_bold), Paragraph("reg_persistence_quarters", table_cell_bold), Paragraph("Streak_length * Dir", table_cell), Paragraph("-3 (Dn-3)", table_cell), Paragraph("<b>Directional Streak:</b> 3 consecutive quarters of Down-Reg.", table_cell)],
        [Paragraph("x10", table_cell_bold), Paragraph("spot_price_eur", table_cell_bold), Paragraph("P_spot(t)", table_cell), Paragraph("180.50 EUR/MWh", table_cell), Paragraph("<b>DayAheadPrices API:</b> Fixed Nord Pool D-1 auction price.", table_cell)],
        [Paragraph("x11", table_cell_bold), Paragraph("spot_squared", table_cell_bold), Paragraph("sign(Spot) * (Spot^2 / 1000)", table_cell), Paragraph("32.58", table_cell), Paragraph("<b>Nonlinear Merit-Order:</b> Captures extreme spot price convexity.", table_cell)],
        [Paragraph("x12", table_cell_bold), Paragraph("spread_dk_de", table_cell_bold), Paragraph("Spot(DK) - Spot(DE)", table_cell), Paragraph("+8.20 EUR/MWh", table_cell), Paragraph("<b>Cross-Border Arbitrage:</b> DK spot spread against Germany.", table_cell)],
        [Paragraph("x13", table_cell_bold), Paragraph("sin_hour / sin_quarter", table_cell_bold), Paragraph("sin(2*pi*Hour / 24)", table_cell), Paragraph("-0.707", table_cell), Paragraph("<b>Fourier Harmonic:</b> Encodes diurnal consumption & solar cycle.", table_cell)],
        [Paragraph("x14", table_cell_bold), Paragraph("is_peak_hour", table_cell_bold), Paragraph("1 if 08:00-20:00 Wkday else 0", table_cell), Paragraph("0 (Off-Peak)", table_cell), Paragraph("<b>Regime Flag:</b> High-demand working hours vs baseload.", table_cell)]
    ]
    t_feat = Table(feat_data, colWidths=[20, 120, 105, 65, 194])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg_light]),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 4: DATA PROVENANCE & ASSEMBLY WORKFLOW
    # =========================================================================
    story.append(Paragraph("4. Real-Time Assembly Workflow & Zero-Leakage Guarantee", h1_style))
    story.append(Paragraph(
        "At every 15-minute clock boundary (e.g. 14:00, 14:15, 14:30), the feature engineering engine assembles <b>x</b> in three phases:<br/>"
        "1. <b>D-1 Static Anchors (x10, x11, x12, x13, x14):</b> Locked at D-1 12:45 CET from the <code>DayAheadPrices</code> dataset.<br/>"
        "2. <b>Dynamic 15m Telemetry (x3, x4, x5, x6, x7, x8, x9):</b> Ingested as Energinet publishes official settlements in <code>ImbalancePrice</code>.<br/>"
        "3. <b>Renewable Delta Ingestion (x1, x2):</b> Live wind/solar forecast errors calculated from <code>Forecasts_Hour</code>.<br/>"
        "4. <b>Inference & Execution:</b> The 14-D vector is evaluated by Transformer-TFT, LightGBM, and Meta-Ensemble models to generate multi-quantile spread bounds (q10, q50, q90) and lock immutable orders before gate closure.",
        body_style
    ))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated Technical Whitepaper: {filename}")


if __name__ == '__main__':
    out_file = sys.argv[1] if len(sys.argv) > 1 else "results/V3_Decision_Hierarchy_and_14D_Features_Guide.pdf"
    build_pdf(out_file)
