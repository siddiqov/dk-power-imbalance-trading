import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def generate_infographic():
    # 16:9 Canvas (2400 x 1350 px at 150 dpi -> 16 x 9 inches)
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Background gradient or dark luxury slate
    bg = patches.Rectangle((0, 0), 16, 9, facecolor='#0B111E', zorder=0)
    ax.add_patch(bg)

    font_main = 'Segoe UI'

    # -------------------------------------------------------------
    # HEADER SECTION
    # -------------------------------------------------------------
    # Header bar background
    header_bar = patches.FancyBboxPatch((0.5, 7.7), 15.0, 1.0, 
                                        boxstyle="round,pad=0.08,rounding_size=0.15",
                                        facecolor='#151F32', edgecolor='#23334E', linewidth=1.5, zorder=1)
    ax.add_patch(header_bar)

    ax.text(0.8, 8.35, "NUREX TRADING SYSTEM  |  COMMERCIAL-READY MODELS V1", 
            fontsize=18, fontweight='bold', color='#FFFFFF', fontfamily=font_main, va='center')
    ax.text(0.8, 7.95, "Danish Power Market (DK1 & DK2 Imbalance)  •  12-Day Continuous Shadow Performance Audit", 
            fontsize=11, color='#94A3B8', fontfamily=font_main, va='center')

    # Drive sync pill badge
    pill = patches.FancyBboxPatch((10.3, 7.88), 4.9, 0.65, 
                                 boxstyle="round,pad=0.05,rounding_size=0.12",
                                 facecolor='#064E3B', edgecolor='#10B981', linewidth=1.2, zorder=2)
    ax.add_patch(pill)
    ax.text(12.75, 8.28, "SYNCED TO GOOGLE DRIVE", 
            fontsize=9.5, fontweight='bold', color='#34D399', fontfamily=font_main, ha='center', va='center')
    ax.text(12.75, 8.04, "G:\\My Drive\\commercial ready models\\v1", 
            fontsize=8.5, color='#A7F3D0', fontfamily='Consolas', ha='center', va='center')

    # -------------------------------------------------------------
    # TOP KPI CARDS (4 CARDS)
    # -------------------------------------------------------------
    kpis = [
        {
            "label": "COMBINED RUN RATE",
            "val": "EUR 1,490 / day",
            "sub": "Goal: EUR 1,500/day (99.3% Reached)",
            "color": "#10B981",
            "bg": "#0D281E",
            "border": "#059669"
        },
        {
            "label": "DK1 CHAMPION NET P&L",
            "val": "+EUR 17,465",
            "sub": "312 Trades • 638.8 MWh • EUR 27.3/MWh",
            "color": "#38BDF8",
            "bg": "#0C2738",
            "border": "#0284C7"
        },
        {
            "label": "DK2 CHAMPION NET P&L",
            "val": "+EUR 415",
            "sub": "69% Win Rate • Max Drawdown -EUR 630",
            "color": "#A78BFA",
            "bg": "#1E1A38",
            "border": "#7C3AED"
        },
        {
            "label": "EVALUATION DATASET",
            "val": "1,072 Quarters",
            "sub": "Real Intraday VWAP Fills • After Friction",
            "color": "#FBBF24",
            "bg": "#2A2413",
            "border": "#D97706"
        }
    ]

    card_w = 3.55
    card_gap = 0.27
    start_x = 0.5
    for i, kpi in enumerate(kpis):
        cx = start_x + i * (card_w + card_gap)
        card = patches.FancyBboxPatch((cx, 6.35), card_w, 1.15,
                                      boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=kpi['bg'], edgecolor=kpi['border'], linewidth=1.2, zorder=1)
        ax.add_patch(card)
        ax.text(cx + 0.2, 7.22, kpi['label'], fontsize=9, fontweight='bold', color=kpi['color'], fontfamily=font_main)
        ax.text(cx + 0.2, 6.80, kpi['val'], fontsize=16, fontweight='bold', color='#FFFFFF', fontfamily=font_main)
        ax.text(cx + 0.2, 6.50, kpi['sub'], fontsize=8.2, color='#94A3B8', fontfamily=font_main)

    # -------------------------------------------------------------
    # MAIN COLUMNS: DK1 ZONE (LEFT) vs DK2 ZONE (RIGHT)
    # -------------------------------------------------------------
    col_w = 7.32
    col_y = 2.15
    col_h = 3.98

    # --- DK1 COLUMN ---
    dk1_box = patches.FancyBboxPatch((0.5, col_y), col_w, col_h,
                                     boxstyle="round,pad=0.08,rounding_size=0.15",
                                     facecolor='#111A2E', edgecolor='#23334E', linewidth=1.5, zorder=1)
    ax.add_patch(dk1_box)

    # DK1 Header
    ax.text(0.8, 5.82, "DK1 ZONE (West Denmark / Continental Sync)", fontsize=13, fontweight='bold', color='#38BDF8', fontfamily=font_main)
    
    # Champion Card inside DK1
    champ_dk1 = patches.FancyBboxPatch((0.75, 3.75), col_w - 0.5, 1.80,
                                       boxstyle="round,pad=0.05,rounding_size=0.1",
                                       facecolor='#152840', edgecolor='#0284C7', linewidth=1.5, zorder=2)
    ax.add_patch(champ_dk1)
    
    # Ready Badge
    badge_dk1 = patches.FancyBboxPatch((6.0, 5.15), 1.6, 0.32,
                                       boxstyle="round,pad=0.03,rounding_size=0.06",
                                       facecolor='#065F46', edgecolor='#10B981', linewidth=1, zorder=3)
    ax.add_patch(badge_dk1)
    ax.text(6.8, 5.31, "COMMERCIAL CHAMPION", fontsize=7.5, fontweight='bold', color='#6EE7B7', fontfamily=font_main, ha='center', va='center')

    ax.text(0.95, 5.25, "DK1 Shadow Test (ID-target + Conformal)", fontsize=12, fontweight='bold', color='#FFFFFF', fontfamily=font_main)
    ax.text(0.95, 4.95, "Architecture: LightGBM ID-Target Classifier with Conformal Uncertainty Intervals", fontsize=8.5, color='#93C5FD', fontfamily=font_main)
    
    # Metric table rows
    metrics_text_dk1 = (
        "• Net P&L: +EUR 17,465  |  EUR 27.30 / MWh  |  Trades: 312 (283 BUY / 29 SELL)\n"
        "• Win Rate: 55.0%  |  Volume: 638.8 MWh  |  Max Drawdown: -EUR 2,947\n"
        "• Why it Wins: Conformal bounds filter out low-confidence signals and systematically\n"
        "  captures wind deficit imbalance spikes across Danish intraday contracts."
    )
    ax.text(0.95, 4.48, metrics_text_dk1, fontsize=8.2, color='#E2E8F0', fontfamily=font_main, linespacing=1.35)

    # Secondary / Baseline inside DK1
    sec_dk1 = patches.FancyBboxPatch((0.75, 2.35), col_w - 0.5, 1.25,
                                     boxstyle="round,pad=0.05,rounding_size=0.08",
                                     facecolor='#131D2D', edgecolor='#1E293B', linewidth=1.0, zorder=2)
    ax.add_patch(sec_dk1)
    ax.text(0.95, 3.32, "DK1 Challenger: Combo Model (IDT + Conformal + Chronos-2)", fontsize=9.5, fontweight='bold', color='#94A3B8', fontfamily=font_main)
    ax.text(0.95, 2.85, 
            "• Net P&L: +EUR 5,422  |  EUR 24.60 / MWh  |  Max Drawdown: -EUR 967\n"
            "• Role: Outstanding low-volatility backup model with ultra-tight drawdown profile.", 
            fontsize=8.0, color='#64748B', fontfamily=font_main, linespacing=1.3)


    # --- DK2 COLUMN ---
    dk2_x = 8.18
    dk2_box = patches.FancyBboxPatch((dk2_x, col_y), col_w, col_h,
                                     boxstyle="round,pad=0.08,rounding_size=0.15",
                                     facecolor='#111A2E', edgecolor='#23334E', linewidth=1.5, zorder=1)
    ax.add_patch(dk2_box)

    # DK2 Header
    ax.text(dk2_x + 0.3, 5.82, "DK2 ZONE (East Denmark / Nordic Sync)", fontsize=13, fontweight='bold', color='#A78BFA', fontfamily=font_main)

    # Champion Card inside DK2
    champ_dk2 = patches.FancyBboxPatch((dk2_x + 0.25, 3.75), col_w - 0.5, 1.80,
                                       boxstyle="round,pad=0.05,rounding_size=0.1",
                                       facecolor='#241B3B', edgecolor='#7C3AED', linewidth=1.5, zorder=2)
    ax.add_patch(champ_dk2)

    # Ready Badge
    badge_dk2 = patches.FancyBboxPatch((dk2_x + 5.5, 5.15), 1.6, 0.32,
                                       boxstyle="round,pad=0.03,rounding_size=0.06",
                                       facecolor='#065F46', edgecolor='#10B981', linewidth=1, zorder=3)
    ax.add_patch(badge_dk2)
    ax.text(dk2_x + 6.3, 5.31, "COMMERCIAL CHAMPION", fontsize=7.5, fontweight='bold', color='#6EE7B7', fontfamily=font_main, ha='center', va='center')

    ax.text(dk2_x + 0.45, 5.25, "DK2 Chronos-2 Alone", fontsize=12, fontweight='bold', color='#FFFFFF', fontfamily=font_main)
    ax.text(dk2_x + 0.45, 4.95, "Architecture: Time-Series Foundation Model (Probabilistic Quantile Forecasts)", fontsize=8.5, color='#DDD6FE', fontfamily=font_main)

    metrics_text_dk2 = (
        "• Net P&L: +EUR 415  |  Highest Win Rate: 69.0%  |  Trades: 131 (55 BUY / 76 SELL)\n"
        "• Lowest Drawdown: -EUR 630  |  Volume: 100.4 MWh  |  Beats Live Baseline (+EUR 145)\n"
        "• Why it Wins: Accurately captures Nordic/Swedish interconnector mean-reversion swings;\n"
        "  exceptionally reliable risk profile with zero blown quarters."
    )
    ax.text(dk2_x + 0.45, 4.48, metrics_text_dk2, fontsize=8.2, color='#E2E8F0', fontfamily=font_main, linespacing=1.35)

    # Research / Blocked Card inside DK2
    sec_dk2 = patches.FancyBboxPatch((dk2_x + 0.25, 2.35), col_w - 0.5, 1.25,
                                     boxstyle="round,pad=0.05,rounding_size=0.08",
                                     facecolor='#261519', edgecolor='#991B1B', linewidth=1.0, zorder=2)
    ax.add_patch(sec_dk2)
    ax.text(dk2_x + 0.45, 3.32, "DK2 Research Checkpoint: Combo Model (DO NOT DEPLOY)", fontsize=9.5, fontweight='bold', color='#F87171', fontfamily=font_main)
    ax.text(dk2_x + 0.45, 2.85, 
            "• Net P&L: -EUR 582  |  Last 3 Days: -EUR 1,424  |  Max Drawdown: -EUR 1,967\n"
            "• Reason to Hold: Overtrades on BUY side during oversupply regimes. Held for re-tuning.", 
            fontsize=8.0, color='#FCA5A5', fontfamily=font_main, linespacing=1.3)

    # -------------------------------------------------------------
    # BOTTOM ACTION & STRUCTURE STRIP
    # -------------------------------------------------------------
    bot_box = patches.FancyBboxPatch((0.5, 0.4), 15.0, 1.55,
                                     boxstyle="round,pad=0.08,rounding_size=0.15",
                                     facecolor='#151F32', edgecolor='#23334E', linewidth=1.5, zorder=1)
    ax.add_patch(bot_box)

    # Left: Folder Structure
    ax.text(0.8, 1.65, "DIRECTORY ARTIFACTS IN GOOGLE DRIVE (G:\\My Drive\\commercial ready models\\v1\\)", 
            fontsize=10.5, fontweight='bold', color='#FFFFFF', fontfamily=font_main)
    
    file_tree = (
        "├── models/           DK1_shadow_idt_champion.pkl (+EUR 17.5k), DK2_combo_model.pkl, DK1_v4_1_champion.pkl, features\n"
        "├── configs/          live_constraints.txt, shadow_combo_status.json, tournament_status.json\n"
        "└── audits/           COMMERCIAL_READINESS_AUDIT.md, MANIFEST_V1.json, LLM_AND_AI_FOUNDATION_NOTE.md"
    )
    ax.text(0.8, 0.95, file_tree, fontsize=8.2, color='#94A3B8', fontfamily='Consolas', linespacing=1.4)

    # Right: Recommended Next Step
    ax.text(10.2, 1.65, "RECOMMENDED ROLLOUT ACTION", fontsize=10.5, fontweight='bold', color='#FBBF24', fontfamily=font_main)
    steps = (
        "1. Phase 1 (Live Pilot): Switch DK1 Live to Shadow IDT Champion at 0.5 MWh volume cap.\n"
        "2. Phase 2 (DK2 Scale): Promote Chronos-2 Alone to live execution with 1.0 MWh cap.\n"
        "3. Combined Target: Lock in €1,490/day net P&L with tight automated drawdown stops."
    )
    ax.text(10.2, 0.95, steps, fontsize=8.2, color='#FDE68A', fontfamily=font_main, linespacing=1.4)

    # Save outputs
    local_path = r"c:\energy-trading\commercial ready models\v1\commercial_readiness_executive_summary.png"
    gdrive_path = r"G:\My Drive\commercial ready models\v1\commercial_readiness_executive_summary.png"
    
    plt.savefig(local_path, format='png', bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.savefig(gdrive_path, format='png', bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()
    print("SUCCESS: Infographic generated at:")
    print(" - Local:", local_path)
    print(" - Google Drive:", gdrive_path)

if __name__ == '__main__':
    generate_infographic()
