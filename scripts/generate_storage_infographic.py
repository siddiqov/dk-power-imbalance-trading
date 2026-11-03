import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_storage_diagram():
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Background
    ax.add_patch(patches.Rectangle((0, 0), 16, 9, facecolor='#0B1120', zorder=0))

    font_main = 'Segoe UI'
    font_mono = 'Consolas'

    # -------------------------------------------------------------
    # 1. HEADER SECTION
    # -------------------------------------------------------------
    header_box = patches.FancyBboxPatch((0.6, 7.75), 14.8, 0.95,
                                        boxstyle="round,pad=0.06,rounding_size=0.15",
                                        facecolor='#131E32', edgecolor='#253550', linewidth=1.5, zorder=1)
    ax.add_patch(header_box)

    ax.text(0.9, 8.35, "GOOGLE DRIVE MODEL STORAGE ARCHITECTURE",
            fontsize=17, fontweight='bold', color='#FFFFFF', fontfamily=font_main, va='center')
    ax.text(0.9, 8.00, "How & where the commercial-ready trading models and configurations are saved and synced",
            fontsize=11, color='#94A3B8', fontfamily=font_main, va='center')

    # Account pill
    acc_pill = patches.FancyBboxPatch((10.4, 7.90), 4.8, 0.65,
                                      boxstyle="round,pad=0.05,rounding_size=0.12",
                                      facecolor='#0F2E22', edgecolor='#10B981', linewidth=1.2, zorder=2)
    ax.add_patch(acc_pill)
    ax.text(12.8, 8.30, "TARGET GOOGLE ACCOUNT", fontsize=8.5, fontweight='bold', color='#34D399', fontfamily=font_main, ha='center', va='center')
    ax.text(12.8, 8.08, "imabdullahsidiqi@gmail.com", fontsize=10.5, fontweight='bold', color='#FFFFFF', fontfamily=font_mono, ha='center', va='center')

    # -------------------------------------------------------------
    # 2. SYNC PIPELINE (HOW IT'S SAVED)
    # -------------------------------------------------------------
    sync_box = patches.FancyBboxPatch((0.6, 6.35), 14.8, 1.15,
                                      boxstyle="round,pad=0.06,rounding_size=0.15",
                                      facecolor='#0F172A', edgecolor='#1E293B', linewidth=1.2, zorder=1)
    ax.add_patch(sync_box)

    ax.text(0.9, 7.22, "SYNC WORKFLOW: LOCAL DISK  -->  MOUNTED DESKTOP DRIVE  -->  CLOUD GOOGLE DRIVE",
            fontsize=9.5, fontweight='bold', color='#38BDF8', fontfamily=font_main)

    # 3 Steps in Pipeline
    steps = [
        ("1. SOURCE REPOSITORY", "C:\\energy-trading\\commercial ready models\\v1", "#0284C7", "#0C273D"),
        ("2. GOOGLE DRIVE DESKTOP (G:)", "G:\\My Drive\\commercial ready models\\v1", "#059669", "#0D2E24"),
        ("3. GOOGLE CLOUD STORAGE", "drive.google.com (Synced to imabdullahsidiqi@gmail.com)", "#7C3AED", "#24193D")
    ]

    for i, (title, path, border_c, bg_c) in enumerate(steps):
        sx = 0.9 + i * 4.95
        pbox = patches.FancyBboxPatch((sx, 6.48), 4.7, 0.58,
                                      boxstyle="round,pad=0.04,rounding_size=0.08",
                                      facecolor=bg_c, edgecolor=border_c, linewidth=1.2, zorder=2)
        ax.add_patch(pbox)
        ax.text(sx + 0.15, 6.88, title, fontsize=8.5, fontweight='bold', color='#FFFFFF', fontfamily=font_main)
        ax.text(sx + 0.15, 6.64, path, fontsize=8.0, color='#CBD5E1', fontfamily=font_mono)

        # Arrow between steps
        if i < 2:
            ax.annotate("", xy=(sx + 4.90, 6.77), xytext=(sx + 4.72, 6.77),
                        arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=1.8))

    # -------------------------------------------------------------
    # 3. WHERE THEY ARE SAVED (FOLDER HIERARCHY & CONTENTS)
    # -------------------------------------------------------------
    
    # Left Big Card: /models/ folder
    models_card = patches.FancyBboxPatch((0.6, 2.05), 7.6, 4.05,
                                         boxstyle="round,pad=0.06,rounding_size=0.15",
                                         facecolor='#111A2E', edgecolor='#0284C7', linewidth=1.5, zorder=1)
    ax.add_patch(models_card)

    ax.text(0.9, 5.80, "FOLDER: G:\\My Drive\\commercial ready models\\v1\\models\\", 
            fontsize=11.5, fontweight='bold', color='#38BDF8', fontfamily=font_mono)
    ax.text(0.9, 5.52, "Trained binary model weights, feature stores & champion models", 
            fontsize=9.0, color='#94A3B8', fontfamily=font_main)

    model_files = [
        ("DK1_shadow_idt_champion.pkl", "7.6 MB", "LightGBM ID-Target Champion (West Denmark)", "#34D399"),
        ("DK2_chronos_alone (via parquet)", "21.6 MB", "Chronos-2 Probabilistic Foundation Model (East Denmark)", "#34D399"),
        ("DK1_combo_model.pkl", "7.6 MB", "LightGBM + Conformal + Chronos Inputs (Challenger)", "#38BDF8"),
        ("DK2_combo_model.pkl", "2.8 MB", "DK2 Research Checkpoint (Logistic + Flow + Chronos)", "#F87171"),
        ("DK2_batch_logreg.pkl", "2.8 MB", "Logistic Regression Baseline Model (DK2)", "#94A3B8"),
        ("DK1_v4_1_champion.pkl", "4.8 MB", "Production GBDT Baseline Model (DK1)", "#94A3B8"),
        ("DK2_v4_1_champion.pkl", "4.7 MB", "Production GBDT Baseline Model (DK2)", "#94A3B8"),
        ("v4_1_features_DK1.pkl / DK2", "1.2 KB", "Engineered lag & order-flow feature definitions", "#94A3B8"),
    ]

    for j, (fname, fsize, fdesc, tag_color) in enumerate(model_files):
        fy = 5.15 - (j * 0.40)
        # item background
        f_box = patches.FancyBboxPatch((0.85, fy - 0.08), 7.1, 0.34,
                                       boxstyle="round,pad=0.02,rounding_size=0.05",
                                       facecolor='#162238', edgecolor='#1E2D4A', linewidth=0.8, zorder=2)
        ax.add_patch(f_box)

        # Bullet icon
        ax.plot(1.05, fy + 0.08, marker='s', markersize=5, color=tag_color, zorder=3)
        ax.text(1.25, fy + 0.08, fname, fontsize=8.2, fontweight='bold', color='#FFFFFF', fontfamily=font_mono, va='center')
        ax.text(4.45, fy + 0.08, fsize, fontsize=7.5, color='#38BDF8', fontfamily=font_mono, va='center')
        ax.text(5.15, fy + 0.08, fdesc, fontsize=7.2, color='#94A3B8', fontfamily=font_main, va='center')


    # Right Card 1: /configs/ folder
    configs_card = patches.FancyBboxPatch((8.45, 3.85), 6.95, 2.25,
                                          boxstyle="round,pad=0.06,rounding_size=0.15",
                                          facecolor='#111A2E', edgecolor='#10B981', linewidth=1.5, zorder=1)
    ax.add_patch(configs_card)

    ax.text(8.75, 5.80, "FOLDER: ...\\v1\\configs\\", 
            fontsize=11.5, fontweight='bold', color='#34D399', fontfamily=font_mono)
    ax.text(8.75, 5.52, "Execution limits, parameter thresholds & live shadow status feeds", 
            fontsize=9.0, color='#94A3B8', fontfamily=font_main)

    config_files = [
        ("live_constraints.txt", "Position sizing, risk boundaries & execution rules"),
        ("shadow_combo_status.json", "Multi-strategy shadow evaluation sync telemetry"),
        ("tournament_status.json", "Head-to-head model comparison metrics & cycle logs")
    ]

    for k, (cfname, cfdesc) in enumerate(config_files):
        cy = 5.08 - (k * 0.45)
        c_box = patches.FancyBboxPatch((8.70, cy - 0.08), 6.45, 0.38,
                                       boxstyle="round,pad=0.02,rounding_size=0.05",
                                       facecolor='#14282B', edgecolor='#18443B', linewidth=0.8, zorder=2)
        ax.add_patch(c_box)
        ax.plot(8.90, cy + 0.10, marker='o', markersize=5, color='#34D399', zorder=3)
        ax.text(9.10, cy + 0.10, cfname, fontsize=8.4, fontweight='bold', color='#FFFFFF', fontfamily=font_mono, va='center')
        ax.text(9.10, cy - 0.02, cfdesc, fontsize=7.5, color='#6EE7B7', fontfamily=font_main, va='center')


    # Right Card 2: Documentation & Audits (Root folder)
    docs_card = patches.FancyBboxPatch((8.45, 2.05), 6.95, 1.65,
                                       boxstyle="round,pad=0.06,rounding_size=0.15",
                                       facecolor='#111A2E', edgecolor='#A78BFA', linewidth=1.5, zorder=1)
    ax.add_patch(docs_card)

    ax.text(8.75, 3.42, "FOLDER: ...\\v1\\ (Root Documentation)", 
            fontsize=11.5, fontweight='bold', color='#C084FC', fontfamily=font_mono)

    doc_files = [
        ("COMMERCIAL_READINESS_AUDIT.md", "Full 12-day audit, drawdown metrics & go-live plan"),
        ("MANIFEST_V1.json", "Structured JSON catalog of model versions & parameters"),
        ("LLM_AND_AI_FOUNDATION_NOTE.md", "Guide on using foundation models vs local trading models")
    ]

    for m, (dfname, dfdesc) in enumerate(doc_files):
        dy = 3.05 - (m * 0.38)
        d_box = patches.FancyBboxPatch((8.70, dy - 0.08), 6.45, 0.32,
                                       boxstyle="round,pad=0.02,rounding_size=0.05",
                                       facecolor='#211838', edgecolor='#392760', linewidth=0.8, zorder=2)
        ax.add_patch(d_box)
        ax.plot(8.90, dy + 0.08, marker='>', markersize=5, color='#C084FC', zorder=3)
        ax.text(9.10, dy + 0.08, dfname, fontsize=8.0, fontweight='bold', color='#FFFFFF', fontfamily=font_mono, va='center')
        ax.text(12.3, dy + 0.08, dfdesc, fontsize=7.2, color='#DDD6FE', fontfamily=font_main, va='center')


    # -------------------------------------------------------------
    # 4. HOW TO ACCESS (FOOTER BAR)
    # -------------------------------------------------------------
    footer_box = patches.FancyBboxPatch((0.6, 0.5), 14.8, 1.35,
                                        boxstyle="round,pad=0.06,rounding_size=0.15",
                                        facecolor='#131E32', edgecolor='#253550', linewidth=1.5, zorder=1)
    ax.add_patch(footer_box)

    ax.text(0.9, 1.55, "HOW YOUR BROTHER / TEAM CAN ACCESS THESE FILES RIGHT NOW:", 
            fontsize=10.5, fontweight='bold', color='#FBBF24', fontfamily=font_main)

    access_methods = [
        ("METHOD 1: VIA WINDOWS EXPLORER", "Open Windows Explorer and navigate directly to: G:\\My Drive\\commercial ready models\\v1"),
        ("METHOD 2: VIA GOOGLE DRIVE WEB", "Log into drive.google.com with imabdullahsidiqi@gmail.com -> folder 'commercial ready models'"),
        ("METHOD 3: AUTOMATIC REAL-TIME SYNC", "GoogleDriveFS is running continuously; any new update in C:\\energy-trading syncs instantly to G:\\")
    ]

    for n, (mtitle, mdesc) in enumerate(access_methods):
        mx = 0.9 + n * 4.95
        ax.text(mx, 1.25, mtitle, fontsize=8.5, fontweight='bold', color='#FFFFFF', fontfamily=font_main)
        ax.text(mx, 1.00, mdesc, fontsize=7.8, color='#94A3B8', fontfamily=font_main)

    # Output paths
    local_out = r"c:\energy-trading\commercial ready models\v1\drive_storage_structure.png"
    gdrive_out = r"G:\My Drive\commercial ready models\v1\drive_storage_structure.png"
    art_out = r"C:\Users\ziabz\.gemini\antigravity\brain\5a0803d8-3c84-4028-b792-e75ab00d6294\drive_storage_structure.png"

    plt.savefig(local_out, format='png', bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.savefig(gdrive_out, format='png', bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.savefig(art_out, format='png', bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()
    print("SUCCESS: Storage diagram generated.")

if __name__ == '__main__':
    generate_storage_diagram()
