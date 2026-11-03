import base64
import os

png_path = r"c:\energy-trading\commercial ready models\v1\drive_storage_structure.png"
with open(png_path, "rb") as f:
    b64_img = base64.b64encode(f.read()).decode("utf-8")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Google Drive Storage Architecture - Commercial Ready Models V1</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2">
  <div class="max-w-5xl mx-auto bg-[#0b1120] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl text-slate-100">
    
    <!-- Top Header Bar -->
    <div class="px-6 py-4 bg-[#111a2e] border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">Google Drive Synced</span>
          <span class="px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-sky-500/10 text-sky-400 border border-sky-500/30">v1 Directory Structure</span>
        </div>
        <h1 class="text-xl font-bold tracking-tight text-white mt-1.5">Where &amp; How Trading Models Are Saved on Google Drive</h1>
        <p class="text-xs text-slate-400">Target Account: <code class="text-emerald-400 bg-emerald-950/40 px-1.5 py-0.5 rounded border border-emerald-500/20">imabdullahsidiqi@gmail.com</code> &bull; Drive Letter: <code class="text-sky-300 bg-sky-950/40 px-1.5 py-0.5 rounded border border-sky-500/20">G:\My Drive</code></p>
      </div>

      <div class="bg-slate-900 border border-slate-700/60 px-4 py-2 rounded-xl text-right">
        <div class="text-[11px] font-semibold text-slate-400 uppercase tracking-wide">Sync Status</div>
        <div class="text-sm font-bold text-emerald-400 flex items-center gap-1.5 justify-end">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> GoogleDriveFS Active
        </div>
        <div class="text-[10px] text-slate-400 font-mono">11 Items Synced</div>
      </div>
    </div>

    <!-- The Diagram (Native High-Res Embed) -->
    <div class="p-3 bg-[#070b14] border-b border-slate-800 flex justify-center">
      <div class="w-full rounded-xl overflow-hidden border border-slate-800/80 shadow-lg">
        <img src="data:image/png;base64,{b64_img}" 
             alt="Google Drive Storage Architecture &amp; Directory Diagram" 
             class="w-full h-auto block rounded-lg select-none" />
      </div>
    </div>

    <!-- Quick Navigation Breakdown for Lead -->
    <div class="p-5 space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
        
        <div class="bg-[#111a2e] border border-sky-500/30 p-3.5 rounded-xl">
          <div class="font-bold text-sky-400 mb-1 flex items-center gap-1.5">
            <span>📁</span> 1. Binary Models Folder
          </div>
          <div class="font-mono text-[11px] text-slate-300 mb-1.5">...\\v1\\models\\</div>
          <p class="text-slate-400 text-[11px] leading-relaxed">
            Contains 8 trained model bundles (<code class="text-slate-200">.pkl</code>) including DK1 Champion, DK2 Chronos-2, and V4.1 baseline feature dictionaries.
          </p>
        </div>

        <div class="bg-[#111a2e] border border-emerald-500/30 p-3.5 rounded-xl">
          <div class="font-bold text-emerald-400 mb-1 flex items-center gap-1.5">
            <span>⚙️</span> 2. Strategy Configs Folder
          </div>
          <div class="font-mono text-[11px] text-slate-300 mb-1.5">...\\v1\\configs\\</div>
          <p class="text-slate-400 text-[11px] leading-relaxed">
            Contains <code class="text-slate-200">live_constraints.txt</code> (position caps &amp; risk gates) and live tournament tracking feeds.
          </p>
        </div>

        <div class="bg-[#111a2e] border border-purple-500/30 p-3.5 rounded-xl">
          <div class="font-bold text-purple-400 mb-1 flex items-center gap-1.5">
            <span>📄</span> 3. Audits &amp; Manifest
          </div>
          <div class="font-mono text-[11px] text-slate-300 mb-1.5">...\\v1\\ (Root)</div>
          <p class="text-slate-400 text-[11px] leading-relaxed">
            Includes <code class="text-slate-200">COMMERCIAL_READINESS_AUDIT.md</code> and <code class="text-slate-200">MANIFEST_V1.json</code> registry for version traceability.
          </p>
        </div>

      </div>

      <!-- Quick Open Commands -->
      <div class="bg-[#0f172a] border border-slate-800 rounded-xl p-3 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-xs">
        <div class="text-slate-300">
          <span class="font-bold text-amber-400">Direct Explorer Path:</span>
          <code class="ml-1.5 bg-[#0b1120] text-sky-300 px-2 py-0.5 rounded border border-slate-800 font-mono">G:\My Drive\commercial ready models\v1</code>
        </div>
        <div class="text-slate-400 text-[11px]">
          Target Google Account: <span class="text-slate-200 font-semibold">imabdullahsidiqi@gmail.com</span>
        </div>
      </div>

    </div>

  </div>
</body>
</html>
"""

out_path = r"C:\Users\ziabz\.gemini\antigravity\brain\5a0803d8-3c84-4028-b792-e75ab00d6294\explanatory_card.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print("SUCCESS: Updated explanatory card with Drive storage diagram.")
