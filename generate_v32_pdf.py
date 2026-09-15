from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Nurex Trading: V3.2 Algorithm Architecture & Improvement Strategy', border=False, new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, new_x="RIGHT", new_y="TOP", align='C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, title, 0, new_x="LMARGIN", new_y="NEXT", align='L', fill=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('helvetica', '', 10)
        self.multi_cell(0, 6, body)
        self.ln(4)

def generate_pdf():
    pdf = PDF()
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.chapter_title('1. Executive Summary')
    body1 = (
        "The V3.2 Meta-Model represents a fundamental paradigm shift in the Nurex intraday trading architecture. "
        "While V3.1 successfully demonstrated the viability of BiLSTM neural networks for predicting pure price signals, "
        "it suffered from a fatal flaw: 'blind aggression' during tail-event structural crashes. When the Imbalance price "
        "crashed to zero (or negative) due to severe oversupply, V3.1 would historically issue high-conviction BUY orders, "
        "leading to catastrophic PnL drawdowns.\n\n"
        "V3.2 solves this by introducing a 'Meta-Model Ensemble' and 'Physical Grid Awareness'. By chaining the original V3.1 "
        "outputs with deterministic physical data (Wind Error, Flow Data) and dynamic risk limits (Circuit Breakers), "
        "V3.2 actively filters out statistical noise and defends capital during tail-event crashes, flipping catastrophic losses "
        "into significant gains."
    )
    pdf.chapter_body(body1)
    
    # 2. The Core Problems with V3.1
    pdf.chapter_title('2. The Limitations of V3.1')
    body2 = (
        "- Price-Only Myopia: V3.1 predicted direction based solely on historical price matrices. It lacked awareness of the physical grid.\n"
        "- The 'German Flood' Blind Spot: V3.1 was unaware of cross-border scheduled exchanges. It could not detect when massive volumes of cheap power were flooding into DK, crushing the Imbalance price.\n"
        "- Asymmetric Risk Profile: V3.1 treated a 10 EUR spread at 150 EUR the same as a 10 EUR spread at 500 EUR. It failed to recognize that buying during 500+ EUR spot peaks carries exponentially higher risk of down-regulation crashes."
    )
    pdf.chapter_body(body2)

    # 3. V3.2 Architecture & Improvement Strategy
    pdf.chapter_title('3. V3.2 Improvement Strategy & Architecture')
    body3 = (
        "The V3.2 strategy abandons hardcoded heuristics in favor of a true Machine Learning Ensemble, operating on a two-tier framework:\n\n"
        "A. The Flow-Aware Meta-Model (Offense & Noise Filtering)\n"
        "Instead of treating V3.1's output as an ultimate truth, V3.2 feeds the V3.1 score into a new Random Forest Regressor alongside exogenous physical features:\n"
        "  - Scheduled Cross-Border Flow (ENTSO-E)\n"
        "  - Open-Meteo Wind Forecast Error (Proxy for physical renewables)\n"
        "  - DK/DE Spread Volatility (Market interconnectivity)\n"
        "This allows the ML to mathematically 'overrule' V3.1. When physical grid data (e.g., massive German imports combined with high wind generation) mathematically guarantees an oversupply crash, the Random Forest actively reverses V3.1's blind BUY signals into Aggressive Shorts (CRASH PRED (SELL)).\n\n"
        "B. The 95th Percentile Dynamic Circuit Breaker (Defense)\n"
        "V3.2 queries the last 30 days of ENTSO-E spot prices to calculate a dynamic 95th percentile threshold (e.g., 217.14 EUR). "
        "If the current Spot Price exceeds this ceiling, and the ML model does not have the conviction to short-sell, the Engine violently blocks any BUY signals, enforcing a 'C.BREAKER (HOLD)'. "
        "This explicitly decapitates tail-risk, saving massive capital during dangerous price spikes."
    )
    pdf.chapter_body(body3)

    # 4. Working Algorithm Flow
    pdf.chapter_title('4. The Working Algorithm (Execution Flow)')
    body4 = (
        "The live execution loop operates as follows every 15 minutes:\n\n"
        "Step 1: Ingestion\n"
        "  - Fetch V3.1 BiLSTM Score.\n"
        "  - Fetch real-time Wind Data (Open-Meteo) and ENTSO-E Cross-border scheduled flows.\n"
        "  - Calculate 30-day dynamic price ceiling.\n\n"
        "Step 2: Meta-Inference (Random Forest)\n"
        "  - Feed the feature vector [V3.1_Score, Wind_Error, Volatility, Flow, Hour, Quarter] into the V3.2 Random Forest.\n"
        "  - Generate a raw Meta-Score. Map to BUY (>2.0), SELL (<-2.0), or HOLD.\n\n"
        "Step 3: Algorithmic Overrides (The Waterfall)\n"
        "  - Check 1 (Offense): Did the ML natively predict SELL despite V3.1 predicting BUY? If YES -> Force 'CRASH PRED (SELL)'.\n"
        "  - Check 2 (Defense): Is Spot Price > Dynamic Cap AND signal is BUY? If YES -> Force 'C.BREAKER (HOLD)'.\n"
        "  - Check 3 (Default): Pass the native Meta-Model signal through.\n\n"
        "Step 4: Execution\n"
        "  - Volume allocation logic applies (10 MW for standard trades, 25 MW for high-conviction crash predictions)."
    )
    pdf.chapter_body(body4)
    
    # 5. Training Data Sanctity
    pdf.chapter_title('5. Training Pipeline: Absolute Data Sanctity')
    body5 = (
        "In Q1 2025, the Danish TSO (Energinet) fully transitioned the Imbalance market to a 15-minute settlement resolution. "
        "Prior to this, the market was settled hourly (60-minute resolution).\n\n"
        "To enforce the absolute sanctity of the trading engine, V3.2 strictly adheres to a 'No Synthetic Data' rule. "
        "The Flow-Aware Random Forest Meta-Model was trained exclusively on the pure, physical 15-minute intraday data spanning from March 2025 to the present (~1.5 years). "
        "By refusing to artificially up-sample historical hourly data from 2023-2024, the model learns the true, microscopic volatility and exact physical dynamics of the 15-minute market, ensuring unparalleled robustness in live trading."
    )
    pdf.chapter_body(body5)

    os.makedirs('docs', exist_ok=True)
    file_path = 'docs/V3_2_Improvement_Strategy_and_Algorithm.pdf'
    pdf.output(file_path)
    print(f"PDF successfully generated at {file_path}")

if __name__ == "__main__":
    generate_pdf()
