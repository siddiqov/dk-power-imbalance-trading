# Static stylesheet/script for the unified 96-quarter ledger (plain strings, not f-strings).
LEDGER_CSS = r'''<style>
          *, *:before, *:after {
            box-sizing: border-box;
          }
          html, body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 2px;
            width: 100%;
            max-width: 100%;
            background-color: transparent;
            color: #0F172A;
            overflow-x: hidden;
          }
          .toolbar-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            flex-wrap: wrap;
            gap: 6px;
            width: 100%;
          }
          .btn-action {
            background-color: #0284C7;
            color: #FFFFFF;
            border: none;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.15s;
          }
          .btn-action:hover {
            background-color: #0369A1;
          }
          .search-input {
            padding: 4px 8px;
            border: 1px solid #CBD5E1;
            border-radius: 4px;
            font-size: 11px;
            width: 180px;
            max-width: 100%;
            color: #0F172A;
            background-color: #FFFFFF;
          }
          .table-wrapper {
            border: 1px solid #CBD5E1;
            border-radius: 6px;
            overflow-x: auto;
            overflow-y: auto;
            max-height: 700px;
            width: 100%;
            max-width: 100%;
            background-color: #FFFFFF;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            position: relative;
          }
          table.master-table {
            min-width: 100%;
            width: max-content;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 11px;
            text-align: center;
          }
          table.master-table th {
            background-color: #1E293B;
            color: #F8FAFC;
            padding: 7px 8px;
            font-weight: 600;
            font-size: 10px;
            letter-spacing: 0.01em;
            position: sticky;
            top: 0;
            z-index: 30;
            white-space: nowrap;
            user-select: none;
            cursor: grab;
            border-bottom: 2px solid #0EA5E9;
            text-align: center;
            box-sizing: border-box;
          }
          table.master-table th:hover {
            background-color: #334155;
          }
          table.master-table th.drag-over {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
          }
          table.master-table th.no-drag {
            cursor: default !important;
          }
          /* Column Resize Handle Grip */
          .col-resizer {
            position: absolute;
            top: 0;
            right: 0;
            width: 7px;
            cursor: col-resize !important;
            user-select: none;
            height: 100%;
            z-index: 40;
            background-color: transparent;
            transition: background-color 0.15s;
          }
          .col-resizer:hover, .col-resizer.resizing {
            background-color: #38BDF8 !important;
            width: 7px;
          }
          .col-header-wrap {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            width: 100%;
            height: 100%;
            padding-right: 6px;
            box-sizing: border-box;
          }
          .col-title {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          .btn-col-copy {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border-radius: 3px;
            cursor: pointer;
            background-color: transparent;
            border: none;
            color: #94A3B8;
            font-size: 11px;
            line-height: 1;
            padding: 0;
            margin: 0;
            transition: color 0.15s, background-color 0.15s;
          }
          .btn-col-copy:hover {
            color: #38BDF8 !important;
            background-color: #334155 !important;
          }
          /* Selected Column Visual Highlight */
          table.master-table th.col-selected {
            background-color: #0369A1 !important;
            color: #FFFFFF !important;
            border-bottom: 2px solid #F59E0B !important;
          }
          table.master-table td.col-selected {
            background-color: #E0F2FE !important;
          }
          /* Toast Notification */
          #copyToast {
            display: none;
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #0F172A;
            color: #F8FAFC;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            z-index: 9999;
            border: 1px solid #38BDF8;
            align-items: center;
            gap: 6px;
          }
          table.master-table tr.master-row {
            cursor: pointer;
            border-bottom: 1px solid #E2E8F0;
            transition: background-color 0.1s;
          }
          table.master-table tr.master-row:hover {
            background-color: #DCE7F5 !important;
          }
          table.master-table tr.even-row {
            background-color: #FFFFFF;
          }
          table.master-table tr.odd-row {
            background-color: #F1F5F9;
          }
          table.master-table td {
            padding: 5px 3px;
            white-space: nowrap;
            color: #0F172A;
            font-weight: 500;
            font-size: 10.5px;
            text-align: center;
          }
          .chevron-cell {
            width: 20px !important;
            min-width: 20px !important;
            max-width: 20px !important;
            text-align: center;
            color: #0284C7;
            font-size: 10px;
            cursor: pointer;
            user-select: none;
            padding: 5px 0 !important;
          }
          .chevron-icon {
            display: inline-block;
            transition: transform 0.15s ease;
            font-weight: bold;
          }
          .chevron-icon.expanded {
            color: #0284C7 !important;
          }
          .badge {
            display: inline-block;
            padding: 1.5px 4px;
            border-radius: 3px;
            font-size: 9.5px;
            font-weight: 600;
            white-space: nowrap;
          }
          .badge-buy { background-color: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
          .badge-sell { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
          .badge-hold { background-color: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }
          .badge-breaker { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
          .badge-settled { background-color: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE; }
          .badge-pending { background-color: #FEF9C3; color: #854D0E; border: 1px solid #FEF08A; }
          .pnl-pos { color: #16A34A; font-weight: 700; }
          .pnl-neg { color: #DC2626; font-weight: 700; }
          .pnl-zero { color: #64748B; font-weight: 600; }
          
          /* Drawer Rows - Hidden by default with high specificity */
          table.master-table tr.drawer-row {
            display: none;
            background-color: #F8FAFC !important;
          }
          table.master-table tr.drawer-row.is-open {
            display: table-row !important;
          }
          td.drawer-cell {
            padding: 10px 12px !important;
            border-bottom: 2px solid #CBD5E1;
            background-color: #F8FAFC !important;
            white-space: normal !important;
            max-width: 100%;
            box-sizing: border-box;
          }
          .drawer-banner {
            font-size: 11.5px;
            color: #0284C7;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            border-bottom: 1px dashed #CBD5E1;
            padding-bottom: 4px;
            flex-wrap: wrap;
            gap: 6px;
          }
          .drawer-cards-grid {
            display: grid;
            grid-template-columns: 1fr 1.32fr 1.02fr;
            gap: 12px;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
          }
          @media (max-width: 1100px) {
            .drawer-cards-grid {
              grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            }
          }
          .drawer-card {
            background-color: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 6px;
            padding: 8px 10px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            min-width: 0;
            overflow-x: auto;
            box-sizing: border-box;
          }
          .drawer-card h5 {
            margin: 0 0 5px 0;
            font-size: 11px;
            font-weight: 700;
            color: #1E293B;
            text-transform: uppercase;
            letter-spacing: 0.02em;
          }
          .sub-table {
            width: 100%;
            max-width: 100%;
            border-collapse: collapse;
            font-size: 10.5px;
            margin-top: 2px;
          }
          .sub-table th {
            background-color: #F1F5F9;
            color: #475569;
            padding: 3px 5px;
            font-weight: 600;
            text-align: left;
            border-bottom: 1px solid #E2E8F0;
            white-space: nowrap;
          }
          .sub-table td {
            padding: 3px 5px;
            border-bottom: 1px solid #F1F5F9;
            color: #1E293B;
            white-space: nowrap;
          }
          .sub-table tr.highlight-champ {
            background-color: #ECFDF5;
            font-weight: 600;
          }
          .cf-row {
            display: flex;
            justify-content: space-between;
            padding: 2.5px 0;
            font-size: 11px;
            border-bottom: 1px solid #F1F5F9;
          }
          .cf-row.total {
            border-top: 2px solid #E2E8F0;
            border-bottom: none;
            font-weight: 700;
            padding-top: 4px;
            margin-top: 3px;
          }
          .audit-notice {
            margin-top: 5px;
            font-size: 10px;
            color: #64748B;
            background-color: #F8FAFC;
            padding: 3px 5px;
            border-radius: 4px;
            border: 1px solid #E2E8F0;
          }
        </style>'''
