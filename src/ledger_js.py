# Static stylesheet/script for the unified 96-quarter ledger (plain strings, not f-strings).
LEDGER_JS = r'''<script>
            // Column Drag & Drop Reordering
            (function() {
              var table = document.getElementById('masterTable');
              var headers = table.querySelectorAll('th[draggable="true"]');
              var dragSrcIndex = null;

              headers.forEach(function(th) {
                th.addEventListener('dragstart', function(e) {
                  if (e.target.classList.contains('col-resizer') || e.target.classList.contains('btn-col-copy')) {
                    e.preventDefault();
                    return;
                  }
                  dragSrcIndex = this.cellIndex;
                  e.dataTransfer.effectAllowed = 'move';
                  e.dataTransfer.setData('text/plain', dragSrcIndex);
                  this.style.opacity = '0.5';
                });

                th.addEventListener('dragover', function(e) {
                  e.preventDefault();
                  e.dataTransfer.dropEffect = 'move';
                  this.classList.add('drag-over');
                });

                th.addEventListener('dragleave', function() {
                  this.classList.remove('drag-over');
                });

                th.addEventListener('dragend', function() {
                  this.style.opacity = '1';
                  headers.forEach(function(h) { h.classList.remove('drag-over'); });
                });

                th.addEventListener('drop', function(e) {
                  e.preventDefault();
                  this.classList.remove('drag-over');
                  var targetIndex = this.cellIndex;
                  if (dragSrcIndex === null || dragSrcIndex === targetIndex || targetIndex === 0) return;

                  // Move header
                  var headerRow = table.querySelector('thead tr');
                  var srcTh = headerRow.children[dragSrcIndex];
                  var targetTh = headerRow.children[targetIndex];
                  if (dragSrcIndex < targetIndex) {
                    headerRow.insertBefore(srcTh, targetTh.nextSibling);
                  } else {
                    headerRow.insertBefore(srcTh, targetTh);
                  }

                  // Move cells in all master-rows
                  var masterRows = table.querySelectorAll('tr.master-row');
                  masterRows.forEach(function(row) {
                    var srcTd = row.children[dragSrcIndex];
                    var targetTd = row.children[targetIndex];
                    if (srcTd && targetTd) {
                      if (dragSrcIndex < targetIndex) {
                        row.insertBefore(srcTd, targetTd.nextSibling);
                      } else {
                        row.insertBefore(srcTd, targetTd);
                      }
                    }
                  });

                  dragSrcIndex = null;
                });
              });
            })();

            // Interactive Mouse Column Resizing Logic
            (function() {
              var table = document.getElementById('masterTable');
              if (!table) return;
              var allTh = table.querySelectorAll('thead th');

              allTh.forEach(function(th) {
                if (th.classList.contains('no-drag')) return;

                th.style.position = 'sticky';

                var resizer = document.createElement('div');
                resizer.className = 'col-resizer';
                th.appendChild(resizer);

                resizer.addEventListener('mousedown', function(e) {
                  e.stopPropagation();
                  e.preventDefault();

                  var colIndex = th.cellIndex;
                  var startX = e.clientX;
                  var startWidth = th.getBoundingClientRect().width;
                  resizer.classList.add('resizing');
                  th.setAttribute('draggable', 'false');

                  var masterRows = table.querySelectorAll('tr.master-row');
                  var targetTds = [];
                  masterRows.forEach(function(row) {
                    if (row.children[colIndex]) {
                      targetTds.push(row.children[colIndex]);
                    }
                  });

                  function onMouseMove(moveEvent) {
                    var delta = moveEvent.clientX - startX;
                    var newWidth = Math.max(45, Math.round(startWidth + delta));
                    var widthPx = newWidth + 'px';
                    th.style.width = widthPx;
                    th.style.minWidth = widthPx;
                    th.style.maxWidth = widthPx;

                    targetTds.forEach(function(td) {
                      td.style.width = widthPx;
                      td.style.minWidth = widthPx;
                      td.style.maxWidth = widthPx;
                    });
                  }

                  function onMouseUp() {
                    resizer.classList.remove('resizing');
                    th.setAttribute('draggable', 'true');
                    window.removeEventListener('mousemove', onMouseMove);
                    window.removeEventListener('mouseup', onMouseUp);
                  }

                  window.addEventListener('mousemove', onMouseMove);
                  window.addEventListener('mouseup', onMouseUp);
                });
              });
            })();

            function toggleRow(idx) {
              var drawer = document.getElementById('drawer-' + idx);
              var icon = document.getElementById('icon-' + idx);
              if (!drawer) return;
              var isHidden = (drawer.style.display === 'none' || !drawer.classList.contains('is-open'));
              if (isHidden) {
                drawer.style.display = 'table-row';
                drawer.classList.add('is-open');
                if (icon) {
                  icon.innerHTML = '&#9660;';
                  icon.classList.add('expanded');
                }
              } else {
                drawer.style.display = 'none';
                drawer.classList.remove('is-open');
                if (icon) {
                  icon.innerHTML = '&#9654;';
                  icon.classList.remove('expanded');
                }
              }
            }

            function expandAll() {
              document.querySelectorAll('tr.drawer-row').forEach(function(d) { d.style.display = 'table-row'; d.classList.add('is-open'); });
              document.querySelectorAll('.chevron-icon').forEach(function(i) {
                i.innerHTML = '&#9660;';
                i.classList.add('expanded');
              });
            }

            function collapseAll() {
              document.querySelectorAll('tr.drawer-row').forEach(function(d) { d.style.display = 'none'; d.classList.remove('is-open'); });
              document.querySelectorAll('.chevron-icon').forEach(function(i) {
                i.innerHTML = '&#9654;';
                i.classList.remove('expanded');
              });
            }

            function filterQuarters() {
              var input = document.getElementById('filterInput');
              var filter = input.value.toUpperCase();
              var rows = document.querySelectorAll('tr.master-row');
              rows.forEach(function(r) {
                var text = r.innerText || r.textContent;
                var idx = r.id.replace('row-', '');
                var drawer = document.getElementById('drawer-' + idx);
                if (text.toUpperCase().indexOf(filter) > -1) {
                  r.style.display = '';
                } else {
                  r.style.display = 'none';
                  if (drawer) { drawer.style.display = 'none'; drawer.classList.remove('is-open'); }
                }
              });
            }

            function showToast(msg) {
              var toast = document.getElementById('copyToast');
              if (!toast) return;
              toast.innerText = msg;
              toast.style.display = 'flex';
              setTimeout(function() {
                toast.style.display = 'none';
              }, 2200);
            }

            function copyToClipboard(text, successMsg) {
              if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(text).then(function() {
                  showToast(successMsg);
                }).catch(function() {
                  fallbackCopy(text, successMsg);
                });
              } else {
                fallbackCopy(text, successMsg);
              }
            }

            function fallbackCopy(text, successMsg) {
              var textArea = document.createElement('textarea');
              textArea.value = text;
              textArea.style.position = 'fixed';
              textArea.style.left = '-9999px';
              document.body.appendChild(textArea);
              textArea.focus();
              textArea.select();
              try {
                document.execCommand('copy');
                showToast(successMsg);
              } catch (err) {
                showToast('❌ Copy failed');
              }
              document.body.removeChild(textArea);
            }

            function getColumnData(colIndex) {
              var table = document.getElementById('masterTable');
              var headerTh = table.querySelector('thead tr').children[colIndex];
              var titleSpan = headerTh.querySelector('.col-title') || headerTh;
              var title = titleSpan.innerText.trim();
              var rows = table.querySelectorAll('tr.master-row');
              var values = [title];
              rows.forEach(function(r) {
                if (r.children[colIndex]) {
                  values.push(r.children[colIndex].innerText.trim());
                }
              });
              return values;
            }

            window.copySingleColumn = function(btn, e) {
              if (e) {
                e.stopPropagation();
                e.preventDefault();
              }
              var th = btn.closest('th');
              var colIndex = th.cellIndex;
              var colData = getColumnData(colIndex);
              var text = colData.join('\n');
              var title = colData[0];
              copyToClipboard(text, '✅ Copied column \"' + title + '\" (' + (colData.length - 1) + ' rows)');
            };

            // Header Click Selection for Multi-Column Copy
            (function() {
              var table = document.getElementById('masterTable');
              if (!table) return;
              var headers = table.querySelectorAll('thead th');

              headers.forEach(function(th) {
                if (th.classList.contains('no-drag')) return;
                th.addEventListener('click', function(e) {
                  if (e.target.classList.contains('btn-col-copy') || e.target.classList.contains('col-resizer')) return;

                  var colIdx = this.cellIndex;
                  this.classList.toggle('col-selected');
                  var isSel = this.classList.contains('col-selected');

                  var masterRows = table.querySelectorAll('tr.master-row');
                  masterRows.forEach(function(row) {
                    if (row.children[colIdx]) {
                      if (isSel) {
                        row.children[colIdx].classList.add('col-selected');
                      } else {
                        row.children[colIdx].classList.remove('col-selected');
                      }
                    }
                  });
                });
              });
            })();

            window.copySelectedColumns = function() {
              var table = document.getElementById('masterTable');
              var selThs = table.querySelectorAll('thead th.col-selected');
              if (selThs.length === 0) {
                showToast('ℹ️ Click any column header to select it first, then click Copy Selected.');
                return;
              }

              var colIndices = [];
              var colTitles = [];
              selThs.forEach(function(th) {
                colIndices.push(th.cellIndex);
                var titleSpan = th.querySelector('.col-title') || th;
                colTitles.push(titleSpan.innerText.trim());
              });

              var masterRows = table.querySelectorAll('tr.master-row');
              var lines = [];
              // Header line (tab-separated for direct Excel/Sheets paste)
              lines.push(colTitles.join('\t'));

              // Row lines
              masterRows.forEach(function(row) {
                var rowVals = [];
                colIndices.forEach(function(cIdx) {
                  var td = row.children[cIdx];
                  rowVals.push(td ? td.innerText.trim() : '');
                });
                lines.push(rowVals.join('\t'));
              });

              var text = lines.join('\n');
              copyToClipboard(text, '✅ Copied ' + colIndices.length + ' selected columns (' + (lines.length - 1) + ' rows, TSV format)');
            };
          </script>'''
