/**
 * LLM Eval Pipeline - Interactive Dashboard Controller
 * Implements high-end visualizations, completions browsing, and statistics
 * calculated dynamically from the dbt/DuckDB exported eval.duckdb dataset.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Vector Lucide Icons
    lucide.createIcons();
    
    // 2. Validate Data Load
    if (!window.EVAL_DATA) {
        console.error("Evaluation data not found. Please ensure docs/data.js is compiled.");
        renderEmptyStates();
        return;
    }
    
    const data = window.EVAL_DATA;
    
    // Global Dashboard State
    const state = {
        activeTab: "overview",
        completions: {
            filtered: [...data.completions],
            currentPage: 1,
            pageSize: 12,
            selectedRow: null
        },
        filters: {
            search: "",
            model: "all",
            subject: "all",
            outcome: "all"
        },
        charts: {
            comparison: null,
            drift: null
        }
    };
    
    // 3. Kickoff Dashboard Initialization
    initTabs();
    initOverviewStats();
    initLeaderboard();
    initOverviewCharts();
    initBreakdownSelectors();
    initCompletionsExplorer();
    initCompletionsDrawer();
    
    // Handle window resize dynamically for Chart.js
    window.addEventListener("resize", () => {
        if (state.charts.comparison) state.charts.comparison.resize();
        if (state.charts.drift) state.charts.drift.resize();
    });

    // ==================== INITIALIZERS & LOGIC ====================

    /**
     * Set up page tab switching with smooth transitions
     */
    function initTabs() {
        const tabButtons = document.querySelectorAll(".tab-btn");
        const tabContents = document.querySelectorAll(".tab-content");
        
        tabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-tab");
                
                // Update active buttons
                tabButtons.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                
                // Update active contents
                tabContents.forEach(c => {
                    c.classList.remove("active");
                    if (c.getAttribute("id") === targetTab) {
                        c.classList.add("active");
                    }
                });
                
                state.activeTab = targetTab;
                
                // Trigger chart updates to ensure canvas dimensions are correct
                if (targetTab === "overview" && state.charts.comparison) {
                    state.charts.comparison.update();
                } else if (targetTab === "performance" && state.charts.drift) {
                    state.charts.drift.update();
                }
            });
        });
    }

    /**
     * Compute and display core KPI statistics including Wilson 95% Confidence Intervals
     */
    function initOverviewStats() {
        const completions = data.completions;
        if (completions.length === 0) return;
        
        const totalEvals = completions.length;
        const correctEvals = completions.filter(c => c.is_correct).length;
        const passRate = correctEvals / totalEvals;
        
        // Wilson Score 95% Confidence Interval Calculation
        // Formula: (p + z^2/(2n) +/- z * sqrt(p*(1-p)/n + z^2/(4n^2))) / (1 + z^2/n)
        const z = 1.96; // 95% confidence level
        const n = totalEvals;
        const p = passRate;
        const denominator = 1 + (z * z) / n;
        const center = (p + (z * z) / (2 * n)) / denominator;
        const spread = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n)) / denominator;
        
        const ciLower = Math.max(0, center - spread);
        const ciUpper = Math.min(1, center + spread);
        
        // Compute Latency
        const latencies = completions.map(c => c.latency_ms).filter(Boolean);
        const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
        
        // Compute Unparseable
        const unparseableCount = completions.filter(c => !c.parsed_answer || c.parsed_answer === "unparseable").length;
        const unparseableRate = unparseableCount / totalEvals;
        
        // Render values in DOM
        document.getElementById("stat-pass-rate").innerText = `${(passRate * 100).toFixed(1)}%`;
        document.getElementById("stat-ci-lower").innerText = `${(ciLower * 100).toFixed(1)}%`;
        document.getElementById("stat-ci-upper").innerText = `${(ciUpper * 100).toFixed(1)}%`;
        
        document.getElementById("stat-total-evals").innerText = totalEvals.toLocaleString();
        document.getElementById("stat-run-badge").innerText = `${data.runs.length} runs total`;
        document.getElementById("stat-latency").innerText = `${Math.round(avgLatency)}ms`;
        
        document.getElementById("stat-unparseable").innerText = `${(unparseableRate * 100).toFixed(1)}%`;
        
        const unparseableBadge = document.getElementById("stat-unparseable-badge");
        if (unparseableRate > 0.08) {
            unparseableBadge.className = "metric-badge red";
            unparseableBadge.innerText = "Parser missed >8%";
        } else {
            unparseableBadge.className = "metric-badge green";
            unparseableBadge.innerText = "Parser clean";
        }
        
        // Position Wilson CI slider elements
        const markerPos = passRate * 100;
        const rangeLeft = ciLower * 100;
        const rangeWidth = (ciUpper - ciLower) * 100;
        
        document.getElementById("stat-ci-marker").style.left = `${markerPos}%`;
        const rangeBar = document.getElementById("stat-ci-range-bar");
        rangeBar.style.left = `${rangeLeft}%`;
        rangeBar.style.width = `${rangeWidth}%`;
    }

    /**
     * Render the ranked leaderboard of models
     */
    function initLeaderboard() {
        const listContainer = document.getElementById("model-leaderboard-list");
        if (!listContainer) return;
        
        // Aggregate statistics per model
        const modelStats = {};
        data.completions.forEach(c => {
            if (!modelStats[c.model]) {
                modelStats[c.model] = {
                    name: c.model,
                    total: 0,
                    correct: 0,
                    latency: 0,
                    tokens: 0
                };
            }
            const stats = modelStats[c.model];
            stats.total++;
            if (c.is_correct) stats.correct++;
            stats.latency += c.latency_ms || 0;
            stats.tokens += (c.input_tokens || 0) + (c.output_tokens || 0);
        });
        
        // Format and Sort
        const models = Object.values(modelStats).map(m => {
            const avgLat = m.latency / m.total;
            const avgTok = m.tokens / m.total;
            return {
                name: formatModelName(m.name),
                rawName: m.name,
                passRate: m.correct / m.total,
                avgLatency: avgLat,
                costIndex: avgTok / 1000 // Proxy for tokens
            };
        }).sort((a, b) => b.passRate - a.passRate);
        
        // Generate HTML list
        listContainer.innerHTML = models.map((model, idx) => {
            const rankEmoji = idx === 0 ? "🏆" : (idx === 1 ? "🥈" : "🥉");
            return `
                <div class="leader-item">
                    <div class="model-profile">
                        <div class="model-rank">${rankEmoji}</div>
                        <div class="model-meta-info">
                            <h4>${model.name}</h4>
                            <p>Avg Latency: ${Math.round(model.avgLatency)}ms | Cost Index: ${model.costIndex.toFixed(1)}k/t</p>
                        </div>
                    </div>
                    <div class="model-perf-score">
                        <div class="score-main">${(model.passRate * 100).toFixed(1)}%</div>
                        <div class="score-sub">pass rate</div>
                    </div>
                </div>
            `;
        }).join("");
    }

    /**
     * Initialize high-end Chart.js visualizations
     */
    function initOverviewCharts() {
        // Chart 1: Pass Rate by Model & Subject Group (Grouped Bar Chart)
        const compCtx = document.getElementById("overview-comparison-chart");
        if (compCtx) {
            // Extract unique subject groups and models
            const subjectGroups = [...new Set(data.pass_rates.map(p => p.subject_group))].sort();
            const models = [...new Set(data.pass_rates.map(p => p.model))];
            
            // Format labels for display
            const groupLabels = subjectGroups.map(g => formatSubjectGroup(g));
            
            // Configure datasets colors
            const palette = {
                "haiku": {
                    fill: "rgba(59, 130, 246, 0.4)",
                    border: "rgb(59, 130, 246)"
                },
                "sonnet": {
                    fill: "rgba(168, 85, 247, 0.4)",
                    border: "rgb(168, 85, 247)"
                },
                "opus": {
                    fill: "rgba(236, 72, 153, 0.4)",
                    border: "rgb(236, 72, 153)"
                }
            };
            
            const datasets = models.map(model => {
                const lower = model.toLowerCase();
                const key = lower.includes("haiku") ? "haiku"
                    : lower.includes("sonnet") ? "sonnet"
                    : lower.includes("opus") ? "opus"
                    : "sonnet";
                const colors = palette[key];
                
                // Map pass rate to each subject group for this model
                const rates = subjectGroups.map(group => {
                    const row = data.pass_rates.find(p => p.model === model && p.subject_group === group);
                    return row ? row.pass_rate * 100 : 0;
                });
                
                return {
                    label: formatModelName(model),
                    data: rates,
                    backgroundColor: colors.fill,
                    borderColor: colors.border,
                    borderWidth: 1.5,
                    borderRadius: 4,
                    barPercentage: 0.75,
                    categoryPercentage: 0.8
                };
            });
            
            state.charts.comparison = new Chart(compCtx, {
                type: 'bar',
                data: {
                    labels: groupLabels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            min: 0,
                            max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: {
                                color: '#94a3b8',
                                callback: value => `${value}%`
                            }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: '#f8fafc',
                                font: { family: 'Outfit', weight: '600' }
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: context => ` ${context.dataset.label}: ${context.raw.toFixed(1)}%`
                            }
                        }
                    }
                }
            });
        }
        
        // Chart 2: Run Drift Timeline (Line Chart)
        const driftCtx = document.getElementById("run-drift-chart");
        if (driftCtx) {
            const models = [...new Set(data.drift.map(d => d.model))];
            
            // Generate sequence IDs for X axis (Run 1, Run 2, Run 3)
            const runs = [...new Set(data.drift.map(d => d.run_seq))].sort((a,b) => a-b);
            const runLabels = runs.map(r => `Run ${r}`);
            
            const linePalette = {
                "haiku": "rgb(59, 130, 246)",
                "sonnet": "rgb(168, 85, 247)",
                "opus": "rgb(236, 72, 153)"
            };
            
            const lineDatasets = models.map(model => {
                const lower = model.toLowerCase();
                const key = lower.includes("haiku") ? "haiku"
                    : lower.includes("sonnet") ? "sonnet"
                    : lower.includes("opus") ? "opus"
                    : "sonnet";
                const color = linePalette[key];
                
                const rates = runs.map(run => {
                    const row = data.drift.find(d => d.model === model && d.run_seq === run);
                    return row ? row.pass_rate * 100 : null;
                });
                
                return {
                    label: formatModelName(model),
                    data: rates,
                    borderColor: color,
                    backgroundColor: color.replace("rgb", "rgba").replace(")", ", 0.15)"),
                    borderWidth: 3,
                    tension: 0.25,
                    fill: true,
                    pointBackgroundColor: color,
                    pointBorderColor: '#070913',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                };
            });
            
            state.charts.drift = new Chart(driftCtx, {
                type: 'line',
                data: {
                    labels: runLabels,
                    datasets: lineDatasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            min: 0,
                            max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: {
                                color: '#94a3b8',
                                callback: value => `${value}%`
                            }
                        },
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.03)' },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#f8fafc',
                                font: { family: 'Outfit', weight: '600' }
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: context => ` ${context.dataset.label}: ${context.raw.toFixed(1)}%`
                            }
                        }
                    }
                }
            });
        }
    }

    /**
     * Set up category selectors and trigger dynamic breakdowns on model select
     */
    function initBreakdownSelectors() {
        const select = document.getElementById("breakdown-model-select");
        if (!select) return;
        
        const models = [...new Set(data.categories.map(c => c.model))];
        select.innerHTML = models.map(m => `<option value="${m}">${formatModelName(m)}</option>`).join("");
        
        // Render list for the first model automatically
        if (models.length > 0) {
            renderSubjectBreakdown(models[0]);
        }
        
        select.addEventListener("change", (e) => {
            renderSubjectBreakdown(e.target.value);
        });
    }

    /**
     * Render the top 5 and bottom 5 subjects for a selected model
     */
    function renderSubjectBreakdown(model) {
        const categories = data.categories.filter(c => c.model === model);
        
        // Top 5 subjects
        const topCats = [...categories].sort((a,b) => b.pass_rate - a.pass_rate).slice(0, 5);
        // Bottom 5 subjects 
        const bottomCats = [...categories].sort((a,b) => a.pass_rate - b.pass_rate).slice(0, 5);
        
        const renderList = (cats, containerId, isSuccess) => {
            const container = document.getElementById(containerId);
            if (!container) return;
            
            container.innerHTML = cats.map(cat => {
                const percent = cat.pass_rate * 100;
                const fillClass = isSuccess ? "success" : "danger";
                return `
                    <div class="cat-bar-item">
                        <div class="cat-bar-header">
                            <span class="cat-bar-name">${formatSubject(cat.subject)}</span>
                            <span class="cat-bar-val">${percent.toFixed(0)}%</span>
                        </div>
                        <div class="cat-bar-track">
                            <div class="cat-bar-fill ${fillClass}" style="width: ${percent}%;"></div>
                        </div>
                    </div>
                `;
            }).join("");
        };
        
        renderList(topCats, "top-categories-list", true);
        renderList(bottomCats, "bottom-categories-list", false);
    }

    /**
     * Populate filter controls and paginate table rows in the Completions tab
     */
    function initCompletionsExplorer() {
        const completions = data.completions;
        
        // 1. Populate Dropdowns dynamically
        const filterModel = document.getElementById("filter-model");
        const filterSubject = document.getElementById("filter-subject");
        
        const models = [...new Set(completions.map(c => c.model))];
        const subjects = [...new Set(completions.map(c => c.subject_group))].sort();
        
        models.forEach(m => {
            filterModel.innerHTML += `<option value="${m}">${formatModelName(m)}</option>`;
        });
        
        subjects.forEach(s => {
            filterSubject.innerHTML += `<option value="${s}">${formatSubjectGroup(s)}</option>`;
        });
        
        // 2. Attach Event Listeners
        document.getElementById("search-input").addEventListener("input", (e) => {
            state.filters.search = e.target.value.toLowerCase();
            state.completions.currentPage = 1;
            applyFilters();
        });
        
        filterModel.addEventListener("change", (e) => {
            state.filters.model = e.target.value;
            state.completions.currentPage = 1;
            applyFilters();
        });
        
        filterSubject.addEventListener("change", (e) => {
            state.filters.subject = e.target.value;
            state.completions.currentPage = 1;
            applyFilters();
        });
        
        document.getElementById("filter-outcome").addEventListener("change", (e) => {
            state.filters.outcome = e.target.value;
            state.completions.currentPage = 1;
            applyFilters();
        });
        
        // Pagination Buttons
        document.getElementById("prev-page-btn").addEventListener("click", () => {
            if (state.completions.currentPage > 1) {
                state.completions.currentPage--;
                renderCompletionsTable();
            }
        });
        
        document.getElementById("next-page-btn").addEventListener("click", () => {
            const maxPage = Math.ceil(state.completions.filtered.length / state.completions.pageSize);
            if (state.completions.currentPage < maxPage) {
                state.completions.currentPage++;
                renderCompletionsTable();
            }
        });
        
        // Run initial render
        applyFilters();
    }

    /**
     * Compute search indices and outcomes filters, then re-render
     */
    function applyFilters() {
        const completions = data.completions;
        
        state.completions.filtered = completions.filter(c => {
            // Search filter (handles question and subject)
            const matchSearch = state.filters.search === "" || 
                                c.question.toLowerCase().includes(state.filters.search) || 
                                c.subject.toLowerCase().includes(state.filters.search);
                                
            // Model filter
            const matchModel = state.filters.model === "all" || c.model === state.filters.model;
            
            // Subject Group filter
            const matchSubject = state.filters.subject === "all" || c.subject_group === state.filters.subject;
            
            // Outcome filter
            let matchOutcome = true;
            if (state.filters.outcome === "correct") {
                matchOutcome = c.is_correct === true && c.parsed_answer !== "unparseable";
            } else if (state.filters.outcome === "incorrect") {
                matchOutcome = c.is_correct === false && c.parsed_answer !== "unparseable";
            } else if (state.filters.outcome === "unparseable") {
                matchOutcome = !c.parsed_answer || c.parsed_answer === "unparseable";
            }
            
            return matchSearch && matchModel && matchSubject && matchOutcome;
        });
        
        renderCompletionsTable();
    }

    /**
     * Render paginated table rows inside Completions Explorer
     */
    function renderCompletionsTable() {
        const list = state.completions.filtered;
        const page = state.completions.currentPage;
        const size = state.completions.pageSize;
        const start = (page - 1) * size;
        const end = Math.min(start + size, list.length);
        const pageData = list.slice(start, end);
        
        const tbody = document.getElementById("completions-table-body");
        const paginationInfo = document.getElementById("pagination-info");
        const prevBtn = document.getElementById("prev-page-btn");
        const nextBtn = document.getElementById("next-page-btn");
        
        // Empty State Handler
        if (list.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6">
                        <div class="empty-state">
                            <i data-lucide="search-code"></i>
                            <h3>No completions match your criteria</h3>
                            <p>Try resetting filters or adjusting search terms</p>
                        </div>
                    </td>
                </tr>
            `;
            lucide.createIcons();
            paginationInfo.innerText = "Showing 0-0 of 0 records";
            prevBtn.disabled = true;
            nextBtn.disabled = true;
            return;
        }
        
        // Generate rows
        tbody.innerHTML = pageData.map((row) => {
            let outcomeBadge = '<span class="badge-outcome correct"><i data-lucide="check" style="width:12px; height:12px;"></i> Correct</span>';
            if (!row.parsed_answer || row.parsed_answer === "unparseable") {
                outcomeBadge = '<span class="badge-outcome unparseable"><i data-lucide="alert-octagon" style="width:12px; height:12px;"></i> Unparseable</span>';
            } else if (!row.is_correct) {
                outcomeBadge = '<span class="badge-outcome incorrect"><i data-lucide="x" style="width:12px; height:12px;"></i> Incorrect</span>';
            }
            
            return `
                <tr data-id="${row.response_id}">
                    <td class="row-question">${escapeHtml(row.question)}</td>
                    <td style="font-weight: 500;">${formatModelName(row.model)}</td>
                    <td>${formatSubject(row.subject)}</td>
                    <td style="font-weight: 700; color: var(--accent-blue); text-align: center;">${row.ground_truth}</td>
                    <td style="font-weight: 700; text-align: center;">${row.parsed_answer || "—"}</td>
                    <td>${outcomeBadge}</td>
                </tr>
            `;
        }).join("");
        
        // Re-bind SVG icons for dynamically generated items
        lucide.createIcons();
        
        // Row Click Listeners
        tbody.querySelectorAll("tr").forEach(tr => {
            tr.addEventListener("click", () => {
                const responseId = tr.getAttribute("data-id");
                const rowObj = data.completions.find(c => c.response_id === responseId);
                if (rowObj) {
                    openDrawer(rowObj);
                }
            });
        });
        
        // Update pagination states
        paginationInfo.innerText = `Showing ${list.length === 0 ? 0 : start + 1}-${end} of ${list.length} records`;
        prevBtn.disabled = page === 1;
        nextBtn.disabled = end >= list.length;
    }

    /**
     * Initialize sidebar drawer bindings (Close, Escape keys, overlay)
     */
    function initCompletionsDrawer() {
        const drawer = document.getElementById("completions-drawer");
        const overlay = document.getElementById("drawer-overlay");
        const closeBtn = document.getElementById("drawer-close-btn");
        
        const closeDrawer = () => {
            drawer.classList.remove("open");
            overlay.classList.remove("active");
            state.completions.selectedRow = null;
        };
        
        closeBtn.addEventListener("click", closeDrawer);
        overlay.addEventListener("click", closeDrawer);
        
        // Close on Escape Key
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && drawer.classList.contains("open")) {
                closeDrawer();
            }
        });
    }

    /**
     * Slide open the drawer and render full completion codeblocks & choices
     */
    function openDrawer(row) {
        state.completions.selectedRow = row;
        
        const drawer = document.getElementById("completions-drawer");
        const overlay = document.getElementById("drawer-overlay");
        
        // Populate Title and Headers
        document.getElementById("drawer-question-id").innerText = `Question ${row.question_id.split('-').pop() || row.question_id}`;
        document.getElementById("drawer-badge-model").innerText = formatModelName(row.model);
        document.getElementById("drawer-badge-subject").innerText = formatSubject(row.subject);
        
        // Populate Stats
        document.getElementById("drawer-stat-latency").innerText = `${row.latency_ms.toLocaleString()}ms`;
        document.getElementById("drawer-stat-input-tokens").innerText = row.input_tokens ? row.input_tokens.toLocaleString() : "—";
        document.getElementById("drawer-stat-output-tokens").innerText = row.output_tokens ? row.output_tokens.toLocaleString() : "—";
        
        // Render Question Text
        document.getElementById("drawer-question-text").innerText = row.question;
        
        // Render Multiple Choice list
        const choicesList = document.getElementById("drawer-choices-list");
        const choices = [
            { letter: "A", text: row.choice_a },
            { letter: "B", text: row.choice_b },
            { letter: "C", text: row.choice_c },
            { letter: "D", text: row.choice_d }
        ];
        
        choicesList.innerHTML = choices.map(choice => {
            let itemClass = "choice-item";
            
            // Highlighting states based on correctness
            if (choice.letter === row.ground_truth) {
                itemClass += " correct-answer";
            } else if (choice.letter === row.parsed_answer && !row.is_correct) {
                itemClass += " incorrect-selection";
            }
            
            return `
                <div class="${itemClass}">
                    <div class="choice-letter">${choice.letter}</div>
                    <div>${escapeHtml(choice.text)}</div>
                </div>
            `;
        }).join("");
        
        // Render Raw Completion String
        const block = document.getElementById("drawer-completion-text");
        block.innerText = row.raw_completion || "No completion response text captured.";
        
        // Slide open!
        drawer.classList.add("open");
        overlay.classList.add("active");
    }

    // ==================== FORMATTERS & UTILITIES ====================

    function formatModelName(name) {
        if (!name) return "";
        const m = name.toLowerCase();
        if (m.includes("haiku")) return "Claude Haiku 4.5";
        if (m.includes("sonnet")) return "Claude Sonnet 4.6";
        if (m.includes("opus")) return "Claude Opus 4.7";
        return name;
    }

    function formatSubjectGroup(group) {
        if (!group) return "";
        const mapping = {
            "stem": "STEM",
            "humanities": "Humanities",
            "social_science": "Social science",
            "other": "Other"
        };
        return mapping[group.toLowerCase()] || group;
    }

    function formatSubject(subject) {
        if (!subject) return "";
        return subject.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    function escapeHtml(text) {
        if (!text) return "";
        return text
            .toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function renderEmptyStates() {
        const overview = document.getElementById("overview");
        overview.innerHTML = `
            <div class="empty-state cyber-glass" style="margin-top: 50px;">
                <i data-lucide="database-backup" style="width: 60px; height: 60px;"></i>
                <h2>Evaluation Data Is Missing</h2>
                <p>Run <code>python scripts/export_dashboard_data.py</code> to extract DuckDB records and build the data layer.</p>
            </div>
        `;
        lucide.createIcons();
    }
});
