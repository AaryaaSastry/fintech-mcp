/**
 * Fintech AI Analytics - Chat-First Logic
 * The application starts with a conversational gateway and reveals data insights
 * only after the initial interaction.
 */

const API_BASE = "http://localhost:8000/api/v1";

// Global Chart Instances
let trendChart = null;
let categoryChart = null;

// State
let dashboardUnlocked = false;

// DOM Elements
const clientSelect = document.getElementById("client-id-select");
const heroSection = document.getElementById("hero-section");
const mainInput = document.getElementById("main-ai-input");
const mainSend = document.getElementById("main-ai-send");
const chatHistory = document.getElementById("main-chat-history");
const dashboardContent = document.getElementById("dashboard-content");

/**
 * Initialization
 */
document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Hero Chat interaction
    mainSend.addEventListener("click", handleMainQuery);
    mainInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") handleMainQuery();
    });

    // Client selection change (refresh dashboard if active)
    clientSelect.addEventListener("change", (e) => {
        if (dashboardUnlocked) {
            loadDashboardData(e.target.value);
        }
    });
}

/**
 * Main Flow Controller
 */
async function handleMainQuery() {
    const text = mainInput.value.trim();
    if (!text) return;

    // 1. Show message in history
    if (!dashboardUnlocked) {
        chatHistory.classList.remove("dashboard-hidden");
    }
    appendBubble(text, 'user');
    mainInput.value = "";

    // 2. Process query with AI
    const typingId = "typing-" + Date.now();
    appendBubble("Thinking...", 'ai', typingId);

    try {
        const response = await fetch(`${API_BASE}/conversation/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text + (dashboardUnlocked ? ` for client ${clientSelect.value}` : "")
            })
        });
        const data = await response.json();

        // Remove typing indicator
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();

        // 3. Display AI response
        appendBubble(data.message, 'ai');

        // 4. Unlock Dashboard if not already
        if (!dashboardUnlocked) {
            unlockDashboard();
        }

        // 5. Always refresh data to match the conversation
        loadDashboardData(clientSelect.value);

    } catch (e) {
        console.error("AI Error:", e);
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.textContent = "I'm having trouble connecting to my analytics engine right now.";
    }
}

/**
 * UI Transitions
 */
function unlockDashboard() {
    dashboardUnlocked = true;

    // Transition Hero Section
    heroSection.classList.add("minimized");

    // Reveal Dashboard Content
    dashboardContent.classList.remove("dashboard-hidden");

    // Animate reveal
    dashboardContent.style.opacity = "0";
    setTimeout(() => {
        dashboardContent.style.transition = "opacity 0.8s ease-in";
        dashboardContent.style.opacity = "1";
    }, 100);
}

function appendBubble(text, sender, id = null) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${sender}`;
    if (id) bubble.id = id;

    // Support markdown-like simple formatting for conversational feel
    bubble.textContent = text;

    chatHistory.appendChild(bubble);

    // Smooth scroll to latest bubble
    chatHistory.scrollTop = chatHistory.scrollHeight;
    bubble.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

/**
 * Data Fetching & Visualization
 */
async function loadDashboardData(clientId) {
    console.log(`Updating dashboard for client: ${clientId}`);

    try {
        // Parallel fetch for speed
        const [spending, trend, fraud, credit, forecast] = await Promise.all([
            fetchToolData("get_spending_summary", { client_id: clientId, months: 6 }),
            fetchToolData("monthly_spending_trend", { client_id: clientId }),
            fetchToolData("client_fraud_risk", { client_id: clientId }),
            fetchToolData("credit_score_history", { client_id: clientId }),
            fetchToolData("spending_forecast", { client_id: clientId, months: 3 })
        ]);

        updateStats(spending, credit, fraud, forecast);
        renderCharts(trend, spending);
        loadTransactions(clientId);

    } catch (error) {
        console.warn("Dashboard sync issue:", error);
    }
}

async function fetchToolData(toolName, params) {
    const endpoint = toolName === "get_spending_summary" ? "/analytics/spending" :
        toolName === "monthly_spending_trend" ? "/chat?q=trend" :
            toolName === "client_fraud_risk" ? "/analytics/fraud-check" :
                toolName === "credit_score_history" ? "/chat?q=credit score" :
                    "/analytics/forecast";

    try {
        const response = await fetch(`${API_BASE}${endpoint.includes('?') ? endpoint + " for client " + params.client_id : endpoint}`, {
            method: endpoint.includes('?') ? "GET" : "POST",
            headers: { "Content-Type": "application/json" },
            body: endpoint.includes('?') ? null : JSON.stringify(params)
        });
        const data = await response.json();
        return data.raw_data ? Object.values(data.raw_data)[0] : data;
    } catch (e) {
        return { error: e.message };
    }
}

function updateStats(spending, credit, fraud, forecast) {
    document.getElementById("stat-total-spending").textContent = formatCurrency(spending.total_spent || 0);
    document.getElementById("stat-credit-score").textContent = credit.credit_score || "---";
    document.getElementById("stat-credit-grade").textContent = credit.grade ? credit.grade.toUpperCase() : "---";
    document.getElementById("stat-fraud-prob").textContent = `${((fraud.risk_score || 0) * 100).toFixed(1)}%`;

    const fraudLevel = document.getElementById("stat-fraud-level");
    fraudLevel.textContent = fraud.risk_level || "Secure";
    fraudLevel.style.color = fraud.risk_level === 'high' ? 'var(--danger)' : 'var(--primary)';

    const forecastVal = forecast.forecast ? Object.values(forecast.forecast)[0] : 0;
    document.getElementById("stat-forecast").textContent = formatCurrency(forecastVal);
}

function renderCharts(trend, spending) {
    // Trend Chart
    const trendCtx = document.getElementById("spending-trend-chart").getContext("2d");
    if (trendChart) trendChart.destroy();

    const months = trend.months ? Object.keys(trend.months) : [];
    const values = trend.months ? Object.values(trend.months) : [];

    trendChart = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: months,
            datasets: [{
                label: 'Monthly',
                data: values,
                borderColor: '#52ecab',
                backgroundColor: 'rgba(82, 236, 171, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 2
            }]
        },
        options: chartOptions
    });

    // Category Chart
    const catCtx = document.getElementById("category-pie-chart").getContext("2d");
    if (categoryChart) categoryChart.destroy();

    const cats = spending.top_merchants ? Object.keys(spending.top_merchants) : [];
    const catValues = spending.top_merchants ? Object.values(spending.top_merchants) : [];

    categoryChart = new Chart(catCtx, {
        type: 'doughnut',
        data: {
            labels: cats,
            datasets: [{
                data: catValues,
                backgroundColor: ['#52ecab', '#3b82f6', '#f59e0b', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            ...chartOptions,
            cutout: '75%'
        }
    });
}

async function loadTransactions(clientId) {
    const list = document.getElementById("transactions-list");
    list.innerHTML = "";
    const data = await fetchToolData("high_risk_transactions", { client_id: clientId, threshold: 200 });
    const txs = data.transactions || [];

    if (txs.length === 0) {
        list.innerHTML = '<p style="color: var(--text-dim); text-align: center;">No significant items flags.</p>';
        return;
    }

    txs.slice(0, 4).forEach(tx => {
        const item = document.createElement("div");
        item.style.padding = "10px";
        item.style.borderBottom = "1px solid var(--border-color)";
        item.style.display = "flex";
        item.style.justifyContent = "space-between";
        item.innerHTML = `<span>${tx.merchant_category}</span> <b>${formatCurrency(tx.amount)}</b>`;
        list.appendChild(item);
    });
}

function formatCurrency(val) {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
}

const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
        x: { grid: { display: false }, ticks: { color: '#64748b' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } }
    }
};
