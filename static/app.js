// 🚀 Post Recommendation System - Script

// DOM Elements
const form = document.getElementById("recommendationForm");
const titleInput = document.getElementById("titleInput");
const numRecommendations = document.getElementById("numRecommendations");
const resultsSection = document.getElementById("resultsSection");
const resultsContainer = document.getElementById("resultsContainer");
const resultsTitle = document.getElementById("resultsTitle");
const loading = document.getElementById("loading");
const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");

// Utility
function showLoading(show) {
    loading.classList.toggle("hidden", !show);
}
function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove("hidden");
}
function hideError() {
    errorSection.classList.add("hidden");
}
function renderResults(title, items) {
    resultsTitle.textContent = `📋 Recommendations for "${title}"`;
    resultsContainer.innerHTML = "";
    items.forEach(item => {
        const card = document.createElement("div");
        card.className = "result-card";
        card.innerHTML = `<strong>${item.title}</strong> <br> Score: ${item.score || "N/A"}`;
        resultsContainer.appendChild(card);
    });
    resultsSection.classList.remove("hidden");
}
function clearResults() {
    resultsSection.classList.add("hidden");
    resultsContainer.innerHTML = "";
}

// API calls
async function fetchAPI(endpoint, options = {}) {
    try {
        showLoading(true);
        hideError();
        const response = await fetch(endpoint, options);
        if (!response.ok) throw new Error(await response.text());
        return await response.json();
    } catch (err) {
        showError(err.message);
        return null;
    } finally {
        showLoading(false);
    }
}

// Submit form
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const title = titleInput.value.trim();
    const n = parseInt(numRecommendations.value, 10) || 5;
    const data = await fetchAPI("/api/recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, n_recommendations: n })
    });
    if (data && data.success) {
        renderResults(data.query_title, data.recommendations);
    }
});

// Random Recommendations
async function getRecommendations() {
  let response = await fetch("/api/random");
  let data = await response.json();
  document.getElementById("results").innerText = JSON.stringify(data, null, 2);
}

// Top Posts
async function getTopPosts() {
    const data = await fetchAPI(`/api/top-posts?top_n=${numRecommendations.value}`);
    if (data && data.success) {
        renderResults("Top Posts", data.top_posts);
    }
}

// Keyboard Shortcuts
document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") form.requestSubmit();
    if (e.ctrlKey && e.key.toLowerCase() === "r") getRandomRecommendations();
    if (e.key === "Escape") clearResults();
});
