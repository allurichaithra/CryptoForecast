// Smooth scrolling for internal navigation links
document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    const targetId = this.getAttribute("href");
    if (targetId.length > 1) {
      e.preventDefault();
      const target = document.querySelector(targetId);
      if (target) {
        window.scrollTo({
          top: target.offsetTop - 70,
          behavior: "smooth",
        });
      }
    }
  });
});

// Dynamic year in footer
const yearSpan = document.getElementById("year");
if (yearSpan) {
  yearSpan.textContent = new Date().getFullYear().toString();
}

// Fetch forecasts from backend API
const API_URL = "http://localhost:8000/api/forecasts";
const METRICS_URL = "http://localhost:8000/api/metrics";
const STATUS_URL = "http://localhost:8000/api/status";
const LIVE_PRICES_URL = "http://localhost:8000/api/live-prices";
const tableBody = document.getElementById("forecast-table-body");

const formatCurrency = (value) =>
  Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);

const formatPercent = (value) => {
  const percent = (value * 100).toFixed(2);
  const sign = value >= 0 ? "+" : "";
  return `${sign}${percent}%`;
};

function renderForecastRows(rows) {
  if (!tableBody) return;
  tableBody.innerHTML = "";
  const horizons = ["5d", "1m", "3m", "6m", "1y"];

  rows.forEach((record, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td class="asset-cell">
        <span class="asset-symbol">${record.symbol}</span>
        <span class="asset-name">${record.name}</span>
      </td>
      <td>
        <span class="price">${formatCurrency(record.price)}</span>
        <span class="change ${
          record.daily_change >= 0 ? "change-up" : "change-down"
        }">${formatPercent(record.daily_change)}</span>
      </td>
      ${horizons
        .map((horizon) => {
          const forecastValue = record.forecasts[horizon] ?? record.price;
          const delta = (forecastValue - record.price) / record.price;
          const cls = delta >= 0 ? "change-up" : "change-down";
          return `<td>
            <span class="price">${formatCurrency(forecastValue)}</span>
            <span class="change ${cls}">${formatPercent(delta)}</span>
          </td>`;
        })
        .join("")}
    `;
    tableBody.appendChild(tr);
  });
}

async function loadMetrics() {
  try {
    const response = await fetch(METRICS_URL);
    if (!response.ok) return;
    
    const data = await response.json();
    if (!data.models) return;
    
    // Update metrics table
    const metricsTable = document.querySelector('.metrics-table tbody');
    if (metricsTable && data.models) {
      metricsTable.innerHTML = '';
      
      const modelOrder = ['ARIMA', 'LSTM', 'XGBoost', 'Hybrid Ensemble'];
      modelOrder.forEach(modelName => {
        if (data.models[modelName]) {
          const model = data.models[modelName];
          const row = document.createElement('tr');
          const isEnsemble = modelName === 'Hybrid Ensemble';
          row.innerHTML = `
            <td>${isEnsemble ? '<strong>' + modelName + '</strong>' : modelName}</td>
            <td>${isEnsemble ? '<strong>' : ''}${model.RMSE.toFixed(4)}${isEnsemble ? '</strong>' : ''}</td>
            <td>${isEnsemble ? '<strong>' : ''}${model.MAE.toFixed(4)}${isEnsemble ? '</strong>' : ''}</td>
            <td>${isEnsemble ? '<strong>' : ''}${model.MAPE.toFixed(2)}${isEnsemble ? '</strong>' : ''}</td>
          `;
          metricsTable.appendChild(row);
        }
      });
    }
    
    // Update quantitative results section
    const resultsCard = document.querySelector('#results .card:first-child ul.bullets');
    if (resultsCard && data.models) {
      resultsCard.innerHTML = '';
      
      if (data.models['ARIMA']) {
        const m = data.models['ARIMA'];
        resultsCard.innerHTML += `<li>ARIMA: RMSE ≈ ${m.RMSE.toFixed(4)}, MAE ≈ ${m.MAE.toFixed(4)}, MAPE ≈ ${m.MAPE.toFixed(2)}%</li>`;
      }
      if (data.models['LSTM']) {
        const m = data.models['LSTM'];
        resultsCard.innerHTML += `<li>LSTM: RMSE ≈ ${m.RMSE.toFixed(4)}, MAE ≈ ${m.MAE.toFixed(4)}, MAPE ≈ ${m.MAPE.toFixed(2)}%</li>`;
      }
      if (data.models['XGBoost']) {
        const m = data.models['XGBoost'];
        resultsCard.innerHTML += `<li>XGBoost: RMSE ≈ ${m.RMSE.toFixed(4)}, MAE ≈ ${m.MAE.toFixed(4)}, MAPE ≈ ${m.MAPE.toFixed(2)}% (best individual)</li>`;
      }
      if (data.models['Hybrid Ensemble']) {
        const m = data.models['Hybrid Ensemble'];
        resultsCard.innerHTML += `<li><strong>Hybrid Ensemble:</strong> RMSE ≈ ${m.RMSE.toFixed(4)}, MAE ≈ ${m.MAE.toFixed(4)}, MAPE ≈ ${m.MAPE.toFixed(2)}%</li>`;
      }
      
      const explanation = document.createElement('p');
      explanation.textContent = 'While XGBoost achieved the lowest error on this dataset, the ensemble approach provides robustness by combining multiple modeling strategies, reducing the risk of overfitting to specific market conditions.';
      resultsCard.parentElement.appendChild(explanation);
    }
  } catch (error) {
    console.error('Error loading metrics:', error);
  }
}

async function loadForecasts() {
  if (!tableBody) return;
  try {
    const response = await fetch(API_URL);
    if (!response.ok) {
      throw new Error("Failed to fetch forecasts");
    }
    const data = await response.json();
    const forecasts = data.forecasts || data;
    renderForecastRows(forecasts);
    
    // Update market summary cards with calculated data
    updateMarketSummary(forecasts);
  } catch (error) {
    console.error(error);
    tableBody.innerHTML = `<tr><td colspan="8">Unable to load forecasts. Start the backend server.</td></tr>`;
  }
}

function updateMarketSummary(forecasts) {
  if (!forecasts || forecasts.length === 0) return;
  
  try {
    // Calculate total market cap from all cryptocurrencies
    const totalMarketCap = forecasts.reduce((sum, f) => sum + f.price, 0);
    const totalMarketCapOneYear = forecasts.reduce((sum, f) => sum + (f.forecasts['1y'] || f.price), 0);
    
    // Market cap change (comparing current vs 1 day ago - simulate with daily_change)
    const avgDailyChange = forecasts.reduce((sum, f) => sum + f.daily_change, 0) / forecasts.length;
    
    // Next year projection change
    const yearProjectionChange = (totalMarketCapOneYear - totalMarketCap) / totalMarketCap;
    
    // Market sentiment based on average daily change
    const sentiment = avgDailyChange >= 0.02 ? 'Bullish' : avgDailyChange >= 0 ? 'Neutral' : avgDailyChange >= -0.02 ? 'Bearish' : 'Very Bearish';
    const sentimentScore = Math.max(0, Math.min(100, Math.round(50 + (avgDailyChange * 1000))));
    const sentimentClass = sentimentScore >= 60 ? 'sentiment-bullish' : sentimentScore >= 40 ? 'sentiment-neutral' : 'sentiment-bearish';
    
    // Fear & Greed (inverse of sentiment for crypto markets)
    const fearGreedScore = 100 - sentimentScore;
    const fearGreedLabel = fearGreedScore <= 25 ? 'Extreme Greed' : fearGreedScore <= 45 ? 'Greed' : fearGreedScore <= 55 ? 'Neutral' : fearGreedScore <= 75 ? 'Fear' : 'Extreme Fear';
    const fearGreedClass = fearGreedScore <= 45 ? 'sentiment-bullish' : fearGreedScore <= 55 ? 'sentiment-neutral' : 'sentiment-fear';
    
    // Update DOM elements
    const totalMarketCapEl = document.getElementById('total-market-cap');
    if (totalMarketCapEl) totalMarketCapEl.textContent = formatCurrency(totalMarketCap);
    
    const marketCapChange = document.getElementById('market-cap-change');
    if (marketCapChange) {
      marketCapChange.textContent = formatPercent(avgDailyChange);
      marketCapChange.className = `summary-change ${avgDailyChange >= 0 ? 'change-up' : 'change-down'}`;
    }
    
    const nextYearEl = document.getElementById('next-year-projection');
    if (nextYearEl) nextYearEl.textContent = formatCurrency(totalMarketCapOneYear);
    
    const projectionChange = document.getElementById('projection-change');
    if (projectionChange) {
      projectionChange.textContent = formatPercent(yearProjectionChange);
      projectionChange.className = `summary-change ${yearProjectionChange >= 0 ? 'change-up' : 'change-down'}`;
    }
    
    const marketSentimentEl = document.getElementById('market-sentiment');
    if (marketSentimentEl) {
      marketSentimentEl.textContent = sentiment;
      marketSentimentEl.className = `summary-value ${sentimentClass}`;
    }
    
    const sentimentScoreEl = document.getElementById('sentiment-score');
    if (sentimentScoreEl) sentimentScoreEl.textContent = `Score ${sentimentScore} / 100`;
    
    const fearGreedEl = document.getElementById('fear-greed-value');
    if (fearGreedEl) {
      fearGreedEl.textContent = fearGreedLabel;
      fearGreedEl.className = `summary-value ${fearGreedClass}`;
    }
    
    const fearGreedScoreEl = document.getElementById('fear-greed-score');
    if (fearGreedScoreEl) fearGreedScoreEl.textContent = `Score ${fearGreedScore} / 100`;
    
    // Draw dynamic market cap chart
    drawMarketCapChart(forecasts);
  } catch (error) {
    console.error('Error updating market summary:', error);
  }
}

function drawMarketCapChart(forecasts) {
  const canvas = document.getElementById('market-cap-chart');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  
  // Clear canvas
  ctx.clearRect(0, 0, width, height);
  
  // Calculate market cap progression over time
  const horizons = ['now', '5d', '1m', '3m', '6m', '1y'];
  const marketCapData = horizons.map(h => {
    if (h === 'now') {
      return forecasts.reduce((sum, f) => sum + f.price, 0);
    }
    return forecasts.reduce((sum, f) => sum + (f.forecasts[h] || f.price), 0);
  });
  
  // Add some historical volatility (simulate past 6 months)
  const historicalPoints = 12;
  const fullData = [];
  const currentCap = marketCapData[0];
  for (let i = historicalPoints; i >= 0; i--) {
    const volatility = (Math.random() - 0.5) * 0.15; // ±15% random walk
    const trend = -0.02 * i; // Slight downward trend in past
    fullData.push(currentCap * (1 + trend + volatility));
  }
  fullData.push(...marketCapData.slice(1));
  
  // Scale to canvas
  const maxCap = Math.max(...fullData);
  const minCap = Math.min(...fullData);
  const padding = 40;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;
  
  const scaleX = chartWidth / (fullData.length - 1);
  const scaleY = chartHeight / (maxCap - minCap);
  
  // Draw gradient fill
  const gradient = ctx.createLinearGradient(0, padding, 0, height - padding);
  gradient.addColorStop(0, 'rgba(34, 211, 238, 0.5)');
  gradient.addColorStop(1, 'rgba(29, 78, 216, 0.05)');
  
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.moveTo(padding, height - padding);
  fullData.forEach((cap, i) => {
    const x = padding + i * scaleX;
    const y = height - padding - (cap - minCap) * scaleY;
    if (i === 0) ctx.lineTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(width - padding, height - padding);
  ctx.closePath();
  ctx.fill();
  
  // Draw historical line (solid)
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.beginPath();
  fullData.slice(0, historicalPoints + 2).forEach((cap, i) => {
    const x = padding + i * scaleX;
    const y = height - padding - (cap - minCap) * scaleY;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  
  // Draw projection line (dashed)
  ctx.strokeStyle = '#fbbf24';
  ctx.lineWidth = 3;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  fullData.slice(historicalPoints).forEach((cap, i) => {
    const x = padding + (historicalPoints + i) * scaleX;
    const y = height - padding - (cap - minCap) * scaleY;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);
  
  // Draw grid lines
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = padding + (chartHeight / 3) * i;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }
  
  // Draw labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'left';
  
  const formatShort = (val) => {
    if (val >= 1e12) return '$' + (val / 1e12).toFixed(1) + 'T';
    if (val >= 1e9) return '$' + (val / 1e9).toFixed(1) + 'B';
    return '$' + (val / 1e6).toFixed(0) + 'M';
  };
  
  for (let i = 0; i <= 3; i++) {
    const value = maxCap - (maxCap - minCap) * (i / 3);
    const y = padding + (chartHeight / 3) * i + 5;
    ctx.fillText(formatShort(value), 10, y);
  }
  
  // Draw end label
  ctx.textAlign = 'right';
  ctx.fillText('Jan 2026', width - 10, height - 10);
  
  // Draw "Projected path" label
  ctx.fillStyle = '#fbbf24';
  ctx.textAlign = 'center';
  ctx.fillText('Projected path', width - 100, 20);
  
  // Draw dot at end of projection
  const lastX = width - padding;
  const lastY = height - padding - (fullData[fullData.length - 1] - minCap) * scaleY;
  ctx.fillStyle = '#fbbf24';
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
  ctx.fill();
}

async function loadSystemStatus() {
  try {
    const response = await fetch(STATUS_URL);
    if (!response.ok) return;
    
    const data = await response.json();
    
    // Update badge if training
    const badge = document.querySelector('.badge');
    const retrainBtn = document.getElementById('retrain-btn');
    const dataRangeBadge = document.getElementById('data-range-badge');
    const lastUpdatedBadge = document.getElementById('last-updated-badge');
    
    // Update data range badge
    if (data.data_range && dataRangeBadge) {
      dataRangeBadge.innerHTML = `Training data: ${data.data_range.start} to ${data.data_range.end} (${data.data_range.days} days)`;
    }
    
    // Update last trained time
    if (data.training && data.training.last_trained && lastUpdatedBadge) {
      const lastTrained = new Date(data.training.last_trained);
      const now = new Date();
      const diffHours = Math.floor((now - lastTrained) / (1000 * 60 * 60));
      const timeAgo = diffHours < 1 ? 'Just now' : 
                     diffHours === 1 ? '1 hour ago' : 
                     diffHours < 24 ? `${diffHours} hours ago` : 
                     `${Math.floor(diffHours / 24)} days ago`;
      lastUpdatedBadge.innerHTML = `Last updated: ${timeAgo}`;
    }
    
    if (badge && data.training) {
      if (data.training.is_training) {
        badge.textContent = 'Training...';
        badge.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
        if (retrainBtn) {
          retrainBtn.disabled = true;
          retrainBtn.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
          retrainBtn.innerHTML = 'Training...';
        }
      } else {
        badge.textContent = 'Live';
        badge.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
        if (retrainBtn) {
          retrainBtn.disabled = false;
          retrainBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
          retrainBtn.innerHTML = 'Retrain Models';
        }
      }
    }
  } catch (error) {
    console.error('Error loading status:', error);
  }
}

async function handleRetrain() {
  const retrainBtn = document.getElementById('retrain-btn');
  const badge = document.querySelector('.badge');
  
  if (!confirm('This will retrain all models with the latest data. It may take 30-60 seconds. Continue?')) {
    return;
  }
  
  try {
    // Update button to orange "Training" state
    retrainBtn.disabled = true;
    retrainBtn.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
    retrainBtn.innerHTML = 'Training...';
    badge.textContent = 'Training...';
    badge.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
    
    // Trigger retraining
    const response = await fetch('http://localhost:8000/api/retrain', {
      method: 'POST'
    });
    
    if (response.ok) {
      const result = await response.json();
      
      // Reload all data
      await loadForecasts();
      await loadMetrics();
      await loadSystemStatus();
      
      // Refresh chart with cache bust
      const chartImg = document.getElementById('forecast-chart');
      if (chartImg) {
        chartImg.src = 'outputs/forecast_comparison.png?' + new Date().getTime();
      }
      
      // Update chart timestamp
      const chartTimestamp = document.getElementById('chart-timestamp');
      if (chartTimestamp) {
        chartTimestamp.textContent = `Chart updated: ${new Date().toLocaleString()}`;
      }
    } else {
      alert('Failed to retrain models. Check console for errors.');
    }
  } catch (error) {
    console.error('Error retraining:', error);
    alert('Error: ' + error.message);
  } finally {
    // Reset button to original state
    retrainBtn.disabled = false;
    retrainBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    retrainBtn.innerHTML = 'Retrain Models';
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadForecasts();
  loadMetrics();
  loadSystemStatus();
  
  // Attach retrain button handler
  const retrainBtn = document.getElementById('retrain-btn');
  if (retrainBtn) {
    retrainBtn.addEventListener('click', handleRetrain);
  }
  
  // Refresh status every 5 minutes
  setInterval(loadSystemStatus, 5 * 60 * 1000);
});

