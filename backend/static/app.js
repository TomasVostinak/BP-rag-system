const API_BASE_URL = window.location.origin + "/api";
let cachedLeaderboardModels = [];
let lastRetrievedChunks = null;
let lastQuery = "";

let currentABTestModels = { model_a_name: "", model_b_name: "" };
let currentABTestQuery = "";

const TRUNCATE_LENGTH = 200;

function showStatus(elementId, message, type = 'info') {
    const statusElement = document.getElementById(elementId);
    statusElement.innerHTML = `<span class="${type}">${message}</span>`;
}

function showSpinner(elementId, message) {
    const statusElement = document.getElementById(elementId);
    statusElement.innerHTML = `<span class="spinner"></span> ${message}`;
}

async function runEval() {
    const modelName = document.getElementById("modelNameInput").value;
    showStatus('evalStatus', '');

    if (!modelName) {
        showStatus('evalStatus', 'Zadejte název modelu.', 'warning');
        return;
    }

    showSpinner('evalStatus', `Spouštím evaluaci pro ${modelName}...`);
    try {
        const response = await fetch(`${API_BASE_URL}/evaluate`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({model_name: modelName})
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Neznámá chyba při evaluaci.');
        }
        showStatus('evalStatus', result.message, 'info');
        document.getElementById("modelNameInput").value = '';
        await loadLeaderboard();
    } catch (error) {
        console.error('Chyba při evaluaci modelu:', error);
        showStatus('evalStatus', `Chyba: ${error.message}`, 'error');
    }
}

async function loadLeaderboard() {
    showSpinner('leaderboardStatus', 'Načítám leaderboard...');
    try {
        const res = await fetch(`${API_BASE_URL}/leaderboard`);
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        const data = await res.json();
        updateLeaderboardTable(data);
        updateModelCheckboxes(data);
        showStatus('leaderboardStatus', 'Leaderboard aktualizován.', 'info');
    } catch (error) {
        console.error('Chyba při načítání leaderboardu:', error);
        showStatus('leaderboardStatus', 'Chyba při načítání leaderboardu.', 'error');
    }
}

function updateLeaderboardTable(models) {
    const tableBody = document.querySelector('#leaderboardTable tbody');
    tableBody.innerHTML = '';

    if (models.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7">Zatím nejsou uloženy žádné výsledky.</td></tr>';
        return;
    }

    models.forEach(model => {
        const row = tableBody.insertRow();
        row.insertCell().textContent = model.placement;
        row.insertCell().textContent = model.model;
        row.insertCell().textContent = model.score.toFixed(4);
        row.insertCell().textContent = model['recall@10'].toFixed(4);
        row.insertCell().textContent = model['recall@20'].toFixed(4);
        row.insertCell().textContent = model['recall@30'].toFixed(4);
        row.insertCell().textContent = model.mrr.toFixed(4);
    });
}

function updateModelCheckboxes(models) {
    const checkboxesDiv = document.getElementById('modelCheckboxes');
    checkboxesDiv.innerHTML = '';
    cachedLeaderboardModels = models.map(m => m.model);

    if (models.length === 0) {
        checkboxesDiv.innerHTML = '<p class="warning">Není k dispozici žádný model pro porovnání. Spusťte evaluaci modelů nejprve.</p>';
        return;
    }

    cachedLeaderboardModels.forEach(modelName => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.name = 'compareModel';
        checkbox.value = modelName;
        checkbox.checked = true;

        const span = document.createElement('span');
        span.textContent = modelName;

        label.appendChild(checkbox);
        label.appendChild(span);
        checkboxesDiv.appendChild(label);
    });
}

async function compareModels() {
    const query = document.getElementById("queryInput").value;
    const topK = parseInt(document.getElementById("topKSlider").value);
    const resultsDiv = document.getElementById('retrievedChunksResults'); // Tady je reference OK
    const generateBtn = document.getElementById('generateLLMResponseBtn');
    
    showStatus('compareStatus', '');
    resultsDiv.innerHTML = ''; // Tohle je taky OK
    generateBtn.disabled = true;

    if (!query) {
        showStatus('compareStatus', 'Zadejte otázku.', 'warning');
        return;
    }

    const selectedModels = Array.from(document.querySelectorAll('#modelCheckboxes input[name="compareModel"]:checked'))
                                .map(cb => cb.value);
    if (selectedModels.length === 0) {
        showStatus('compareStatus', 'Vyberte alespoň jeden model pro porovnání.', 'warning');
        return;
    }

    showSpinner('compareStatus', 'Vyhledávám relevantní chunky...');
    try {
        const response = await fetch(`${API_BASE_URL}/compare`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ query: query, top_k: topK, models_to_compare: selectedModels })
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Neznámá chyba při vyhledávání.');
        }

        lastRetrievedChunks = result.model_results;
        lastQuery = query;
        displayRetrievedChunks(result.model_results, result.duplicate_ids, 'retrievedChunksResults');
        showStatus('compareStatus', 'Vyhledávání dokončeno.', 'info');
        generateBtn.disabled = false;
    } catch (error) {
        console.error('Chyba při vyhledávání chunků:', error);
        showStatus('compareStatus', `Chyba: ${error.message}`, 'error');
        lastRetrievedChunks = null;
    }
}

function displayRetrievedChunks(modelResults, duplicateIds, targetDivId) {
    const container = document.getElementById(targetDivId);
    container.innerHTML = "";

    const sortedModelNames = cachedLeaderboardModels.filter(modelName => modelResults.hasOwnProperty(modelName));

    for (const modelName of sortedModelNames) {
        const chunks = modelResults[modelName];

        let html = `<h3>${modelName}</h3><table>`;
        html += "<tr><th>Rank</th><th>Chunk ID</th><th>Text</th></tr>";

        if (chunks.length === 0) {
            html += `<tr><td colspan="3">Nenalezeny žádné relevantní chunky pro tento model.</td></tr>`;
        } else {
            chunks.forEach(c => {
                const highlight = duplicateIds.includes(c.chunk_id)
                    ? "class='highlight-duplicate'"
                    : "";
                
                const truncatedText = c.full_text.length > TRUNCATE_LENGTH 
                    ? c.full_text.substring(0, TRUNCATE_LENGTH) + '...'
                    : c.full_text;

                html += `
                    <tr ${highlight}>
                        <td>${c.rank}</td>
                        <td>${c.chunk_id}</td>
                        <td class="chunk-text">
                            <span class="truncated">${truncatedText}</span>
                            ${c.full_text.length > TRUNCATE_LENGTH ? 
                                `<span class="full-text" style="display:none;">${c.full_text}</span>
                                <button class="toggle-text-btn" onclick="toggleChunkText(this)">Zobrazit více</button>` : ''}
                        </td>
                    </tr>
                `;
            });
        }
        html += "</table>";
        container.innerHTML += html;
    }
}

function toggleChunkText(button) {
    const td = button.closest('td');
    const truncatedSpan = td.querySelector('.truncated');
    const fullTextSpan = td.querySelector('.full-text');

    if (truncatedSpan.style.display === 'none') {
        truncatedSpan.style.display = 'inline';
        fullTextSpan.style.display = 'none';
        button.textContent = 'Zobrazit více';
    } else {
        truncatedSpan.style.display = 'none';
        fullTextSpan.style.display = 'inline';
        button.textContent = 'Zobrazit méně';
    }
}

async function generateLLMResponse() {
    const generateStatus = document.getElementById('generateStatus');
    const llmResponseOutput = document.getElementById('llmResponseOutput');
    llmResponseOutput.innerHTML = '';
    showStatus('generateStatus', '');

    if (!lastRetrievedChunks || Object.keys(lastRetrievedChunks).length === 0) {
        showStatus('generateStatus', 'Nejprve proveďte porovnání modelů.', 'warning');
        return;
    }

    const bestModelName = cachedLeaderboardModels[0];
    const contextChunks = lastRetrievedChunks[bestModelName];

    if (!contextChunks || contextChunks.length === 0) {
        showStatus('generateStatus', `Pro nejlepší model (${bestModelName}) nebyl nalezen žádný kontext k vygenerování odpovědi.`, 'warning');
        return;
    }

    const context = contextChunks.map(c => c.full_text).join('\n\n');

    showSpinner('generateStatus', `Generuji odpověď pomocí LLM (kontext z modelu: ${bestModelName})...`);
    try {
        const response = await fetch(`${API_BASE_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: lastQuery, context: context })
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Neznámá chyba při generování odpovědi.');
        }
        llmResponseOutput.innerHTML = `
            <p><strong>Odpověď generována z kontextu modelu:</strong> <em>${bestModelName}</em></p>
            ${marked.parse(result.answer)}
        `;
        showStatus('generateStatus', 'Odpověď LLM vygenerována.', 'info');
    } catch (error) {
        console.error('Chyba při generování odpovědi LLM:', error);
        llmResponseOutput.innerHTML = '<span class="error">Chyba při generování odpovědi LLM.</span>';
        showStatus('generateStatus', `Chyba: ${error.message}`, 'error');
    }
}

async function startABTest() {
    const abTestQuery = document.getElementById('abTestQueryInput').value;
    const topK = parseInt(document.getElementById("topKSlider").value);
    showStatus('abTestStatus', '');
    document.getElementById('abTestComparison').style.display = 'none';

    if (!abTestQuery) {
        showStatus('abTestStatus', 'Zadejte otázku pro A/B test.', 'warning');
        return;
    }
    currentABTestQuery = abTestQuery;

    if (cachedLeaderboardModels.length < 2) {
        showStatus('abTestStatus', 'Nedostatek evaluovaných modelů pro A/B testování (potřeba alespoň 2).', 'warning');
        return;
    }

    showSpinner('abTestStatus', 'Vybírám modely a generuji odpovědi pro A/B test...');
    try {
        const pairResponse = await fetch(`${API_BASE_URL}/abtest/get_pair`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ query: abTestQuery, top_k: topK })
        });
        if (!pairResponse.ok) {
            throw new Error((await pairResponse.json()).detail || 'Chyba při získávání A/B test páru.');
        }
        const result = await pairResponse.json();
        
        currentABTestModels.model_a_name = result.model_a_name_internal;
        currentABTestModels.model_b_name = result.model_b_name_internal;

        const responseA = result.display_order[0] === 'model_a' ? result.response_a : result.response_b;
        const responseB = result.display_order[0] === 'model_a' ? result.response_b : result.response_a;

        document.getElementById('abTestResponseA').innerHTML = marked.parse(responseA);
        document.getElementById('abTestResponseB').innerHTML = marked.parse(responseB);
        
        document.getElementById('abTestComparison').style.display = 'flex';
        showStatus('abTestStatus', 'A/B test připraven.', 'info');

    } catch (error) {
        console.error('Chyba při A/B testování:', error);
        showStatus('abTestStatus', `Chyba při A/B testování: ${error.message}`, 'error');
        document.getElementById('abTestComparison').style.display = 'none';
    }
}

async function submitABTestResult(winner) {
    showSpinner('abTestStatus', 'Ukládám výsledek A/B testu...');
    try {
        const response = await fetch(`${API_BASE_URL}/abtest/submit_result`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: currentABTestQuery,
                model_a_name: currentABTestModels.model_a_name,
                model_b_name: currentABTestModels.model_b_name,
                winner: winner
            })
        });

        const result = await response.json();
        if (!response.ok) {
            let errorMessage = result.detail;
            if (Array.isArray(errorMessage)) {
                errorMessage = errorMessage.map(err => `${err.loc.join('.')} - ${err.msg}`).join('; ');
            } else if (typeof errorMessage === 'object' && errorMessage !== null) {
                errorMessage = JSON.stringify(errorMessage);
            }
            throw new Error(errorMessage || 'Neznámá chyba při ukládání výsledku A/B testu.');
        }
        showStatus('abTestStatus', result.message, 'info');
        document.getElementById('abTestComparison').style.display = 'none';
    } catch (error) {
        console.error('Chyba při ukládání výsledku A/B testu:', error);
        showStatus('abTestStatus', `Chyba při ukládání výsledku: ${error.message}`, 'error');
    }
}

window.onload = () => {
    loadLeaderboard();
};