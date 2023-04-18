document.addEventListener('DOMContentLoaded', () => {
    // API endpoint prefix
    const API_PREFIX = '/molecule-net';
    
    // Registry for model and property configurations
    let MODEL_REGISTRY = {};
    
    // UI Elements
    const moleculeForm = document.getElementById('molecule-form');
    const moleculeInput = document.getElementById('molecule-input');
    const aboutSection = document.getElementById('about-section');
    const resultsSection = document.getElementById('results-section');
    const predictionResults = document.getElementById('prediction-results');
    const moleculeDetails = document.getElementById('molecule-details');
    const backButton = document.getElementById('back-button');
    const predictionOptionsContainer = document.getElementById('prediction-options-container');
    const presetsButtons = document.getElementById('presets-buttons');
    const selectionSummary = document.getElementById('selection-summary');
    const selectionCount = document.getElementById('selection-count');
    const availabilityMessage = document.getElementById('availability-message');
    const runPredictionsBtn = document.getElementById('run-predictions-btn');
    const zeroState = document.getElementById('zero-state');
    
    // Track selected model IDs
    let selectedModelIds = new Set();
    
    // Initialize: Fetch model registry and set up the UI
    async function initializeApp() {
        try {
            // Fetch model registry from server
            const response = await fetch(`${API_PREFIX}/models`);
            
            if (!response.ok) {
                throw new Error("Failed to load model registry");
            }
            
            MODEL_REGISTRY = await response.json();
            console.log("Model registry loaded:", MODEL_REGISTRY);
            
            // Build schema-driven UI
            buildSchemaDrivenUI();
            buildPresets();
            updateSelectionSummary();
            
        } catch (error) {
            console.error("Error initializing app:", error);
            alert("Failed to initialize application. Please reload the page or contact support.");
        }
    }
    
    /**
     * Build the schema-driven UI from model_registry.json
     * Groups models by ui.section, then by ui.group, respecting ui.order
     */
    function buildSchemaDrivenUI() {
        // Clear existing content
        predictionOptionsContainer.innerHTML = '';
        selectedModelIds.clear();
        
        // Filter to only enabled models
        const enabledModels = MODEL_REGISTRY.models.filter(model => model.enabled === true);
        
        if (enabledModels.length === 0) {
            zeroState.style.display = 'block';
            return;
        }
        
        zeroState.style.display = 'none';
        
        // Group models by section
        const sectionsMap = new Map();
        
        enabledModels.forEach(model => {
            const sectionName = model.ui?.section || 'Other';
            const groupName = model.ui?.group || 'Other';
            const order = model.ui?.order || 999;
            
            if (!sectionsMap.has(sectionName)) {
                sectionsMap.set(sectionName, new Map());
            }
            
            const groupsMap = sectionsMap.get(sectionName);
            if (!groupsMap.has(groupName)) {
                groupsMap.set(groupName, []);
            }
            
            groupsMap.get(groupName).push({ model, order });
        });
        
        // Sort sections (we'll use the first model's order in each section as section order)
        const sections = Array.from(sectionsMap.entries()).map(([sectionName, groupsMap]) => {
            // Get minimum order from all groups in this section
            let minOrder = Infinity;
            groupsMap.forEach((models) => {
                models.forEach(({ order }) => {
                    if (order < minOrder) minOrder = order;
                });
            });
            
            return { sectionName, groupsMap, order: minOrder };
        }).sort((a, b) => a.order - b.order);
        
        // Special handling: Normalize SIDER section name and merge if needed
        const normalizedSections = [];
        const siderSections = [];
        
        sections.forEach(({ sectionName, groupsMap, order }) => {
            const isSIDER = sectionName === 'Adverse Effects (Side Effects)' || 
                          sectionName.includes('Adverse Effects') ||
                          sectionName.includes('Side Effects');
            
            if (isSIDER) {
                // Collect all SIDER groups
                siderSections.push({ sectionName, groupsMap, order });
            } else {
                normalizedSections.push({ sectionName, groupsMap, order });
            }
        });
        
        // Merge all SIDER sections into one "Adverse Effects" section
        if (siderSections.length > 0) {
            const mergedSIDERGroups = new Map();
            let minSIDEROrder = Infinity;
            
            siderSections.forEach(({ groupsMap, order }) => {
                if (order < minSIDEROrder) minSIDEROrder = order;
                groupsMap.forEach((models, groupName) => {
                    if (!mergedSIDERGroups.has(groupName)) {
                        mergedSIDERGroups.set(groupName, []);
                    }
                    mergedSIDERGroups.get(groupName).push(...models);
                });
            });
            
            // Sort groups within merged SIDER section
            mergedSIDERGroups.forEach((models, groupName) => {
                models.sort((a, b) => a.order - b.order);
            });
            
            normalizedSections.push({
                sectionName: 'Adverse Effects',
                groupsMap: mergedSIDERGroups,
                order: minSIDEROrder
            });
        }
        
        // Sort all sections by order
        normalizedSections.sort((a, b) => a.order - b.order);
        
        // Build accordion sections
        normalizedSections.forEach(({ sectionName, groupsMap }) => {
            const isSIDER = sectionName === 'Adverse Effects';
            const sectionElement = createAccordionSection(sectionName, groupsMap, isSIDER);
            predictionOptionsContainer.appendChild(sectionElement);
        });
        
        // Add event listeners for all checkboxes
        attachCheckboxListeners();
    }
    
    /**
     * Create an accordion section with groups
     */
    function createAccordionSection(sectionName, groupsMap, isSIDER = false) {
        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'accordion-section';
        
        // Section header
        const sectionHeader = document.createElement('div');
        sectionHeader.className = 'accordion-section-header';
        sectionHeader.innerHTML = `
            <span class="accordion-section-title">${sectionName}</span>
            <span class="accordion-icon">▼</span>
        `;
        
        // Section content
        const sectionContent = document.createElement('div');
        sectionContent.className = 'accordion-section-content';
        
        // Sort groups by order
        const groups = Array.from(groupsMap.entries()).map(([groupName, models]) => {
            // Sort models within group by order
            models.sort((a, b) => a.order - b.order);
            return { groupName, models };
        }).sort((a, b) => {
            // Sort groups by minimum order in group
            const aMin = Math.min(...a.models.map(m => m.order));
            const bMin = Math.min(...b.models.map(m => m.order));
            return aMin - bMin;
        });
        
        // Build groups
        groups.forEach(({ groupName, models }) => {
            const groupDiv = createGroup(groupName, models, isSIDER);
            sectionContent.appendChild(groupDiv);
        });
        
        // Toggle section on header click
        sectionHeader.addEventListener('click', () => {
            sectionDiv.classList.toggle('expanded');
            const icon = sectionHeader.querySelector('.accordion-icon');
            icon.textContent = sectionDiv.classList.contains('expanded') ? '▲' : '▼';
        });
        
        // All sections start collapsed by default
        sectionDiv.classList.remove('expanded');
        
        sectionDiv.appendChild(sectionHeader);
        sectionDiv.appendChild(sectionContent);
        
        return sectionDiv;
    }
    
    /**
     * Create a group with toggle rows
     * For all sections (including SIDER), show models directly without group-level collapsibility
     */
    function createGroup(groupName, models, isSIDER = false) {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'prediction-group';
        
        // For all sections, just add toggle rows directly (one per model)
        // No special collapsible groups - models are shown directly when section is expanded
        models.forEach(({ model }) => {
            const toggleRow = createToggleRow(model);
            groupDiv.appendChild(toggleRow);
        });
        
        return groupDiv;
    }
    
    /**
     * Format property name for display
     * Normalizes property names to be human-readable
     * This serves as a safeguard to ensure consistent display even if data has minor inconsistencies
     */
    function formatPropertyName(propertyName) {
        if (!propertyName) return '';
        
        // If name contains underscores, it needs formatting
        if (propertyName.includes('_')) {
            // Replace underscores with spaces
            let formatted = propertyName.replace(/_/g, ' ');
            
            // Split by spaces and capitalize each word properly
            const words = formatted.split(' ');
            formatted = words.map(word => {
                // If word is all uppercase and short (like "FDA", "CT", "HIV", "BBB"), keep it uppercase
                if (word === word.toUpperCase() && word.length <= 5) {
                    return word;
                }
                // Otherwise, capitalize first letter, lowercase rest
                return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
            }).join(' ');
            
            return formatted;
        }
        
        // If name is all uppercase (like "CT_TOX" without underscore after our JSON fix), format it
        if (propertyName === propertyName.toUpperCase() && propertyName.length > 3) {
            // Split by spaces if any, otherwise treat as single word
            const words = propertyName.split(' ');
            return words.map(word => {
                if (word.length <= 5) {
                    return word; // Keep short acronyms uppercase
                }
                return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
            }).join(' ');
        }
        
        // Already properly formatted, return as-is
        return propertyName;
    }
    
    /**
     * Create a toggle row for a model (shows all properties of the model)
     */
    function createToggleRow(model) {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'toggle-row';
        
        const label = document.createElement('label');
        label.className = 'toggle-label';
        label.title = model.description || '';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = model.id;
        checkbox.dataset.modelId = model.id;
        
        // Dataset badge
        const datasetBadge = model.dataset?.id || '';
        
        // Build label text with all property names (formatted for display)
        const propertyNames = model.properties.map(p => formatPropertyName(p.name)).join(', ');
        
        const labelText = document.createElement('span');
        labelText.className = 'toggle-label-text';
        labelText.innerHTML = `
            ${propertyNames}
            ${datasetBadge ? `<span class="dataset-badge">${datasetBadge}</span>` : ''}
        `;
        
        label.appendChild(checkbox);
        label.appendChild(labelText);
        rowDiv.appendChild(label);
        
        return rowDiv;
    }
    
    /**
     * Attach event listeners to all checkboxes
     */
    function attachCheckboxListeners() {
        const checkboxes = document.querySelectorAll('input[type="checkbox"][data-model-id]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                updateModelSelection(checkbox);
            });
        });
    }
    
    /**
     * Update model selection based on checkbox state
     * This is called both from event listeners and programmatic changes
     */
    function updateModelSelection(checkbox) {
        const modelId = checkbox.dataset.modelId;
        
        if (checkbox.checked) {
            selectedModelIds.add(modelId);
        } else {
            selectedModelIds.delete(modelId);
        }
        
        updateSelectionSummary();
        updateRunButton();
        checkAvailability();
    }
    
    /**
     * Build presets from registry
     */
    function buildPresets() {
        presetsButtons.innerHTML = '';
        
        // Collect all unique presets from enabled models
        const presetsSet = new Set();
        MODEL_REGISTRY.models.forEach(model => {
            if (model.enabled && model.presets) {
                model.presets.forEach(preset => presetsSet.add(preset));
            }
        });
        
        // Add "Full Profile" preset (selects all enabled models)
        const fullProfileBtn = document.createElement('button');
        fullProfileBtn.className = 'preset-btn';
        fullProfileBtn.textContent = 'Full Profile';
        fullProfileBtn.dataset.preset = 'full_profile';
        fullProfileBtn.addEventListener('click', () => selectPreset('full_profile'));
        presetsButtons.appendChild(fullProfileBtn);
        
        // Add other presets
        const presets = Array.from(presetsSet).sort();
        presets.forEach(preset => {
            const presetBtn = document.createElement('button');
            presetBtn.className = 'preset-btn';
            presetBtn.textContent = formatPresetName(preset);
            presetBtn.dataset.preset = preset;
            presetBtn.addEventListener('click', () => selectPreset(preset));
            presetsButtons.appendChild(presetBtn);
        });
    }
    
    /**
     * Format preset name for display
     */
    function formatPresetName(preset) {
        // Special handling for common abbreviations
        const abbreviations = {
            'adme': 'Absorption, Distribution, Metabolism, Excretion'
        };
        
        const lowerPreset = preset.toLowerCase();
        if (abbreviations[lowerPreset]) {
            return abbreviations[lowerPreset];
        }
        
        // Default formatting: capitalize words separated by underscores
        return preset
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    /**
     * Select a preset
     */
    function selectPreset(presetName) {
        // Clear all selections first
        selectedModelIds.clear();
        document.querySelectorAll('input[type="checkbox"][data-model-id]').forEach(cb => {
            cb.checked = false;
        });
        
        // Collect models to select
        const modelsToSelect = [];
        
        if (presetName === 'full_profile') {
            // Select all enabled models
            MODEL_REGISTRY.models.forEach(model => {
                if (model.enabled) {
                    modelsToSelect.push(model.id);
                }
            });
        } else {
            // Select models with this preset
            MODEL_REGISTRY.models.forEach(model => {
                if (model.enabled && model.presets && model.presets.includes(presetName)) {
                    modelsToSelect.push(model.id);
                }
            });
        }
        
        // Update checkboxes and selection state
        // First, add all model IDs to the Set
        modelsToSelect.forEach(modelId => {
            selectedModelIds.add(modelId);
        });
        
        // Then, update all checkboxes (this ensures consistency)
        modelsToSelect.forEach(modelId => {
            document.querySelectorAll(`input[data-model-id="${modelId}"]`).forEach(cb => {
                cb.checked = true;
            });
        });
        
        // Update UI once after all selections
        updateSelectionSummary();
        updateRunButton();
        checkAvailability();
    }
    
    /**
     * Update selection summary
     */
    function updateSelectionSummary() {
        const count = selectedModelIds.size;
        selectionCount.textContent = count;
        
        if (count === 0) {
            selectionSummary.style.display = 'none';
        } else {
            selectionSummary.style.display = 'block';
        }
    }
    
    /**
     * Update Run Predictions button state
     */
    function updateRunButton() {
        const hasEnabledSelections = selectedModelIds.size > 0;
        runPredictionsBtn.disabled = !hasEnabledSelections;
    }
    
    /**
     * Check availability and show message if needed
     */
    function checkAvailability() {
        // Get selected model IDs
        const selected = Array.from(selectedModelIds);
        
        // Check which are actually enabled
        const enabledSelected = selected.filter(modelId => {
            const model = MODEL_REGISTRY.models.find(m => m.id === modelId);
            return model && model.enabled;
        });
        
        const unavailable = selected.length - enabledSelected.length;
        
        if (unavailable > 0) {
            availabilityMessage.textContent = 
                `${enabledSelected.length} predictions will run, ${unavailable} unavailable`;
            availabilityMessage.style.display = 'block';
        } else {
            availabilityMessage.style.display = 'none';
        }
    }
    
    /**
     * Get selected properties and their models
     * Returns all properties from all selected enabled models
     */
    function getSelectedProperties() {
        const selected = [];
        
        selectedModelIds.forEach(modelId => {
            const model = MODEL_REGISTRY.models.find(m => m.id === modelId);
            if (model && model.enabled) {
                // Include all properties from this model
                model.properties.forEach(property => {
                    selected.push({
                        propertyId: property.id,
                        modelId: model.id
                    });
                });
            }
        });
        
        return selected;
    }
    
    // Form submission
    moleculeForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const inputValue = moleculeInput.value.trim();
        if (!inputValue) return;
        
        // Get selected properties and their models
        const selectedProperties = getSelectedProperties();
        if (selectedProperties.length === 0) {
            alert('Please select at least one property to predict');
            return;
        }
        
        // Get unique model IDs needed for these properties (only enabled ones)
        const modelIds = [...new Set(selectedProperties.map(prop => prop.modelId))];
        
        // Hide about section, show results section
        aboutSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        // Clear previous results
        predictionResults.innerHTML = '';
        moleculeDetails.innerHTML = '';
        
        // Create loading cards for each selected property
        selectedProperties.forEach(prop => {
            createPropertyCard(prop.propertyId, prop.modelId, null, true);
        });
        
        // Add molecule info (will be populated later)
        moleculeDetails.innerHTML = `
            <div class="molecule-loading">Loading molecule information...</div>
        `;
        
        try {
            // Send API request to the unified endpoint
            const response = await fetch(`${API_PREFIX}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: inputValue,
                    model_ids: modelIds
                })
            });
            
            // Handle response
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An error occurred');
            }
            
            const predictionData = await response.json();
            
            // Update molecule details
            updateMoleculeDetails(
                predictionData.input_query, 
                predictionData.input_query !== predictionData.smiles ? 'Compound name' : 'SMILES string', 
                predictionData.smiles
            );
            
            // Process predictions for each model
            predictionData.predictions.forEach(modelPrediction => {
                // Process each property prediction
                modelPrediction.properties.forEach(propertyPrediction => {
                    // Find the property in our registry
                    const propertyConfig = findPropertyConfig(
                        modelPrediction.model_id, 
                        propertyPrediction.property_id
                    );
                    
                    // Update the property card
                    updatePropertyCard(
                        propertyPrediction.property_id,
                        modelPrediction.model_id,
                        propertyPrediction.value,
                        propertyConfig
                    );
                });
            });
            
        } catch (error) {
            // Handle errors
            moleculeDetails.innerHTML = `
                <div class="error-message">
                    <strong>Error:</strong> ${error.message}
                </div>
            `;
            
            // Update cards to show error
            selectedProperties.forEach(prop => {
                updatePropertyCard(prop.propertyId, prop.modelId, null, null, error.message);
            });
        }
    });
    
    // Back button event
    backButton.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        aboutSection.classList.remove('hidden');
        moleculeInput.value = '';
    });
    
    // Find property configuration in registry
    function findPropertyConfig(modelId, propertyId) {
        const model = MODEL_REGISTRY.models.find(m => m.id === modelId);
        if (!model) return null;
        
        return model.properties.find(p => p.id === propertyId);
    }
    
    // Create a property card for the prediction
    function createPropertyCard(propertyId, modelId, value = null, isLoading = false) {
        // Find the property in the registry
        const propertyConfig = findPropertyConfig(modelId, propertyId);
        if (!propertyConfig) return null;
        
        const cardHtml = `
            <div id="card-${propertyId}" class="prediction-card ${isLoading ? 'loading' : ''}">
                <div class="prediction-header">
                    <h3 class="prediction-title">${propertyConfig.name}</h3>
                    <span class="prediction-badge">${propertyConfig.badge}</span>
                </div>
                <div class="prediction-content">
                    ${value !== null ? 
                        `<div class="prediction-value">
                            ${value.toFixed(4)}
                            <span class="prediction-unit">${propertyConfig.unit}</span>
                        </div>
                        <div class="prediction-interpretation" style="color: ${interpretValue(propertyConfig, value).color}">
                            ${interpretValue(propertyConfig, value).text}
                        </div>`
                        : 
                        '<div class="prediction-loading">Predicting...</div>'
                    }
                </div>
            </div>
        `;
        
        predictionResults.insertAdjacentHTML('beforeend', cardHtml);
        return document.getElementById(`card-${propertyId}`);
    }
    
    // Update an existing property card with values
    function updatePropertyCard(propertyId, modelId, value, propertyConfig, errorMessage = null) {
        const card = document.getElementById(`card-${propertyId}`);
        if (!card) return;
        
        // Remove loading state
        card.classList.remove('loading');
        
        if (errorMessage) {
            // Show error in card
            card.querySelector('.prediction-content').innerHTML = `
                <div class="prediction-error" style="color: #f44336;">
                    <strong>Error:</strong> Could not predict value
                </div>
            `;
        } else if (value !== null && propertyConfig) {
            // Update with prediction value
            const interpretation = interpretValue(propertyConfig, value);
            
            card.querySelector('.prediction-content').innerHTML = `
                <div class="prediction-value">
                    ${value.toFixed(4)}
                    <span class="prediction-unit">${propertyConfig.unit}</span>
                </div>
                <div class="prediction-interpretation" style="color: ${interpretation.color}">
                    ${interpretation.text}
                </div>
            `;
        }
    }
    
    // Update molecule details section
    async function updateMoleculeDetails(input, inputType, resolvedSmiles) {
        // Create two-column layout for desktop (info on left, visualization on right)
        // On mobile, visualization will appear below due to flex-wrap
        moleculeDetails.innerHTML = `
            <div class="molecule-info-columns">
                <div class="molecule-info-column">
                    <div class="molecule-details-row">
                        <div class="molecule-details-label">Input:</div>
                        <div class="molecule-details-value">${input}</div>
                    </div>
                    <div class="molecule-details-row">
                        <div class="molecule-details-label">Input Type:</div>
                        <div class="molecule-details-value">${inputType}</div>
                    </div>
                    <div class="molecule-details-row">
                        <div class="molecule-details-label">SMILES:</div>
                        <div class="molecule-details-value">${resolvedSmiles}</div>
                    </div>
                </div>
                <div class="molecule-info-column">
                    <div class="molecule-3d-container">
                        <div class="molecule-3d-loading">Loading 3D visualization...</div>
                    </div>
                </div>
            </div>
        `;
        
        // Load 3D visualization
        try {
            const vizResponse = await fetch(`${API_PREFIX}/visualize-3d?smiles=${encodeURIComponent(resolvedSmiles)}`);
            if (vizResponse.ok) {
                const vizData = await vizResponse.json();
                const vizContainer = moleculeDetails.querySelector('.molecule-3d-container');
                if (vizContainer) {
                    // Use insertAdjacentHTML to ensure scripts execute
                    vizContainer.innerHTML = '';
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = vizData.html;
                    
                    // Move all nodes (including script tags) to the container
                    while (tempDiv.firstChild) {
                        vizContainer.appendChild(tempDiv.firstChild);
                    }
                    
                    // Execute any script tags manually (for browsers that don't auto-execute)
                    const scripts = vizContainer.querySelectorAll('script');
                    scripts.forEach(oldScript => {
                        const newScript = document.createElement('script');
                        Array.from(oldScript.attributes).forEach(attr => {
                            newScript.setAttribute(attr.name, attr.value);
                        });
                        newScript.appendChild(document.createTextNode(oldScript.innerHTML));
                        oldScript.parentNode.replaceChild(newScript, oldScript);
                    });
                }
            } else {
                // If visualization fails, just remove the loading message
                const vizContainer = moleculeDetails.querySelector('.molecule-3d-container');
                if (vizContainer) {
                    vizContainer.innerHTML = '<div class="molecule-3d-error">3D visualization unavailable</div>';
                }
            }
        } catch (error) {
            console.error("Error loading 3D visualization:", error);
            const vizContainer = moleculeDetails.querySelector('.molecule-3d-container');
            if (vizContainer) {
                vizContainer.innerHTML = '<div class="molecule-3d-error">3D visualization unavailable</div>';
            }
        }
    }
    
    // Interpret a predicted value based on property configuration
    function interpretValue(propertyConfig, value) {
        if (!propertyConfig || !propertyConfig.interpretations) {
            return { text: 'No interpretation available', color: '#ffffff' };
        }
        
        // Find the right interpretation based on thresholds
        const sortedInterpretations = [...propertyConfig.interpretations].sort((a, b) => b.threshold - a.threshold);
        
        for (const interpretation of sortedInterpretations) {
            if (value > interpretation.threshold) {
                return { text: interpretation.text, color: interpretation.color };
            }
        }
        
        return { text: 'No interpretation available', color: '#ffffff' };
    }
    
    // Initialize the application
    initializeApp();
});
