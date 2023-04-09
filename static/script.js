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
    const propertyCheckboxes = document.querySelector('.checkbox-group');
    
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
            
            // Populate property checkboxes based on available models
            updatePropertyCheckboxes();
            
        } catch (error) {
            console.error("Error initializing app:", error);
            alert("Failed to initialize application. Please reload the page or contact support.");
        }
    }
    
    // Update property checkboxes based on available models
    function updatePropertyCheckboxes() {
        // Clear existing checkboxes
        propertyCheckboxes.innerHTML = '';
        
        // Add checkboxes for each property from each model
        MODEL_REGISTRY.models.forEach(model => {
            // Skip disabled models
            if (!model.enabled) return;
            
            model.properties.forEach(property => {
                const checkboxHtml = `
                    <label class="checkbox-label" title="${model.description}">
                        <input type="checkbox" value="${property.id}" data-model="${model.id}" ${model.id === "esol_solubility" ? "checked" : ""}>
                        ${property.name} <span class="badge">${property.badge}</span>
                    </label>
                `;
                propertyCheckboxes.insertAdjacentHTML('beforeend', checkboxHtml);
            });
        });
    }
    
    // Form submission
    moleculeForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const inputValue = moleculeInput.value.trim();
        if (!inputValue) return;
        
        // Hide about section, show results section
        aboutSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        // Clear previous results
        predictionResults.innerHTML = '';
        moleculeDetails.innerHTML = '';
        
        // Get selected properties and their models
        const selectedProperties = getSelectedProperties();
        if (selectedProperties.length === 0) {
            alert('Please select at least one property to predict');
            return;
        }
        
        // Get unique model IDs needed for these properties
        const modelIds = [...new Set(selectedProperties.map(prop => prop.modelId))];
        
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
    
    // Helper function to get selected properties
    function getSelectedProperties() {
        const checkboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]:checked:not(:disabled)');
        return Array.from(checkboxes).map(checkbox => ({
            propertyId: checkbox.value,
            modelId: checkbox.dataset.model
        }));
    }
    
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
    function updateMoleculeDetails(input, inputType, resolvedSmiles) {
        moleculeDetails.innerHTML = `
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
        `;
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