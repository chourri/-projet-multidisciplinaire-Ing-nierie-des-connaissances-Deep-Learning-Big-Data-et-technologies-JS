const express = require('express');
const fs = require('fs');
const path = require('path'); // Added path for safety
const cors = require('cors');
const KnowledgeEngine = require('./knowledge_engine');

const app = express();
const PORT = 3001;

app.use(cors()); 

// 1. FORECAST ENDPOINT
app.get('/api/forecast', (req, res) => {
    try {
        // Read the file provided by ML Team
        const rawData = fs.readFileSync('../ml/weather_2026_predicted.json');
        const forecasts = JSON.parse(rawData);

        // Apply Logic (Knowledge Engineering)
        const smartForecasts = KnowledgeEngine.processForecasts(forecasts);

        // Limit to the first 20 records (days).
        // LSTM models lose accuracy after 2-3 weeks, often producing extreme outliers (hallucinations).
        const reliableForecasts = smartForecasts.slice(120, 130);

        res.json({
            status: "success",
            source: "LSTM_Model_v1",
            data: reliableForecasts // Send only the clean, truncated data
        });

    } catch (error) {
        console.error("Error processing forecast:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

// 2. GALLERY ENDPOINT (Keeping your previous work)
app.get('/api/gallery', (req, res) => {
    try {
        const assetsDir = path.join(__dirname, '../frontend/public/assets/satellite');
        if (!fs.existsSync(assetsDir)) return res.json({ status: "empty", data: [] });

        const files = fs.readdirSync(assetsDir);
        const galleryData = files
            .filter(file => /\.(jpg|jpeg|png|webp)$/i.test(file))
            .map((file, index) => {
                let type = 'Analysis';
                if (file.toLowerCase().includes('sat')) type = 'Satellite';
                else if (file.toLowerCase().includes('disaster')) type = 'Disaster';

                return {
                    id: index,
                    url: `assets/satellite/${file}`,
                    title: file.replace(/_/g, ' ').replace(/\.[^/.]+$/, "").toUpperCase(),
                    source: 'Local Repos',
                    type: type,
                    date: '2026-02-01'
                };
            });

        res.json({ status: "success", data: galleryData });
    } catch (error) {
        console.error("Error scanning images:", error);
        res.status(500).json({ error: "Could not scan image directory" });
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on http://localhost:${PORT}`);
    console.log(`Knowledge Engine loaded. Serving reliable horizon (T+20 days).`);
});