// The "Rules" are defined as constants
const RULES = {
    // Heat Rules
    TEMP_CRITICAL: 42.0,
    TEMP_HIGH: 38.0,
    HUMIDITY_DANGER: 20.0,

    // NEW: Wind Rules (km/h)
    WIND_GALE: 60.0,    // Level: HIGH
    WIND_STORM: 85.0    // Level: CRITICAL
};

class KnowledgeEngine {

    /**
     * Rule 1: Analyze a single day for immediate risk
     * @param {Object} dayData - { tmax, rhum, wspd, date }
     * @returns {Object|null} - Alert object or null
     */
    static checkDailyRisk(dayData) {
        // --- EXISTING HEAT LOGIC ---
        if (dayData.tmax >= RULES.TEMP_CRITICAL) {
            return {
                type: "HEATWAVE_CRITICAL",
                level: "CRITICAL",
                message: `Critical temperature detected: ${dayData.tmax}°C exceeds safety limit.`,
                date: dayData.date
            };
        }

        if (dayData.tmax >= RULES.TEMP_HIGH && dayData.rhum < RULES.HUMIDITY_DANGER) {
            return {
                type: "DRY_HEAT_RISK",
                level: "HIGH",
                message: `Dangerous dry heat: ${dayData.tmax}°C with only ${dayData.rhum}% humidity.`,
                date: dayData.date
            };
        }

        // --- NEW: WIND & STORM LOGIC ---

        // 1. Severe Storm Rule (Highest Priority)
        // Checks for dangerous wind speeds usually associated with damage
        if (dayData.wspd >= RULES.WIND_STORM) {
            return {
                type: "SEVERE_STORM_EVENT",
                level: "CRITICAL", // This triggers the RED badge in your table
                message: `Storm force winds detected: ${dayData.wspd} km/h. Structure damage possible.`,
                date: dayData.date
            };
        }

        // 2. Gale/Strong Wind Rule
        // Checks for high winds that require agricultural advisory
        if (dayData.wspd >= RULES.WIND_GALE) {
            return {
                type: "STRONG_WIND_ADVISORY",
                level: "HIGH", // This triggers the ORANGE badge in your table
                message: `Gale force winds: ${dayData.wspd} km/h. Crop protection advised.`,
                date: dayData.date
            };
        }

        return null; // No alert
    }

    /**
     * Rule 2: Analyze history for persistent patterns
     */
    static checkPersistentRisk(historyBuffer) {
        if (historyBuffer.length < 3) return null;
        const last3Days = historyBuffer.slice(-3);

        // Heat Persistence
        const isPersistentHeat = last3Days.every(day => day.tmax > 40.0);
        if (isPersistentHeat) {
            return {
                type: "PERSISTENT_HEATWAVE",
                level: "EXTREME",
                message: "Extreme Heatwave: Temperature > 40°C for 3 consecutive days.",
                date: last3Days[last3Days.length - 1].date
            };
        }

        // NEW: Storm Persistence (e.g., Multi-day storm front)
        // If wind stays high (> 50km/h) for 3 days straight
        const isPersistentStorm = last3Days.every(day => day.wspd > 50.0);
        if (isPersistentStorm) {
             return {
                type: "LONG_DURATION_STORM",
                level: "HIGH",
                message: "Persistent Storm Front: High winds > 50km/h sustaining for 72h.",
                date: last3Days[last3Days.length - 1].date
            };
        }

        return null;
    }

    static processForecasts(forecastList) {
        const processedData = [];
        const historyBuffer = [];

        forecastList.forEach(day => {
            const enrichedDay = { ...day, alert: null };

            // 1. Check Daily Rules (Heat + Wind)
            const dailyAlert = this.checkDailyRisk(day);
            if (dailyAlert) {
                enrichedDay.alert = dailyAlert;
            }

            // 2. Check Persistent Rules
            historyBuffer.push(day);
            if (historyBuffer.length > 3) historyBuffer.shift(); 

            const persistentAlert = this.checkPersistentRisk(historyBuffer);
            
            // Persistent alerts usually override daily ones because they are more severe
            if (persistentAlert) {
                enrichedDay.alert = persistentAlert; 
            }

            processedData.push(enrichedDay);
        });

        return processedData;
    }
}

module.exports = KnowledgeEngine;